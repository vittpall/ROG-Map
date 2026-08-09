/* Occlusion-boundary extraction from a ROG-Map occupancy grid.
 *
 * WHY THIS EXISTS
 * ---------------
 * The 2D predecessor of this detector found occlusion corners by walking an
 * ORDERED lidar scan and looking for a range jump between adjacent beams. That
 * requires the cloud to arrive as a dense (n_h, n_v) beam grid: "adjacent beam"
 * has to mean something. A Livox Mid-360 / Avia scans a non-repetitive rosette,
 * so consecutive points are not spatial neighbours and coverage accumulates over
 * time. The jump test has no adjacency to stand on and simply does not port.
 *
 * This node moves the detection from SCAN space into MAP space. It reads the
 * fused occupancy grid, so it is identical for Livox, VLP-32 and OS-128 -- the
 * sensor model is absorbed by the mapper. Slow rosette accumulation stops being
 * a failure mode and becomes ordinary map filling.
 *
 * WHAT IS DETECTED
 * ----------------
 * ROG-Map already computes a frontier: an UNKNOWN voxel adjacent to observed
 * free space (see ProbMap::isFrontier). That is the mouth of the shadow -- the
 * surface an unseen agent must cross to reach the ego. For a forward-reachable
 * -set keep-out that is the correct object, more so than the occluder's own
 * silhouette, because the agent emerges THROUGH the mouth, not out of solid.
 *
 * But raw frontier conflates three causes:
 *   1. occlusion shadows      <- geometry can hide an agent
 *   2. max-range limits       <- nothing is hiding, the sensor just stops
 *   3. FOV edges              <- ditto (very visible on the Mid-360)
 *
 * Only (1) warrants a keep-out, and ONE discriminator separates it:
 *
 *   attachment to an occluder -- a frontier voxel with an OCCUPIED voxel within
 *   `attach_radius_vox` cells is shadow-cast; one floating in free space is a
 *   range/FOV artefact. O(r^3) on a tiny r, so it runs on every candidate.
 *
 * So the full test is: UNKNOWN, touching at least one KNOWN_FREE cell (that is
 * ProbMap::isFrontier), and with an OCCUPIED cell nearby.
 *
 * KNOWN LIMITATION. This admits pinholes: a wall voxel that simply never
 * received a lidar return stays UNKNOWN (one hit makes a voxel OCCUPIED, one
 * traversing ray makes it FREE, so "unknown" means zero rays touched it). Such a
 * hole sits against the free air in front of the wall, so isFrontier() accepts
 * it, and it is buried in wall, so the attachment test accepts it too -- and the
 * wall face fills with keep-out bubbles the drone can see straight into.
 *
 * No neighbourhood COUNT separates those from a real shadow mouth: a pinhole in
 * a wall 4 voxels thick has ~9 free neighbours in front, ~8 occupied around it
 * and ~9 unknown behind, which is what a shadow mouth looks like too. The
 * missing information is DIRECTION -- the pinhole's neighbour toward the ego is
 * free, the shadow voxel's is not -- which is what a visibility ray march
 * supplied before it was removed. Raising the scene's point density (lower
 * point_filt_num) is the mitigation that remains.
 *
 * Both classes are published separately, on purpose: seeing the REJECTED set is
 * what tells you whether the discriminator is doing its job or quietly eating
 * real boundaries.
 *
 * STATUS: visualisation only. Nothing here feeds a controller yet.
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "rog_map/rog_map.h"

using rog_map::Vec3f;

using rog_map::GridType;

class OcclusionBoundaryExtractor {
 public:
  OcclusionBoundaryExtractor(ros::NodeHandle& nh, rog_map::ROGMap::Ptr map)
      : nh_(nh), map_(std::move(map)) {
    nh_.param<std::string>("odom_topic", odom_topic_, "/lidar_slam/odom");
    nh_.param<std::string>("frame_id", frame_id_, "world");
    // Half-extents of the box searched around the ego each tick [m].
    //
    // Cost is brutally sensitive to these: the box is enumerated at map
    // resolution, so volume grows cubically. A 15 m cube at 0.1 m resolution
    // enumerated 5.2M unknown voxels per tick and dragged the node down to
    // ~1.1 Hz against a 5 Hz timer.
    //
    // Z is separate because the local map is only a few metres tall -- asking
    // for +/-15 m of z spends most of the query outside the map entirely.
    nh_.param<double>("query_range", query_range_, 8.0);
    nh_.param<double>("query_range_z", query_range_z_, 2.5);
    // A frontier voxel counts as shadow-cast if an OCCUPIED voxel lies within
    // this many cells (Chebyshev). 1 is strict and drops boundaries whose
    // occluder edge is one voxel thinner than the frontier; 2 is a safer start.
    nh_.param<int>("attach_radius_vox", attach_radius_vox_, 2);
    // Keep every Nth KNOWN_FREE voxel on free_voxels. Free space is the LARGEST
    // class in an observed map -- the query box alone holds ~1.3M cells at 8 m
    // and 0.1 m resolution -- and publishing all of it is what killed RViz on
    // the unknown map. 4 keeps the shape of the visible region at 1/4 the
    // points. 1 publishes everything; expect RViz to struggle.
    nh_.param<int>("free_stride", free_stride_, 4);
    nh_.param<double>("rate", rate_, 5.0);

    pub_occlusion_ = nh_.advertise<sensor_msgs::PointCloud2>("occlusion_frontier", 1);
    pub_open_ = nh_.advertise<sensor_msgs::PointCloud2>("open_frontier", 1);
    pub_free_ = nh_.advertise<sensor_msgs::PointCloud2>("free_voxels", 1);

    sub_odom_ = nh_.subscribe(odom_topic_, 10,
                              &OcclusionBoundaryExtractor::odomCallback, this);

    timer_ = nh_.createTimer(ros::Duration(1.0 / rate_),
                             &OcclusionBoundaryExtractor::update, this);

    ROS_INFO("[occlusion_boundary] odom=%s query=%.1fm (z %.1fm) attach_radius=%d vox",
             odom_topic_.c_str(), query_range_, query_range_z_, attach_radius_vox_);
  }
  
 private:
  void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    std::lock_guard<std::mutex> lk(pose_mutex_);
    ego_ = Vec3f(msg->pose.pose.position.x,
                 msg->pose.pose.position.y,
                 msg->pose.pose.position.z);
    have_pose_ = true;
  }

  /* True if any OCCUPIED voxel sits within attach_radius_vox_ cells of `pos`.
   * This is the range/FOV vs. shadow discriminator described at the top.
   *
   * Stepping in metres rather than grid indices is deliberate: ROG-Map exposes
   * only the Vec3f query overloads publicly (the Vec3i ones are protected), so
   * we walk the neighbourhood in multiples of the map resolution instead. */
  bool attachedToOccluder(const Vec3f& pos) const {
    const int r = attach_radius_vox_;
    const double res = map_->getResolution();
    for (int dx = -r; dx <= r; ++dx) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dz = -r; dz <= r; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const Vec3f n(pos.x() + dx * res, pos.y() + dy * res, pos.z() + dz * res);
          if (map_->isOccupied(n)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void update(const ros::TimerEvent&) {
    Vec3f ego;
    {
      std::lock_guard<std::mutex> lk(pose_mutex_);
      if (!have_pose_) return;
      ego = ego_;
    }

    const Vec3f half(query_range_, query_range_, query_range_z_);
    const Vec3f box_min = ego - half;
    const Vec3f box_max = ego + half;

    // Frontiers are UNKNOWN voxels by construction, so only UNKNOWN needs to be
    // enumerated -- isFrontier() then does the free-neighbour test for us.
    rog_map::vec_E<Vec3f> unknown_pts;
    map_->boxSearch(box_min, box_max, GridType::UNKNOWN, unknown_pts);

    pcl::PointCloud<pcl::PointXYZ> occlusion_cloud, open_cloud;

    for (const auto& p : unknown_pts) {
      if (!map_->isFrontier(p)) continue;

      pcl::PointXYZ pt(static_cast<float>(p.x()),
                       static_cast<float>(p.y()),
                       static_cast<float>(p.z()));
      // Cheap test first: it disposes of most candidates without any marching.
      if (!attachedToOccluder(p)) {
        open_cloud.push_back(pt);        // range / FOV limit: nothing hiding
        continue;
      }
      occlusion_cloud.push_back(pt);     // shadow-cast: an agent could hide here
    }

    publish(occlusion_cloud, pub_occlusion_);
    publish(open_cloud, pub_open_);

    // Observed free space in the same box: what the ego has actually SEEN, as
    // opposed to what it has not (unknown) or what is solid (occupied). Gated on
    // a subscriber because the boxSearch is the expensive part and nothing in
    // the pipeline consumes this -- it is purely for looking at.
    // boxSearch CANNOT serve this: it throws outright on KNOWN_FREE
    // (prob_map.cpp, "Box search does not support KNOWN_FREE") because free is
    // the default state and is not kept in a searchable list the way occupied,
    // unknown and frontier are. So walk the box and test each cell instead --
    // the same Vec3f-stepping pattern attachedToOccluder() uses, and the reason
    // free_stride matters: it is applied to the WALK, not to the result, so a
    // stride of 4 is 64x fewer isKnownFree() calls, not just fewer points.
    if (pub_free_.getNumSubscribers() > 0) {
      const double step = map_->getResolution() * std::max(1, free_stride_);
      pcl::PointCloud<pcl::PointXYZ> free_cloud;
      for (double x = box_min.x(); x <= box_max.x(); x += step) {
        for (double y = box_min.y(); y <= box_max.y(); y += step) {
          for (double z = box_min.z(); z <= box_max.z(); z += step) {
            const Vec3f p(x, y, z);
            if (map_->isKnownFree(p)) {
              free_cloud.push_back(pcl::PointXYZ(static_cast<float>(x),
                                                 static_cast<float>(y),
                                                 static_cast<float>(z)));
            }
          }
        }
      }
      publish(free_cloud, pub_free_);
    }

    if (log_counter_++ % static_cast<int>(rate_ * 2) == 0) {
      ROS_INFO("[occlusion_boundary] unknown=%zu  occlusion=%zu  open=%zu",
               unknown_pts.size(), occlusion_cloud.size(), open_cloud.size());
    }
  }

  void publish(const pcl::PointCloud<pcl::PointXYZ>& cloud,
               const ros::Publisher& pub) const {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = frame_id_;
    msg.header.stamp = ros::Time::now();
    pub.publish(msg);
  }

  ros::NodeHandle nh_;
  rog_map::ROGMap::Ptr map_;

  ros::Publisher pub_occlusion_, pub_open_, pub_free_;
  int free_stride_{4};
  ros::Subscriber sub_odom_;
  ros::Timer timer_;

  std::mutex pose_mutex_;
  Vec3f ego_{0, 0, 0};
  bool have_pose_{false};

  std::string odom_topic_, frame_id_;
  double query_range_{8.0};
  double query_range_z_{2.5};
  double rate_{5.0};
  int attach_radius_vox_{2};
  int log_counter_{0};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "occlusion_boundary_node");
  ros::NodeHandle nh("~");

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  // ROGMap self-subscribes to the cloud/odom topics named in its own params,
  // exactly as marsim_example_node does.
  rog_map::ROGMap::Ptr map = std::make_shared<rog_map::ROGMap>(nh);

  OcclusionBoundaryExtractor extractor(nh, map);

  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
