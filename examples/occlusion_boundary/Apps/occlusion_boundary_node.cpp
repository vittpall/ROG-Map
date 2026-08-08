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
 * Only (1) warrants a keep-out. Two discriminators are applied in series, cheap
 * one first:
 *
 *   a) attachment to an occluder -- a frontier voxel with an OCCUPIED voxel
 *      within `attach_radius_vox` cells is shadow-cast; one floating in free
 *      space is a range/FOV artefact. O(r^3) on a tiny r, so it runs on every
 *      candidate.
 *
 *   b) NOT VISIBLE FROM THE EGO -- see isHiddenFromEgo(). Attachment alone is
 *      not sufficient and fails in a way that ruins the keep-out: a wall voxel
 *      that simply never received a lidar return stays UNKNOWN (one hit is
 *      enough to make a voxel OCCUPIED, and one traversing ray is enough to
 *      make it FREE, so "unknown" means zero rays touched it). That pinhole sits
 *      against the free air the lidar carved in front of the wall, so
 *      isFrontier() accepts it, and it is buried in wall so (a) accepts it too.
 *      Every hole in a wall face then grows its own keep-out bubble and the
 *      whole wall ends up upholstered in them.
 *
 *      The physical property that separates the two cases is visibility: a
 *      pinhole is one the drone can see straight into, so nothing can be hiding
 *      in it. A real shadow voxel is by definition one the drone CANNOT see.
 *
 * Note (b) subsumes most of what (a) does -- a range/FOV frontier is reached by
 * an unobstructed ray and is rejected as visible. (a) is kept because it is much
 * cheaper and disposes of the bulk of the candidates before any ray marching.
 *
 * Both classes are published separately, on purpose: seeing the REJECTED set is
 * what tells you whether the discriminators are doing their job or quietly
 * eating real boundaries.
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
    // Second discriminator: reject frontier voxels the ego can see (see the
    // header). Off restores the old attachment-only behaviour, which is worth
    // having to confirm this filter is what changed a result.
    nh_.param<bool>("sightline_check", sightline_check_, true);
    // Ray marching is the only per-candidate cost here that is not O(1), so it is
    // bounded. A shadow voxel finds its occluder within a step or two; only a
    // VISIBLE voxel marches the full way, and running out of steps is treated as
    // visible -- the conservative direction is the other one (keeping a boundary),
    // but an unbounded march on a 5 Hz timer is worse than a missed voxel.
    nh_.param<double>("sightline_max_range", sightline_max_range_, 15.0);
    // Start the march this far from the voxel so its own cell, and the occluder
    // face it is pressed against, are not what blocks the ray.
    nh_.param<double>("sightline_skip", sightline_skip_, 0.0);
    nh_.param<double>("rate", rate_, 5.0);

    pub_occlusion_ = nh_.advertise<sensor_msgs::PointCloud2>("occlusion_frontier", 1);
    pub_open_ = nh_.advertise<sensor_msgs::PointCloud2>("open_frontier", 1);

    sub_odom_ = nh_.subscribe(odom_topic_, 10,
                              &OcclusionBoundaryExtractor::odomCallback, this);

    timer_ = nh_.createTimer(ros::Duration(1.0 / rate_),
                             &OcclusionBoundaryExtractor::update, this);

    ROS_INFO("[occlusion_boundary] odom=%s query=%.1fm (z %.1fm) attach_radius=%d vox "
             "sightline=%s (max %.1fm)",
             odom_topic_.c_str(), query_range_, query_range_z_, attach_radius_vox_,
             sightline_check_ ? "on" : "OFF", sightline_max_range_);
    if (!sightline_check_) {
      ROS_WARN("[occlusion_boundary] sightline_check DISABLED -- every unobserved "
               "hole in a wall face will be published as an occlusion boundary");
    }
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

  /* True if the straight line from `pos` to the ego is blocked by an OCCUPIED
   * voxel -- i.e. the drone cannot see `pos`, so an agent could be hiding there.
   *
   * This is what separates a genuine shadow voxel from a pinhole in an observed
   * surface. Both are UNKNOWN, both touch free space, both are buried in
   * occupied neighbours, so no neighbourhood COUNT tells them apart: a pinhole
   * in a wall 4 voxels thick has ~9 free neighbours in front, ~8 occupied around
   * it and ~9 unknown behind it (the wall's own unobserved interior), which is
   * indistinguishable from a shadow mouth by counting alone. Direction is the
   * missing information -- the pinhole's neighbour TOWARD the ego is free, the
   * shadow voxel's is not.
   *
   * Marched rather than tested one step out because a single sample is fooled
   * wherever the sightline runs nearly tangent to a surface: the step lands in a
   * neighbouring free cell while the ray is genuinely blocked further along.
   *
   * Stepping in metres, at half the map resolution, for the same reason
   * attachedToOccluder() does: only the Vec3f query overloads are public, and
   * half-resolution steps cannot skip over a one-voxel-thick occluder. */
  bool isHiddenFromEgo(const Vec3f& pos, const Vec3f& ego) const {
    const Vec3f delta = ego - pos;
    const double dist = delta.norm();
    if (dist < 1e-6) return false;

    const double step = map_->getResolution() * 0.5;
    const double reach = std::min(dist, sightline_max_range_);
    const Vec3f dir = delta / dist;

    // Start one step out: the voxel's own cell is UNKNOWN by construction, so
    // sampling it proves nothing. Stop before the ego, whose cell is free anyway.
    for (double s = std::max(step, sightline_skip_); s < reach; s += step) {
      // Materialised into a Vec3f rather than passed as an Eigen expression:
      // isOccupied() is overloaded on Vec3f and Vec3i, an unevaluated CwiseBinaryOp
      // converts to both, and the call is ambiguous.
      const Vec3f sample = pos + dir * s;
      if (map_->isOccupied(sample)) {
        return true;      // something solid between the ego and this voxel
      }
    }
    return false;         // unobstructed line of sight: nothing can hide here
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
    size_t n_visible = 0;   // attached to an occluder, but the ego can see it

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
      // Attached, but visible => a hole in an observed surface, not a shadow.
      if (sightline_check_ && !isHiddenFromEgo(p, ego)) {
        ++n_visible;
        open_cloud.push_back(pt);
        continue;
      }
      occlusion_cloud.push_back(pt);     // shadow-cast: an agent could hide here
    }

    publish(occlusion_cloud, pub_occlusion_);
    publish(open_cloud, pub_open_);

    if (log_counter_++ % static_cast<int>(rate_ * 2) == 0) {
      // visible= is the diagnostic for this filter: a large count means the map
      // is full of surface pinholes (raise the scene's point density / lower
      // point_filt_num), a zero count with sightline_check on means it is not
      // firing at all.
      ROS_INFO("[occlusion_boundary] unknown=%zu  occlusion=%zu  open=%zu "
               "(visible-rejected=%zu)",
               unknown_pts.size(), occlusion_cloud.size(), open_cloud.size(),
               n_visible);
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

  ros::Publisher pub_occlusion_, pub_open_;
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
  bool sightline_check_{true};
  double sightline_max_range_{15.0};
  double sightline_skip_{0.0};
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
