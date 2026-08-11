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
 * free, the shadow voxel's is not.
 *
 * A visibility ray march (walk ego -> candidate, reject anything not blocked by
 * an OCCUPIED voxel) is the textbook answer and has now been tried TWICE and
 * removed twice: it did not visibly reduce the wall-face bubbles in the wall
 * test. Before writing it a third time, establish WHY -- the likely culprits are
 * that the map marks the wall's own front face OCCUPIED so the march terminates
 * on it for pinhole and shadow alike, and that the backoff needed to avoid that
 * is the same order as the wall thickness. Raising the scene's point density
 * (lower point_filt_num) is the mitigation that remains.
 *
 * Both classes are published separately, on purpose: seeing the REJECTED set is
 * what tells you whether the discriminator is doing its job or quietly eating
 * real boundaries.
 *
 * THE SILHOUETTE DETECTOR (parallel, independent)
 * ----------------------------------------------
 * Everything above detects the ABSENCE of data: a frontier voxel is one no ray
 * ever touched, which is also the exact definition of a pinhole. That is why no
 * local test separates them and why the ray march was removed twice.
 *
 * publishSilhouette() takes the opposite primitive. It detects the PRESENCE of
 * matter -- an OCCUPIED voxel at the edge of an occluder -- and then CONSTRUCTS
 * the shadow that edge casts, rather than trying to detect the shadow directly.
 * A pinhole is an absent voxel, so it never enters the computation: it cannot
 * lower a bin's range, cannot raise it, and cannot manufacture a discontinuity.
 *
 * The method is the 2D predecessor's range-jump test, recovered. What did not
 * port was the ORDERING (a rosette has no adjacent beams), not the idea -- so
 * the ordering is rebuilt here from the map by binning occupied voxels into
 * azimuth bins and keeping the nearest range in each. A jump between adjacent
 * bins is an occluder edge; the shadow starts there and runs radially away.
 *
 * Runs ENTIRELY in parallel: separate topics, separate parameters, no shared
 * state with the frontier path above. Compare ~silhouette_gates against
 * ~occlusion_frontier in RViz before trusting either.
 *
 * STATUS: visualisation only. Nothing here feeds a controller yet -- but
 * ~silhouette_gates is deliberately shaped as a drop-in for
 * ~occlusion_frontier (sampled at map resolution, same message type), so the
 * MPPI can be pointed at it by changing boundary_topic and nothing else.
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

    // ---- Silhouette detector (parallel path, see header) --------------------
    nh_.param<bool>("silhouette_en", silhouette_en_, true);
    // Azimuth bin width [deg]. There is a hard floor here: a voxel must span at
    // least one bin at the far end of query_range or the profile fragments into
    // empty bins and every gap reads as two spurious edges. At 0.15 m and 8 m a
    // voxel subtends ~1.07 deg, so 1.0 is the smallest honest value for this
    // config. Going finer does not buy resolution, it buys false edges.
    nh_.param<double>("silhouette_dtheta_deg", silhouette_dtheta_deg_, 1.0);
    // Minimum range discontinuity between adjacent bins to count as an edge [m].
    // Below the occluder's own thickness this fires on surface roughness; above
    // the distance from a wall to whatever is behind it, nothing is detected.
    nh_.param<double>("silhouette_jump_thresh", silhouette_jump_thresh_, 0.8);
    // How far behind the edge the gate runs [m] -- the surface a hidden agent
    // must cross. Match v_target * t_grow_max on the planner side (1.0 * 3.0),
    // since beyond that the keep-out has stopped growing anyway.
    nh_.param<double>("silhouette_shadow_depth", silhouette_shadow_depth_, 3.0);
    // Half-height of the slab of occupied voxels folded into the polar profile
    // [m]. This is the planar assumption: with lock_altitude the ego stays at
    // cruise_z and only obstacles near that altitude can occlude it. Too wide
    // and the floor enters every bin, pinning depth to ~0 everywhere.
    nh_.param<double>("silhouette_z_band", silhouette_z_band_, 0.5);
    // Occupied voxels closer than this are ignored [m]. Guards the atan2 from
    // near-zero ranges and stops the ego's own inflated footprint, if it ever
    // clips a voxel, from swamping every bin.
    nh_.param<double>("silhouette_min_range", silhouette_min_range_, 0.3);
    // How many bins past the transition must stay far before an edge is
    // believed. 1 disables the check. This is the guard against grazing
    // incidence, where dr/dtheta along a continuous oblique surface can exceed
    // jump_thresh on its own -- worst precisely when the ego is close to a wall,
    // which is when the detector matters most.
    nh_.param<int>("silhouette_min_run_bins", silhouette_min_run_bins_, 3);

    pub_occlusion_ = nh_.advertise<sensor_msgs::PointCloud2>("occlusion_frontier", 1);
    pub_open_ = nh_.advertise<sensor_msgs::PointCloud2>("open_frontier", 1);
    pub_free_ = nh_.advertise<sensor_msgs::PointCloud2>("free_voxels", 1);
    // The corner points themselves: two per wall, and the cheapest thing to look
    // at when asking "did it find the edge at all".
    pub_sil_edges_ = nh_.advertise<sensor_msgs::PointCloud2>("silhouette_edges", 1);
    // The gates: sampled at map resolution so this is interchangeable with
    // occlusion_frontier as far as the MPPI's KD-tree is concerned.
    pub_sil_gates_ = nh_.advertise<sensor_msgs::PointCloud2>("silhouette_gates", 1);

    sub_odom_ = nh_.subscribe(odom_topic_, 10,
                              &OcclusionBoundaryExtractor::odomCallback, this);

    timer_ = nh_.createTimer(ros::Duration(1.0 / rate_),
                             &OcclusionBoundaryExtractor::update, this);

    ROS_INFO("[occlusion_boundary] odom=%s query=%.1fm (z %.1fm) attach_radius=%d vox",
             odom_topic_.c_str(), query_range_, query_range_z_, attach_radius_vox_);
    if (silhouette_en_) {
      ROS_INFO("[occlusion_boundary] silhouette: dtheta=%.2f deg jump=%.2f m depth=%.2f m "
               "z_band=%.2f m", silhouette_dtheta_deg_, silhouette_jump_thresh_,
               silhouette_shadow_depth_, silhouette_z_band_);
      // NOTE the old warning here checked whether a voxel subtends a bin at
      // MAX range. That was the wrong end: angular splatting makes the profile
      // continuous at every range, and the failure it was meant to catch (a
      // combed profile spraying phantom edges) actually happens in the NEAR
      // field under centre-binning, which splatting removed outright.
      //
      // What is left worth warning about is a bin so coarse that the edge
      // cannot be localised: the gate is placed on a bin boundary, so the
      // angular quantisation is an arc of dtheta * near_range.
      const double arc_at_range = silhouette_dtheta_deg_ * M_PI / 180.0 * query_range_;
      if (arc_at_range > map_->getResolution() * 4.0) {
        ROS_WARN("[occlusion_boundary] silhouette_dtheta_deg (%.2f) puts %.2f m of angular "
                 "quantisation on an edge at query_range: gates will be coarsely placed",
                 silhouette_dtheta_deg_, arc_at_range);
      }
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

  /* Silhouette + shadow-volume extraction. See the header for why this exists
   * alongside the frontier path rather than replacing it.
   *
   * Three passes, no ray marching anywhere:
   *   1. SPLAT the OCCUPIED voxels into azimuth bins by their angular extent,
   *      keeping the nearest range per bin -- this rebuilds the ordered range
   *      profile a rosette cannot give us directly, at every range rather than
   *      at one (see the pass itself for why the extent matters)
   *   2. find adjacent bins whose ranges jump; the nearer one is an occluder
   *      edge and the ego is looking past it
   *   3. emit the gate: the segment running radially outward from that edge,
   *      which is the surface a hidden agent must cross to reach the ego
   *
   * The occupied set is far smaller than the unknown set the frontier path
   * enumerates, so this is the CHEAPER of the two detectors despite being the
   * more robust one. */
  void publishSilhouette(const Vec3f& ego, const Vec3f& box_min, const Vec3f& box_max) {
    constexpr double kNoDepth = std::numeric_limits<double>::infinity();

    const int n_bins = std::max(
        8, static_cast<int>(std::lround(360.0 / std::max(0.05, silhouette_dtheta_deg_))));
    const double dtheta = 2.0 * M_PI / n_bins;

    rog_map::vec_E<Vec3f> occ_pts;
    map_->boxSearch(box_min, box_max, GridType::OCCUPIED, occ_pts);

    // ---- 1. polar range profile, by ANGULAR SPLATTING ----------------------
    // Each voxel is written to every bin its angular EXTENT covers, not just to
    // the bin holding its centre. This is the difference between a profile that
    // works at one range and one that works at all of them.
    //
    // Centre-binning fails in the near field, and badly. Neighbouring wall
    // voxels are `resolution` apart, which subtends resolution/r radians, so
    // they stop landing in adjacent bins as soon as
    //     r < resolution / dtheta        (0.1 / 1 deg = 5.7 m)
    // and below that the profile combs into filled/empty stripes with EVERY gap
    // reading as two discontinuities. Approaching a wall therefore sprayed
    // phantom edges around the nearest corner -- the exact symptom that sent us
    // back here.
    //
    // Splatting is self-correcting because the half-angle grows exactly as the
    // range shrinks: one bin at 8 m, tens of bins at 1 m, continuous at both.
    // It also makes the old single-bin gap-fill unnecessary, so that heuristic
    // is gone -- there are no gaps left for it to paper over.
    const double r_vox = 0.5 * std::sqrt(2.0) * map_->getResolution();  // XY circumradius
    std::vector<double> depth(n_bins, kNoDepth);
    for (const auto& p : occ_pts) {
      if (std::fabs(p.z() - ego.z()) > silhouette_z_band_) continue;
      const double dx = p.x() - ego.x();
      const double dy = p.y() - ego.y();
      const double r = std::hypot(dx, dy);
      if (r < silhouette_min_range_ || r > query_range_) continue;

      const double theta = std::atan2(dy, dx);
      const double half_ang = std::atan2(r_vox, r);
      const int b0 = static_cast<int>(std::floor((theta - half_ang + M_PI) / dtheta));
      const int b1 = static_cast<int>(std::floor((theta + half_ang + M_PI) / dtheta));
      // b1 - b0 is bounded by the half-angle, which is bounded by
      // silhouette_min_range: the loop cannot run away even for a voxel the ego
      // is sitting on top of.
      for (int b = b0; b <= b1; ++b) {
        const int bw = ((b % n_bins) + n_bins) % n_bins;   // wrap, negatives included
        depth[bw] = std::min(depth[bw], r);
      }
    }

    // ---- 2 & 3. discontinuities -> edges -> gates --------------------------
    const double res = map_->getResolution();
    pcl::PointCloud<pcl::PointXYZ> edge_cloud, gate_cloud;

    // A bin is "far" relative to a near surface if it holds nothing at all or
    // sits at least a jump behind it. Shared by the transition test and the
    // run-length confirmation below so the two cannot drift apart.
    const auto is_far = [&](double d, double near_r) {
      return !std::isfinite(d) || (d - near_r) > silhouette_jump_thresh_;
    };

    for (int i = 0; i < n_bins; ++i) {
      const int j = (i + 1) % n_bins;
      const double a = depth[i], b = depth[j];
      if (!std::isfinite(a) && !std::isfinite(b)) continue;   // open sky both sides

      double near_r;
      bool far_is_forward;   // is the shadow on the j side of the transition?
      if (!std::isfinite(a) || !std::isfinite(b)) {
        // Surface on one side, nothing at all on the other: a silhouette
        // against open space, which is the wall-END case in the wall test.
        near_r = std::isfinite(a) ? a : b;
        far_is_forward = std::isfinite(a);
      } else {
        if (std::fabs(a - b) < silhouette_jump_thresh_) continue;
        near_r = std::min(a, b);
        far_is_forward = (a < b);
      }
      // Walk outward from the FIRST far bin, which is j going forward but i
      // going backward -- starting at j either way would test the near bin
      // itself and reject every backward-facing edge.
      const int step = far_is_forward ? +1 : -1;
      const int far_start = far_is_forward ? j : i;

      // Run-length confirmation. A real shadow keeps its far side far for many
      // bins; a one-bin spike does not. Two things produce those spikes and
      // both are worst exactly where the ego is closest to the wall:
      //   - grazing incidence, where dr/dtheta along a continuous oblique
      //     surface legitimately exceeds jump_thresh
      //   - a single missing voxel in the occupied set
      // Requiring the far side to persist costs nothing on a genuine corner,
      // whose shadow spans far more bins than this.
      bool confirmed = true;
      for (int k = 0; k < silhouette_min_run_bins_; ++k) {
        const int idx = (((far_start + step * k) % n_bins) + n_bins) % n_bins;
        if (!is_far(depth[idx], near_r)) { confirmed = false; break; }
      }
      if (!confirmed) continue;

      // Azimuth of the BOUNDARY between the two bins, not either centre: the
      // edge lies on the transition, and using a bin centre biases every gate
      // half a bin to one side, which at 8 m is ~7 cm of systematic error.
      const double theta = (i + 1) * dtheta - M_PI;
      const double ct = std::cos(theta), st = std::sin(theta);

      const Vec3f edge(ego.x() + near_r * ct, ego.y() + near_r * st, ego.z());
      edge_cloud.push_back(pcl::PointXYZ(static_cast<float>(edge.x()),
                                         static_cast<float>(edge.y()),
                                         static_cast<float>(edge.z())));

      // The gate runs radially AWAY from the ego, starting at the edge. Sampled
      // at map resolution so the downstream KD-tree sees the same point density
      // it gets from occlusion_frontier.
      for (double s = 0.0; s <= silhouette_shadow_depth_; s += res) {
        gate_cloud.push_back(pcl::PointXYZ(
            static_cast<float>(edge.x() + s * ct),
            static_cast<float>(edge.y() + s * st),
            static_cast<float>(edge.z())));
      }
    }

    publish(edge_cloud, pub_sil_edges_);
    publish(gate_cloud, pub_sil_gates_);
    last_sil_edges_ = edge_cloud.size();
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

    // Parallel path, deliberately after the frontier publish and sharing only
    // the ego pose and the query box with it.
    if (silhouette_en_) publishSilhouette(ego, box_min, box_max);

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
      // sil_edges is the headline number for the parallel detector: the wall
      // test should show a small, STABLE count (two per wall end, a few more
      // from the ground clutter). Hundreds means dtheta is too fine for the
      // range or jump_thresh is below the surface roughness.
      ROS_INFO("[occlusion_boundary] unknown=%zu  occlusion=%zu  open=%zu  sil_edges=%zu",
               unknown_pts.size(), occlusion_cloud.size(), open_cloud.size(),
               last_sil_edges_);
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
  ros::Publisher pub_sil_edges_, pub_sil_gates_;
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

  bool silhouette_en_{true};
  double silhouette_dtheta_deg_{1.0};
  double silhouette_jump_thresh_{0.8};
  double silhouette_shadow_depth_{3.0};
  double silhouette_z_band_{0.5};
  double silhouette_min_range_{0.3};
  int silhouette_min_run_bins_{3};
  size_t last_sil_edges_{0};

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
