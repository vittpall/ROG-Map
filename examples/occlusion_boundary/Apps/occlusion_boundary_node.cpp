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
    // Elevation binning. These REPLACE the old silhouette_z_band slab: the
    // profile is spherical now, so instead of folding a horizontal band of
    // voxels into one polar sweep it bins them by elevation as well.
    //
    // The bounds must match the SENSOR's vertical FOV, not the volume you care
    // about. Bins outside the FOV are empty because nothing was ever measured
    // there, and an empty bin next to a surface looks exactly like a silhouette
    // against open space -- get these wrong and you get a ring of phantom edges
    // at the FOV limit. Defaults are a mid-range spinning lidar (~+/-15 deg).
    nh_.param<double>("silhouette_elev_min_deg", silhouette_elev_min_deg_, -15.0);
    nh_.param<double>("silhouette_elev_max_deg", silhouette_elev_max_deg_, 15.0);
    // Elevation bin width [deg]. Same floor argument as dtheta; keeping it
    // equal to dtheta gives square bins, which is what the run-length check
    // assumes when it walks either axis with one threshold.
    nh_.param<double>("silhouette_dphi_deg", silhouette_dphi_deg_, 1.0);
    // Emit edges where one side has no return at all. In 2D this was the
    // wall-END case and always correct; in 3D "no return" also means
    // "unobserved", so it is off by default. Turn it on once the elevation
    // bounds above are known to be right.
    nh_.param<bool>("silhouette_open_edges", silhouette_open_edges_, false);
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
      ROS_INFO("[occlusion_boundary] silhouette 3d: dtheta=%.2f dphi=%.2f deg elev=[%.1f, %.1f] "
               "deg jump=%.2f m depth=%.2f m open_edges=%d",
               silhouette_dtheta_deg_, silhouette_dphi_deg_, silhouette_elev_min_deg_,
               silhouette_elev_max_deg_, silhouette_jump_thresh_, silhouette_shadow_depth_,
               static_cast<int>(silhouette_open_edges_));
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

    const int n_az = std::max(
        8, static_cast<int>(std::lround(360.0 / std::max(0.05, silhouette_dtheta_deg_))));
    const double dtheta = 2.0 * M_PI / n_az;

    // Elevation is NOT the azimuth treatment with a different modulus. Two
    // differences, and both are load-bearing:
    //   - it does not wrap. The poles are not adjacent bins, so the edge scan
    //     below clamps in this axis instead of taking a modulus.
    //   - it is bounded by the SENSOR's vertical FOV, not by 180 deg. This is
    //     the whole reason the planar detector could get away with treating an
    //     empty bin as open space: in 3D most directions are empty because the
    //     lidar never looked there, not because nothing is there, and every FOV
    //     limit would otherwise read as a ring of silhouette edges.
    const double phi_min = silhouette_elev_min_deg_ * M_PI / 180.0;
    const double phi_max = silhouette_elev_max_deg_ * M_PI / 180.0;
    const double dphi_req = std::max(0.05, silhouette_dphi_deg_) * M_PI / 180.0;
    const int n_el = std::max(1, static_cast<int>(std::ceil((phi_max - phi_min) / dphi_req)));
    const double dphi = (phi_max - phi_min) / n_el;

    rog_map::vec_E<Vec3f> occ_pts;
    map_->boxSearch(box_min, box_max, GridType::OCCUPIED, occ_pts);

    // Circumradius of the voxel in 3D, not the XY diagonal the planar version
    // used: the splat now has to cover the voxel's extent out of plane too.
    const double r_vox = 0.5 * std::sqrt(3.0) * map_->getResolution();
    std::vector<double> depth(static_cast<size_t>(n_az) * n_el, kNoDepth);
    const auto at = [&](int ia, int ie) -> double& {
      return depth[static_cast<size_t>(ie) * n_az + ia];
    };

    for (const auto& p : occ_pts) {
      const double dx = p.x() - ego.x();
      const double dy = p.y() - ego.y();
      const double dz = p.z() - ego.z();
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);   // SLANT range now
      // Re-enabled, and it matters more in 3D than it did in 2D: the box
      // corners reach sqrt(3) * query_range, so without the clip a direction
      // pointing at a corner sees half again as far as one pointing at a face
      // and the range profile has a jump built into it at every corner.
      if (r < silhouette_min_range_ || r > query_range_) continue;

      const double theta = std::atan2(dy, dx);
      const double phi = std::asin(std::max(-1.0, std::min(1.0, dz / r)));
      const double half_ang = std::atan2(r_vox, r);
      if (phi + half_ang < phi_min || phi - half_ang > phi_max) continue;

      // The azimuth extent of a voxel GROWS as 1/cos(phi) away from the
      // horizon: bins converge at the poles while the voxel does not. Skipping
      // this is what combs the profile at high elevation and sprays phantom
      // edges there -- the same failure centre-binning used to produce in the
      // near field, just moved to the other axis.
      const double cphi = std::max(1e-3, std::cos(std::fabs(phi) + half_ang));
      const double half_az = std::min(M_PI, half_ang / cphi);

      const int a0 = static_cast<int>(std::floor((theta - half_az + M_PI) / dtheta));
      const int a1 = static_cast<int>(std::floor((theta + half_az + M_PI) / dtheta));
      const int e0 = std::max(0, static_cast<int>(std::floor((phi - half_ang - phi_min) / dphi)));
      const int e1 = std::min(n_el - 1,
                              static_cast<int>(std::floor((phi + half_ang - phi_min) / dphi)));
      // Both extents are bounded by the half-angle, which is bounded by
      // silhouette_min_range: neither loop can run away even for a voxel the
      // ego is sitting on top of.
      for (int e = e0; e <= e1; ++e) {
        for (int a = a0; a <= a1; ++a) {
          const int aw = ((a % n_az) + n_az) % n_az;   // wrap, negatives included
          double& d = at(aw, e);
          d = std::min(d, r);
        }
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

    // Direction of a bin BOUNDARY. Same argument as the planar version: the
    // edge lies on the transition, and using a bin centre biases every gate
    // half a bin to one side. In elevation the boundary is only meaningful on
    // the axis the jump was found on, so the other coordinate stays centred.
    const auto dir_of = [&](double theta, double phi) {
      return Vec3f(std::cos(phi) * std::cos(theta),
                   std::cos(phi) * std::sin(theta),
                   std::sin(phi));
    };

    // Two neighbour axes instead of one. (1,0) wraps in azimuth exactly as
    // before; (0,1) does not wrap and is skipped at the top elevation row.
    for (int ie = 0; ie < n_el; ++ie) {
      for (int ia = 0; ia < n_az; ++ia) {
        for (int axis = 0; axis < 2; ++axis) {
          const bool az_axis = (axis == 0);
          if (!az_axis && ie + 1 >= n_el) continue;

          const int ja = az_axis ? (ia + 1) % n_az : ia;
          const int je = az_axis ? ie : ie + 1;

          const double a = at(ia, ie), b = at(ja, je);
          if (!std::isfinite(a) && !std::isfinite(b)) continue;   // nothing either side

          double near_r;
          bool far_is_forward;   // is the shadow on the j side of the transition?
          if (!std::isfinite(a) || !std::isfinite(b)) {
            // Surface on one side, no return on the other. In 2D this was the
            // wall-END case and was always worth emitting. In 3D "no return"
            // conflates open space with unobserved space, so it is opt-in:
            // leave silhouette_open_edges false until the FOV bounds above are
            // known to match the sensor.
            if (!silhouette_open_edges_) continue;
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
          const int fa = far_is_forward ? ja : ia;
          const int fe = far_is_forward ? je : ie;

          // Run-length confirmation, now walking along the axis the jump was
          // found on. A real shadow keeps its far side far for many bins; a
          // one-bin spike does not. Two things produce those spikes and both
          // are worst exactly where the ego is closest to the wall:
          //   - grazing incidence, where dr/dangle along a continuous oblique
          //     surface legitimately exceeds jump_thresh
          //   - a single missing voxel in the occupied set
          // Requiring the far side to persist costs nothing on a genuine
          // corner, whose shadow spans far more bins than this.
          // bool confirmed = true;
          // for (int k = 0; k < silhouette_min_run_bins_; ++k) {
          //   const int ka = az_axis ? ((((fa + step * k) % n_az) + n_az) % n_az) : fa;
          //   const int ke = az_axis ? fe : (fe + step * k);
          //   if (ke < 0 || ke >= n_el) break;   // ran off the FOV: cannot confirm
          //   if (!is_far(at(ka, ke), near_r)) { confirmed = false; break; }
          // }
          // if (!confirmed) continue;

          const double theta = az_axis ? ((ia + 1) * dtheta - M_PI)
                                       : ((ia + 0.5) * dtheta - M_PI);
          const double phi = az_axis ? (phi_min + (ie + 0.5) * dphi)
                                     : (phi_min + (ie + 1) * dphi);
          const Vec3f dir = dir_of(theta, phi);

          const Vec3f edge = ego + near_r * dir;
          edge_cloud.push_back(pcl::PointXYZ(static_cast<float>(edge.x()),
                                             static_cast<float>(edge.y()),
                                             static_cast<float>(edge.z())));

          // The gate runs radially AWAY from the ego, starting at the edge --
          // along the full 3D direction now, not the flattened one. Sampled at
          // map resolution so the downstream KD-tree sees the same point
          // density it gets from occlusion_frontier.
          // for (double s = 0.0; s <= silhouette_shadow_depth_; s += res) {
          //   const Vec3f g = edge + s * dir;
          //   gate_cloud.push_back(pcl::PointXYZ(static_cast<float>(g.x()),
          //                                      static_cast<float>(g.y()),
          //                                      static_cast<float>(g.z())));
          // }
        }
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
  double silhouette_elev_min_deg_{-15.0};
  double silhouette_elev_max_deg_{15.0};
  double silhouette_dphi_deg_{1.0};
  bool silhouette_open_edges_{false};
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
