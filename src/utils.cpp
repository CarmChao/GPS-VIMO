#include "ros2_msckf/utils.h"
#include <string>
#include <vector>

#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/thread/mutex.hpp>
// #include <boost/math/constants/constants.hpp>
#include <boost/random/variate_generator.hpp>

namespace ros2_msckf {
namespace utils {

Eigen::Isometry3d getTransformEigen(const rclcpp::Node::SharedPtr nh,
                                    const std::string &field) {
  Eigen::Isometry3d T;
  cv::Mat c = getTransformCV(nh, field);

  T.linear()(0, 0)   = c.at<double>(0, 0);
  T.linear()(0, 1)   = c.at<double>(0, 1);
  T.linear()(0, 2)   = c.at<double>(0, 2);
  T.linear()(1, 0)   = c.at<double>(1, 0);
  T.linear()(1, 1)   = c.at<double>(1, 1);
  T.linear()(1, 2)   = c.at<double>(1, 2);
  T.linear()(2, 0)   = c.at<double>(2, 0);
  T.linear()(2, 1)   = c.at<double>(2, 1);
  T.linear()(2, 2)   = c.at<double>(2, 2);
  T.translation()(0) = c.at<double>(0, 3);
  T.translation()(1) = c.at<double>(1, 3);
  T.translation()(2) = c.at<double>(2, 3);
  return T;
}

cv::Mat getTransformCV(const rclcpp::Node::SharedPtr nh,
                       const std::string &field) {
  cv::Mat T;
  // try {
  //   // first try reading kalibr format
  //   T = getKalibrStyleTransform(nh, field);
  // } catch (std::runtime_error &e) {
  //   // maybe it's the old style format?
  //   // ROS_DEBUG_STREAM("cannot read transform " << field
  //   //                  << " in kalibr format, trying old one!");
  //   // RCLCPP_DEBUG("cannot read transform " << field
  //   //              << " in kalibr format, trying old one!");
    try {
      T = getVec16Transform(nh, field);
    } catch (std::runtime_error &e) {
      std::string msg = "cannot read transform " + field + " error: " + e.what();
    //   ROS_ERROR_STREAM(msg);
      throw std::runtime_error(msg);
    }
  // }
  return T;
}

cv::Mat getVec16Transform(const rclcpp::Node::SharedPtr nh,
                          const std::string &field) {
  std::vector<double> v;
  rclcpp::Parameter v_param;
  nh->declare_parameter(field);
  nh->get_parameter(field, v_param);
  v = v_param.as_double_array();
  if (v.size() != 16) {
    throw std::runtime_error("invalid vec16!");
  }
  cv::Mat T = cv::Mat(v).clone().reshape(1, 4); // one channel 4 rows
  return T;
}

void fromMsg(const geometry_msgs::msg::Vector3 &ros_vec, Eigen::Vector3d &eigen_vec)
{
    eigen_vec[0] = ros_vec.x;
    eigen_vec[1] = ros_vec.y;
    eigen_vec[2] = ros_vec.z;
}

// cv::Mat getKalibrStyleTransform(const rclcpp::Node::SharedPtr &nh,
//                                 const std::string &field) {
//   cv::Mat T = cv::Mat::eye(4, 4, CV_64FC1);
//   XmlRpc::XmlRpcValue lines;
//   if (!nh->get_parameter(field, lines)) {
//     throw (std::runtime_error("cannot find transform " + field));
//   }
//   if (lines.size() != 4 || lines.getType() != XmlRpc::XmlRpcValue::TypeArray) {
//     throw (std::runtime_error("invalid transform " + field));
//   }
//   for (int i = 0; i < lines.size(); i++) {
//     if (lines.size() != 4 || lines.getType() != XmlRpc::XmlRpcValue::TypeArray) {
//       throw (std::runtime_error("bad line for transform " + field));
//     }
//     for (int j = 0; j < lines[i].size(); j++) {
//       if (lines[i][j].getType() != XmlRpc::XmlRpcValue::TypeDouble) {
//         throw (std::runtime_error("bad value for transform " + field));
//       } else {
//         T.at<double>(i,j) = static_cast<double>(lines[i][j]);
//       }
//     }
//   }
//   return T;
// }

} // namespace utils

static boost::uint32_t first_seed_ = 0;

/// Compute the first seed to be used; this function should be called only once
static boost::uint32_t firstSeed(void)
{
  boost::scoped_ptr<int> mem(new int());
  first_seed_ = (boost::uint32_t)(
      (boost::posix_time::microsec_clock::universal_time() - boost::posix_time::ptime(boost::date_time::min_date_time))
          .total_microseconds() +
      (unsigned long long)(mem.get()));
  return first_seed_;
}

/// We use a different random number generator for the seeds of the
/// Other random generators. The root seed is from the number of
/// nano-seconds in the current time.
static boost::uint32_t nextSeed(void)
{
  static boost::mutex rngMutex;
  boost::mutex::scoped_lock slock(rngMutex);
  static boost::lagged_fibonacci607 sGen(firstSeed());
  static boost::uniform_int<> sDist(1, 1000000000);
  static boost::variate_generator<boost::lagged_fibonacci607&, boost::uniform_int<> > s(sGen, sDist);
  boost::uint32_t v = s();
  return v;
}

RandomGenerators::RandomGenerators():generator_(nextSeed())
{}
} // namespace msckf_vio
