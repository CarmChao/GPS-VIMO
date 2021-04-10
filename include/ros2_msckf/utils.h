#ifndef MSCKF_UTILS_H
#define MSCKF_UTILS_H

#include "rclcpp/rclcpp.hpp"
#include "opencv2/core.hpp"
#include "Eigen/Geometry"
#include "Eigen/Dense"
#include "geometry_msgs/msg/vector3.hpp"

#include <string>
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace ros2_msckf
{

namespace utils
{
Eigen::Isometry3d getTransformEigen(const rclcpp::Node::SharedPtr nh,
                                    const std::string &field);

cv::Mat getTransformCV(const rclcpp::Node::SharedPtr nh,
                       const std::string &field);

cv::Mat getVec16Transform(const rclcpp::Node::SharedPtr nh,
                          const std::string &field);

// cv::Mat getKalibrStyleTransform(const rclcpp::Node::SharedPtr nh,
//                                 const std::string &field);
void fromMsg(const geometry_msgs::msg::Vector3 &ros_vec, Eigen::Vector3d &eigen_vec);
}//end utils

class RandomGenerators
{
public:
    RandomGenerators();

    int uniformInteger(int min, int max)
    {
        boost::uniform_int<> dis(min, max);
        return dis(generator_);
    }

private:
    boost::mt19937 generator_;
};

struct MagMsg
{
    double timestamp;

    Eigen::Vector3d magnetometer_ga;
};
}//end ros2_msckf

#endif