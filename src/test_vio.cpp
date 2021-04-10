#include "ros2_msckf/msckf_vio.h"
#include "rclcpp/rclcpp.hpp"
#include "glog/logging.h"

int main(int argc, char* argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    google::InitGoogleLogging("msckf vio");
    FLAGS_log_dir = "/home/chao/px4_ros/glogs";
    auto node_ptr = std::make_shared<ros2_msckf::MsckfVio>();
    node_ptr->initialize();
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    google::ShutdownGoogleLogging();
    return 0;
}