#include "ros2_msckf/image_processor.h"
#include "rclcpp/rclcpp.hpp"
#include "glog/logging.h"

int main(int argc, char* argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    google::InitGoogleLogging("image_processor");
    FLAGS_log_dir = "/home/chao/SLAM/work_slam/glogs";
    std::shared_ptr<ros2_msckf::ImageProcessor> ptr(new ros2_msckf::ImageProcessor());
    ptr->initialize();
    rclcpp::spin(ptr);
    rclcpp::shutdown();
    google::ShutdownGoogleLogging();
    return 0;
}