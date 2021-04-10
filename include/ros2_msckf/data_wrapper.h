#ifndef DATA_WRAPPER_H
#define DATA_WRAPPER_H

#include "ros2_msckf/image_processor.h"
#include "ros2_msckf/msckf_vio.h"

#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

#include <vector>
#include <map>
#include <memory>


class DataWrapper: public rclcpp::Node
{
public:

    DataWrapper();

    ~DataWrapper();

private:

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    
private:

    message_filters::Subscriber<sensor_msgs::msg::Image> cam0_img_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> cam1_img_sub;
    message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image> stereo_sub;
    rclcpp::Publisher<custom_msgs::msg::CameraMeasurement>::SharedPtr feature_pub;
    rclcpp::Publisher<custom_msgs::msg::TrackingInfo>::SharedPtr tracking_info_pub;
    image_transport::Publisher debug_stereo_pub;

};

#endif