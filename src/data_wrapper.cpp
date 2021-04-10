#include "ros2_msckf/data_wrapper.h"


using namespace std::placeholders::_1

DataWrapper::DataWrapper() : Node("ros2_msckf")
{
    image_transport::ImageTransport it(shared_from_this());
    debug_stereo_pub = it.advertise("debug_stereo_image", 1);
    cam0_img_sub.subscribe(shared_from_this(), "cam0_image");
    cam1_img_sub.subscribe(shared_from_this(), "cam1_image");
    imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", 50
                std::bind(&DataWrapper::imuCallback, this, _1));
}

DataWrapper::~DataWrapper()
{
    // destroy
}