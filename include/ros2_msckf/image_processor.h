/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/features2d.hpp"

#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "custom_msgs/msg/camera_measurement.hpp"
#include "custom_msgs/msg/tracking_info.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

#include "glog/logging.h"

#include <vector>
#include <map>
#include <memory>
namespace ros2_msckf{

class ImageProcessor: public rclcpp::Node
{
public:
    // COMPOSITION_PUBLIC
    // explicit ImageProcessor(const rclcpp::NodeOptions& options);
    ImageProcessor();

    ImageProcessor(const ImageProcessor&) = delete;
    ImageProcessor operator=(const ImageProcessor&) = delete;

    ~ImageProcessor();

    bool initialize();

    typedef std::shared_ptr<ImageProcessor> Ptr;
    typedef std::shared_ptr<const ImageProcessor> ConstPtr;

private:

    struct ProcessorConfig
    {
        int grid_row;
        int grid_col;
        int grid_min_feature_num;
        int grid_max_feature_num;

        int pyramid_levels;
        int patch_size;
        int fast_threshold;
        int max_iteration;
        double track_precision;
        double ransac_threshold;
        double stereo_threshold;
    };

    typedef unsigned long long FeatureIDType;

    struct FeatureMetaData
    {
        FeatureIDType id;
        float response;
        int lifetime;
        cv::Point2f cam0_point;
        cv::Point2f cam1_point;
    };

    typedef std::map<int, std::vector<FeatureMetaData> > GridFeatures;

    static bool keyPointCompareByResponse(const cv::KeyPoint& pt1, const cv::KeyPoint& pt2)
    {
        return pt1.response>pt2.response;
    }

    static bool featureCompareByLifetime(const FeatureMetaData& f1, const FeatureMetaData& f2)
    {
        return f1.lifetime > f2.lifetime;
    }

    static bool featureCompareByResponse(const FeatureMetaData& f1, const FeatureMetaData& f2)
    {
        return f1.response > f2.response;
    }

    static builtin_interfaces::msg::Time stampMinus(const builtin_interfaces::msg::Time& stamp1, const builtin_interfaces::msg::Time& stamp2)
    {
        builtin_interfaces::msg::Time stamp;
        stamp.sec = stamp1.sec - stamp2.sec;
        stamp.nanosec = stamp1.nanosec - stamp2.nanosec;
        return stamp;
    }

    static double stamp2sec(const builtin_interfaces::msg::Time& stamp)
    {
        return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec)/1e9;
    }

    bool loadParameters();

    bool createRosIO();

    void stereoCallback(
        const sensor_msgs::msg::Image::SharedPtr& cam0_img,
        const sensor_msgs::msg::Image::SharedPtr& cam1_img);

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    void initializeFirstFrame();

    void trackFeatures();

    void addNewFeatures();

    void pruneGridFeatures();

    void publish();

    void drawFeaturesMono();

    void drawFeaturesStereo();

    void createImagePyramids();

    void integrateImuData(cv::Matx33f& cam0_R_p_c, cv::Matx33f& cam1_R_p_c);

    void predictFeatureTracking(
        const std::vector<cv::Point2f>& input_pts,
        const cv::Matx33f& R_p_c,
        const cv::Vec4d& intrinsics,
        std::vector<cv::Point2f>& compenstated_pts);
    
    void twoPointRansac(
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        const cv::Matx33f& R_p_c,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs,
        const double& inlier_error,
        const double& success_probability,
        std::vector<int>& inlier_markers);
    void undistortPoints(
        const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs,
        std::vector<cv::Point2f>& pts_out,
        const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));
    void rescalePoints(
        std::vector<cv::Point2f>& pts1,
        std::vector<cv::Point2f>& pts2,
        float& scaling_factor);
    std::vector<cv::Point2f> distortPoints(
        const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs);

    void stereoMatch(
        const std::vector<cv::Point2f>& cam0_points,
        std::vector<cv::Point2f>& cam1_points,
        std::vector<unsigned char>& inlier_markers);

    template <typename T>
    void removeUnmarkedElements(
        const std::vector<T>& raw_vec,
        const std::vector<unsigned char>& markers,
        std::vector<T>& refined_vec) {
        if (raw_vec.size() != markers.size()) {
        RCLCPP_WARN(get_logger(), "The input size of raw_vec(%lu) and markers(%lu) does not match...",
            raw_vec.size(), markers.size());
        }
        for (int i = 0; i < markers.size(); ++i) {
        if (markers[i] == 0) continue;
        refined_vec.push_back(raw_vec[i]);
        }
        return;
    }

    void debugDraw(std::vector<cv::Point2f> &cam0_pts,
                   std::vector<cv::Point2f> &cam1_pts, 
                   std::vector<unsigned char> &status);

private:
    
    bool is_first_img;

    FeatureIDType next_feature_id;

    ProcessorConfig processor_config;
    cv::Ptr<cv::Feature2D> detector_ptr;

    std::vector<sensor_msgs::msg::Imu> imu_msg_buffer;

    std::string cam0_distortion_model;
    cv::Vec2i cam0_resolution;
    cv::Vec4d cam0_intrinsics;
    cv::Vec4d cam0_distortion_coeffs;

    std::string cam1_distortion_model;
    cv::Vec2i cam1_resolution;
    cv::Vec4d cam1_intrinsics;
    cv::Vec4d cam1_distortion_coeffs;

    cv::Matx33d R_cam0_imu;
    cv::Vec3d t_cam0_imu;
    cv::Matx33d R_cam1_imu;
    cv::Vec3d t_cam1_imu;

    cv_bridge::CvImageConstPtr cam0_prev_img_ptr;
    cv_bridge::CvImageConstPtr cam0_curr_img_ptr;
    cv_bridge::CvImageConstPtr cam1_curr_img_ptr;

    std::vector<cv::Mat> prev_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam1_pyramid_;

    std::shared_ptr<GridFeatures> prev_features_ptr;
    std::shared_ptr<GridFeatures> curr_features_ptr;

    int before_tracking;
    int after_tracking;
    int after_matching;
    int after_ransac;

    message_filters::Subscriber<sensor_msgs::msg::Image> cam0_img_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> cam1_img_sub;
    message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image> stereo_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Publisher<custom_msgs::msg::CameraMeasurement>::SharedPtr feature_pub;
    rclcpp::Publisher<custom_msgs::msg::TrackingInfo>::SharedPtr tracking_info_pub;
    image_transport::Publisher debug_stereo_pub;
    
    std::map<FeatureIDType, int> feature_lifetime;
    void updateFeatureLifetime();
    void featureLifetimeStatistics();

    rclcpp::Clock clocker;
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstP;
}   //end namespace ros2_msckf

#endif