#include "ros2_msckf/image_processor.h"
#include "ros2_msckf/utils.h"

#include "Eigen/Dense"
#include "sensor_msgs/image_encodings.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <algorithm>
#include <set>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace ros2_msckf {
ImageProcessor::ImageProcessor() :
    Node("image_processor"),
    is_first_img(true),
    stereo_sub(10),
    prev_features_ptr(new GridFeatures()),
    curr_features_ptr(new GridFeatures()) {
    // RCLCPP_INFO(this->get_logger(), "construct finished!");
    // LOG(INFO) << "info test: "<<0.11<<" "<<0.23342341242;
    return;
}

ImageProcessor::~ImageProcessor() {
  destroyAllWindows();
  //RCLCPP_INFO(this->get_logger(), "Feature lifetime statistics:");
  //featureLifetimeStatistics();
  // google::ShutdownGoogleLogging();
  return;
}

bool ImageProcessor::loadParameters() {
    cam0_distortion_model = this->declare_parameter<string>("cam0.distortion_model", "radtan");
    cam1_distortion_model = this->declare_parameter<string>("cam1.distortion_model", "radtan");
    // cam0_distortion_model = this->get_parameter("cam0.distortion_model");
    // cam1_distortion_model = this->get_parameter("cam1_distortion_model");

    this->declare_parameter("cam0.resolution");
    vector<int64_t> cam0_resolution_temp(2);
    rclcpp::Parameter cam0_resolution_param;
    this->get_parameter("cam0.resolution", cam0_resolution_param);
    cam0_resolution_temp = cam0_resolution_param.as_integer_array();
    cam0_resolution[0] = cam0_resolution_temp[0];
    cam0_resolution[1] = cam0_resolution_temp[1];

    this->declare_parameter("cam1.resolution");
    vector<int64_t> cam1_resolution_temp(2);
    rclcpp::Parameter cam1_resolution_param;
    this->get_parameter("cam1.resolution", cam1_resolution_param);
    cam1_resolution_temp = cam1_resolution_param.as_integer_array();
    cam1_resolution[0] = cam1_resolution_temp[0];
    cam1_resolution[1] = cam1_resolution_temp[1];

    this->declare_parameter("cam0.intrinsics");
    vector<double> cam0_intrinsics_temp(4);
    rclcpp::Parameter cam0_intrinsics_param;
    this->get_parameter("cam0.intrinsics", cam0_intrinsics_param);
    cam0_intrinsics_temp = cam0_intrinsics_param.as_double_array();
    cam0_intrinsics[0] = cam0_intrinsics_temp[0];
    cam0_intrinsics[1] = cam0_intrinsics_temp[1];
    cam0_intrinsics[2] = cam0_intrinsics_temp[2];
    cam0_intrinsics[3] = cam0_intrinsics_temp[3];

    this->declare_parameter("cam1.intrinsics");
    vector<double> cam1_intrinsics_temp(4);
    rclcpp::Parameter cam1_intrinsics_param;
    this->get_parameter("cam1.intrinsics", cam1_intrinsics_param);
    cam1_intrinsics_temp = cam1_intrinsics_param.as_double_array();
    cam1_intrinsics[0] = cam1_intrinsics_temp[0];
    cam1_intrinsics[1] = cam1_intrinsics_temp[1];
    cam1_intrinsics[2] = cam1_intrinsics_temp[2];
    cam1_intrinsics[3] = cam1_intrinsics_temp[3];

    this->declare_parameter("cam0.distortion_coeffs");
    vector<double> cam0_distortion_coeffs_temp(4);
    rclcpp::Parameter cam0_distortion_coeffs_param;
    this->get_parameter("cam0.distortion_coeffs",
        cam0_distortion_coeffs_param);
    cam0_distortion_coeffs_temp = cam0_distortion_coeffs_param.as_double_array();
    cam0_distortion_coeffs[0] = cam0_distortion_coeffs_temp[0];
    cam0_distortion_coeffs[1] = cam0_distortion_coeffs_temp[1];
    cam0_distortion_coeffs[2] = cam0_distortion_coeffs_temp[2];
    cam0_distortion_coeffs[3] = cam0_distortion_coeffs_temp[3];

    this->declare_parameter("cam1.distortion_coeffs");
    vector<double> cam1_distortion_coeffs_temp(4);
    rclcpp::Parameter cam1_distortion_coeffs_param;
    this->get_parameter("cam1.distortion_coeffs",
        cam1_distortion_coeffs_param);
    cam1_distortion_coeffs_temp = cam1_distortion_coeffs_param.as_double_array();
    cam1_distortion_coeffs[0] = cam1_distortion_coeffs_temp[0];
    cam1_distortion_coeffs[1] = cam1_distortion_coeffs_temp[1];
    cam1_distortion_coeffs[2] = cam1_distortion_coeffs_temp[2];
    cam1_distortion_coeffs[3] = cam1_distortion_coeffs_temp[3];

    cv::Mat     T_imu_cam0 = utils::getTransformCV(shared_from_this(), "cam0.T_cam_imu");
    R_cam0_imu = T_imu_cam0(cv::Rect(0,0,3,3));
    t_cam0_imu = T_imu_cam0(cv::Rect(3,0,1,3));
    // R_cam0_imu = R_imu_cam0.t();
    // t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;

    // cv::Mat T_cam0_cam1 = utils::getTransformCV(shared_from_this(), "cam1.T_cn_cnm1");
    // cv::Mat T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;
    cv::Mat T_imu_cam1 = utils::getTransformCV(shared_from_this(), "cam1.T_cam_imu");
    R_cam1_imu = T_imu_cam1(cv::Rect(0,0,3,3));
    t_cam1_imu = T_imu_cam1(cv::Rect(3,0,1,3));
    // R_cam1_imu = R_imu_cam1.t();
    // t_cam1_imu = -R_imu_cam1.t() * t_imu_cam1;

    processor_config.grid_row = this->declare_parameter<int>("grid_row", 4);
    processor_config.grid_col = this->declare_parameter<int>("grid_col", 4);
    processor_config.grid_min_feature_num = this->declare_parameter<int>("grid_min_feature_num", 2);
    processor_config.grid_max_feature_num = this->declare_parameter<int>("grid_max_feature_num", 4);
    processor_config.pyramid_levels = this->declare_parameter<int>("pyramid_levels", 3);
    processor_config.patch_size = this->declare_parameter<int>("patch_size", 31);
    processor_config.fast_threshold = this->declare_parameter<int>("fast_threshold", 20);
    processor_config.max_iteration = this->declare_parameter<int>("max_iteration", 30);
    processor_config.track_precision = this->declare_parameter<double>("track_precision", 0.01);
    processor_config.ransac_threshold = this->declare_parameter<double>("ransac_threshold", 3.0);

    processor_config.stereo_threshold = this->declare_parameter<double>("stereo_threshold", 3.0);

    RCLCPP_INFO(this->get_logger(), "===========================================");
    RCLCPP_INFO(this->get_logger(), "cam0_resolution: %d, %d",
        cam0_resolution[0], cam0_resolution[1]);
    RCLCPP_INFO(this->get_logger(), "cam0_intrinscs: %f, %f, %f, %f",
        cam0_intrinsics[0], cam0_intrinsics[1],
        cam0_intrinsics[2], cam0_intrinsics[3]);
    RCLCPP_INFO(this->get_logger(), "cam0_distortion_model: %s",
        cam0_distortion_model.c_str());
    RCLCPP_INFO(this->get_logger(), "cam0_distortion_coefficients: %f, %f, %f, %f",
        cam0_distortion_coeffs[0], cam0_distortion_coeffs[1],
        cam0_distortion_coeffs[2], cam0_distortion_coeffs[3]);

    RCLCPP_INFO(this->get_logger(), "cam1_resolution: %d, %d",
        cam1_resolution[0], cam1_resolution[1]);
    RCLCPP_INFO(this->get_logger(), "cam1_intrinscs: %f, %f, %f, %f",
        cam1_intrinsics[0], cam1_intrinsics[1],
        cam1_intrinsics[2], cam1_intrinsics[3]);
    RCLCPP_INFO(this->get_logger(), "cam1_distortion_model: %s",
        cam1_distortion_model.c_str());
    RCLCPP_INFO(this->get_logger(), "cam1_distortion_coefficients: %f, %f, %f, %f",
        cam1_distortion_coeffs[0], cam1_distortion_coeffs[1],
        cam1_distortion_coeffs[2], cam1_distortion_coeffs[3]);

    cout << R_cam0_imu << endl;
    cout << t_cam0_imu.t() << endl;

    RCLCPP_INFO(this->get_logger(), "grid_row: %d",
        processor_config.grid_row);
    RCLCPP_INFO(this->get_logger(), "grid_col: %d",
        processor_config.grid_col);
    RCLCPP_INFO(this->get_logger(), "grid_min_feature_num: %d",
        processor_config.grid_min_feature_num);
    RCLCPP_INFO(this->get_logger(), "grid_max_feature_num: %d",
        processor_config.grid_max_feature_num);
    RCLCPP_INFO(this->get_logger(), "pyramid_levels: %d",
        processor_config.pyramid_levels);
    RCLCPP_INFO(this->get_logger(), "patch_size: %d",
        processor_config.patch_size);
    RCLCPP_INFO(this->get_logger(), "fast_threshold: %d",
        processor_config.fast_threshold);
    RCLCPP_INFO(this->get_logger(), "max_iteration: %d",
        processor_config.max_iteration);
    RCLCPP_INFO(this->get_logger(), "track_precision: %f",
        processor_config.track_precision);
    RCLCPP_INFO(this->get_logger(), "ransac_threshold: %f",
        processor_config.ransac_threshold);
    RCLCPP_INFO(this->get_logger(), "stereo_threshold: %f",
        processor_config.stereo_threshold);
    RCLCPP_INFO(this->get_logger(), "===========================================");
    return true;
}

bool ImageProcessor::createRosIO() {
  feature_pub = this->create_publisher<custom_msgs::msg::CameraMeasurement>(
      "features", 3);
  tracking_info_pub = this->create_publisher<custom_msgs::msg::TrackingInfo>(
      "tracking_info", 1);
  image_transport::ImageTransport it(shared_from_this());
  debug_stereo_pub = it.advertise("debug_stereo_image", 1);

  cam0_img_sub.subscribe(shared_from_this(),"cam0_image");
  cam1_img_sub.subscribe(shared_from_this(), "cam1_image");
  stereo_sub.connectInput(cam0_img_sub, cam1_img_sub);
  stereo_sub.registerCallback(&ImageProcessor::stereoCallback, this);
  imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", 50,
      std::bind(&ImageProcessor::imuCallback, this, std::placeholders::_1));

  return true;
}

bool ImageProcessor::initialize() {
  if (!loadParameters()) return false;
  RCLCPP_INFO(this->get_logger(), "Finish loading ROS parameters...");

  // Create feature detector.
  detector_ptr = FastFeatureDetector::create(
      processor_config.fast_threshold);

  if (!createRosIO()) return false;
  RCLCPP_INFO(this->get_logger(), "Finish creating ROS IO...");

  return true;
}

void ImageProcessor::stereoCallback(
    const sensor_msgs::msg::Image::SharedPtr& cam0_img,
    const sensor_msgs::msg::Image::SharedPtr& cam1_img) {

  //cout << "==================================" << endl;
  // RCLCPP_INFO(this->get_logger(), "time: %lf\n", rclcpp::Time(cam0_img->header.stamp).seconds());
  
  // return;
  // Get the current image.
  cam0_curr_img_ptr = cv_bridge::toCvShare(cam0_img,
      sensor_msgs::image_encodings::MONO8);
  cam1_curr_img_ptr = cv_bridge::toCvShare(cam1_img,
      sensor_msgs::image_encodings::MONO8);

  // cv::imshow("left_img", cam0_curr_img_ptr->image);
  // int key_val = cv::waitKey(2);
  // if(key_val == 32)
  // {
  //   cv::imwrite("/home/chao/Desktop/distort.jpg", cam0_curr_img_ptr->image);
  // }
  // return;
  // cv::equalizeHist(cam0_curr_img_ptr->image, cam0_curr_img_ptr->image);
  // cv::equalizeHist(cam1_curr_img_ptr->image, cam1_curr_img_ptr->image);

  // Build the image pyramids once since they're used at multiple places
  createImagePyramids();

  // Detect features in the first frame.

  if (is_first_img) {
    rclcpp::Time start_time = clocker.now();
    initializeFirstFrame();
    //RCLCPP_INFO(this->get_logger(), "Detection time: %f",
    //    (ros::Time::now()-start_time).toSec());
    is_first_img = false;

    // Draw results.
    start_time = clocker.now();
    drawFeaturesStereo();
    //RCLCPP_INFO(this->get_logger(), "Draw features: %f",
    //    (ros::Time::now()-start_time).toSec());
  } else {
    // Track the feature in the previous image.
    rclcpp::Time start_time = clocker.now();
    trackFeatures();
    //RCLCPP_INFO(this->get_logger(), "Tracking time: %f",
    //    (ros::Time::now()-start_time).toSec());

    // Add new features into the current image.
    start_time = clocker.now();
    addNewFeatures();
    //RCLCPP_INFO(this->get_logger(), "Addition time: %f",
    //    (ros::Time::now()-start_time).toSec());

    // Add new features into the current image.
    start_time = clocker.now();
    pruneGridFeatures();
    //RCLCPP_INFO(this->get_logger(), "Prune grid features: %f",
    //    (ros::Time::now()-start_time).toSec());

    // Draw results.
    start_time = clocker.now();
    drawFeaturesStereo();
    //RCLCPP_INFO(this->get_logger(), "Draw features: %f",
    //    (ros::Time::now()-start_time).toSec());
  }

  //ros::Time start_time = ros::Time::now();
  //updateFeatureLifetime();
  //RCLCPP_INFO(this->get_logger(), "Statistics: %f",
  //    (ros::Time::now()-start_time).toSec());

  // Publish features in the current image.
  rclcpp::Time start_time = clocker.now();
  publish();
  //RCLCPP_INFO(this->get_logger(), "Publishing: %f",
  //    (ros::Time::now()-start_time).toSec());

  // Update the previous image and previous features.
  cam0_prev_img_ptr = cam0_curr_img_ptr;
  prev_features_ptr = curr_features_ptr;
  std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

  // Initialize the current features to empty vectors.
  curr_features_ptr.reset(new GridFeatures());
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
  }

  return;
}

void ImageProcessor::imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg) {
  // Wait for the first image to be set.
  if (is_first_img) return;
  imu_msg_buffer.push_back(*msg);
  return;
}

void ImageProcessor::createImagePyramids() {
  const Mat& curr_cam0_img = cam0_curr_img_ptr->image;
  buildOpticalFlowPyramid(
      curr_cam0_img, curr_cam0_pyramid_,
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels, true, BORDER_REFLECT_101,
      BORDER_CONSTANT, false);

  const Mat& curr_cam1_img = cam1_curr_img_ptr->image;
  buildOpticalFlowPyramid(
      curr_cam1_img, curr_cam1_pyramid_,
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels, true, BORDER_REFLECT_101,
      BORDER_CONSTANT, false);
}

void ImageProcessor::debugDraw(std::vector<cv::Point2f> &cam0_pts, std::vector<cv::Point2f> &cam1_pts, std::vector<unsigned char> &status)
{
    Scalar tracked(0, 255, 0);
    Scalar new_feature(0, 255, 255);

    static int grid_height =
      cam0_curr_img_ptr->image.rows / processor_config.grid_row;
    static int grid_width =
      cam0_curr_img_ptr->image.cols / processor_config.grid_col;

    // Create an output image.
    int img_height = cam0_curr_img_ptr->image.rows;
    int img_width = cam0_curr_img_ptr->image.cols;
    Mat out_img(img_height, img_width*2, CV_8UC3);
    cvtColor(cam0_curr_img_ptr->image,
             out_img.colRange(0, img_width), CV_GRAY2RGB);
    cvtColor(cam1_curr_img_ptr->image,
             out_img.colRange(img_width, img_width*2), CV_GRAY2RGB);

    // Draw grids on the image.
    for (int i = 1; i < processor_config.grid_row; ++i) {
      Point pt1(0, i*grid_height);
      Point pt2(img_width*2, i*grid_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i) {
      Point pt1(i*grid_width, 0);
      Point pt2(i*grid_width, img_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i) {
      Point pt1(i*grid_width+img_width, 0);
      Point pt2(i*grid_width+img_width, img_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    // vector<FeatureIDType> prev_ids(0);
    // for (const auto& grid_features : *prev_features_ptr)
    //   for (const auto& feature : grid_features.second)
    //     prev_ids.push_back(feature.id);

    // // Collect feature points in the previous frame.
    // map<FeatureIDType, Point2f> prev_cam0_points;
    // map<FeatureIDType, Point2f> prev_cam1_points;
    // for (const auto& grid_features : *prev_features_ptr)
    //   for (const auto& feature : grid_features.second) {
    //     prev_cam0_points[feature.id] = feature.cam0_point;
    //     prev_cam1_points[feature.id] = feature.cam1_point;
    //   }

    // // Collect feature points in the current frame.
    // map<FeatureIDType, Point2f> curr_cam0_points;
    // map<FeatureIDType, Point2f> curr_cam1_points;
    // for (const auto& grid_features : *curr_features_ptr)
    //   for (const auto& feature : grid_features.second) {
    //     curr_cam0_points[feature.id] = feature.cam0_point;
    //     curr_cam1_points[feature.id] = feature.cam1_point;
    //   }

    // Draw tracked features.
    // for (const auto& id : prev_ids) {
    //   if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
    //       curr_cam0_points.find(id) != curr_cam0_points.end()) {
    //     cv::Point2f prev_pt0 = prev_cam0_points[id];
    //     cv::Point2f prev_pt1 = prev_cam1_points[id] + Point2f(img_width, 0.0);
    //     cv::Point2f curr_pt0 = curr_cam0_points[id];
    //     cv::Point2f curr_pt1 = curr_cam1_points[id] + Point2f(img_width, 0.0);

    //     circle(out_img, curr_pt0, 3, tracked, -1);
    //     circle(out_img, curr_pt1, 3, tracked, -1);
    //     line(out_img, prev_pt0, curr_pt0, tracked, 1);
    //     line(out_img, prev_pt1, curr_pt1, tracked, 1);

    //     prev_cam0_points.erase(id);
    //     prev_cam1_points.erase(id);
    //     curr_cam0_points.erase(id);
    //     curr_cam1_points.erase(id);
    //   }
    // }

    // // Draw new features.
    // for (const auto& new_cam0_point : curr_cam0_points) {
    //   cv::Point2f pt0 = new_cam0_point.second;
    //   cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
    //     Point2f(img_width, 0.0);

    //   circle(out_img, pt0, 3, new_feature, -1);
    //   circle(out_img, pt1, 3, new_feature, -1);
    // }

    for(size_t i=0; i<status.size(); i++)
    {
      if(status[i])
      {
        circle(out_img, cam0_pts[i], 3, tracked, 1);
        cv::Point2f draw_pt1 = cam1_pts[i] + cv::Point2f(img_width, 0.0);
        circle(out_img, draw_pt1, 3, tracked, 1);
      }
      else 
      {
        circle(out_img, cam0_pts[i], 3, new_feature, -1);
      }
    }

    // cv_bridge::CvImage debug_image(cam0_curr_img_ptr->header, "bgr8", out_img);
    // debug_stereo_pub.publish(debug_image.toImageMsg());
  imshow("Feature", out_img);
  waitKey(1);

  return;
}

void ImageProcessor::initializeFirstFrame() {
  // Size of each grid.
  const Mat& img = cam0_curr_img_ptr->image;
  static int grid_height = img.rows / processor_config.grid_row;
  static int grid_width = img.cols / processor_config.grid_col;

  // Detect new features on the frist image.
  vector<KeyPoint> new_features(0);
  detector_ptr->detect(img, new_features);
  // LOG(INFO)<< "first img feature size: "<<new_features.size();
  // RCLCPP_INFO(this->get_logger(), "detect feature nums: %d", new_features.size());

  // Find the stereo matched points for the newly
  // detected features.
  vector<cv::Point2f> cam0_points(new_features.size());
  for (int i = 0; i < new_features.size(); ++i)
    cam0_points[i] = new_features[i].pt;
  //!! change part
  // sort(new_features.begin(), new_features.end(), &ImageProcessor::keyPointCompareByResponse);
  // vector<cv::Point2f> cam0_points(100);
  // for (int i = 0; i < 100; ++i)
  //   cam0_points[i] = new_features[i].pt;

  vector<cv::Point2f> cam1_points(0);
  vector<unsigned char> inlier_markers(0);
  stereoMatch(cam0_points, cam1_points, inlier_markers);

  vector<cv::Point2f> cam0_inliers(0);
  vector<cv::Point2f> cam1_inliers(0);
  vector<float> response_inliers(0);
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cam0_inliers.push_back(cam0_points[i]);
    cam1_inliers.push_back(cam1_points[i]);
    response_inliers.push_back(new_features[i].response);
  }
  // LOG(INFO)<<"features num after stereo match: "<<cam0_inliers.size();
  // RCLCPP_INFO(this->get_logger(), "match feature nums: %d", cam1_inliers.size());

  // Group the features into grids
  GridFeatures grid_new_features;
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code)
      grid_new_features[code] = vector<FeatureMetaData>(0);

  for (int i = 0; i < cam0_inliers.size(); ++i) {
    const cv::Point2f& cam0_point = cam0_inliers[i];
    const cv::Point2f& cam1_point = cam1_inliers[i];
    const float& response = response_inliers[i];

    int row = static_cast<int>(cam0_point.y / grid_height);
    int col = static_cast<int>(cam0_point.x / grid_width);
    int code = row*processor_config.grid_col + col;

    FeatureMetaData new_feature;
    new_feature.response = response;
    new_feature.cam0_point = cam0_point;
    new_feature.cam1_point = cam1_point;
    grid_new_features[code].push_back(new_feature);
  }

  // Sort the new features in each grid based on its response.
  for (auto& item : grid_new_features)
    std::sort(item.second.begin(), item.second.end(),
        &ImageProcessor::featureCompareByResponse);

  // Collect new features within each grid with high response.
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
    vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

    for (int k = 0; k < processor_config.grid_min_feature_num &&
        k < new_features_this_grid.size(); ++k) {
      features_this_grid.push_back(new_features_this_grid[k]);
      features_this_grid.back().id = next_feature_id++;
      features_this_grid.back().lifetime = 1;
    }
  }

  // debugDraw(cam0_points, cam1_points, inlier_markers);

  return;
}

void ImageProcessor::predictFeatureTracking(
    const vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts) {

  // Return directly if there are no input features.
  if (input_pts.size() == 0) {
    compensated_pts.clear();
    return;
  }
  compensated_pts.resize(input_pts.size());

  // Intrinsic matrix.
  cv::Matx33f K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);
  cv::Matx33f H = K * R_p_c * K.inv();

  for (int i = 0; i < input_pts.size(); ++i) {
    cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
    cv::Vec3f p2 = H * p1;
    compensated_pts[i].x = p2[0] / p2[2];
    compensated_pts[i].y = p2[1] / p2[2];
  }

  return;
}

void ImageProcessor::trackFeatures() {
  // Size of each grid.
  static int grid_height =
    cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =
    cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  // Compute a rough relative rotation which takes a vector
  // from the previous frame to the current frame.
  Matx33f cam0_R_p_c;
  Matx33f cam1_R_p_c;
  integrateImuData(cam0_R_p_c, cam1_R_p_c);

  // Organize the features in the previous image.
  vector<FeatureIDType> prev_ids(0);
  vector<int> prev_lifetime(0);
  vector<Point2f> prev_cam0_points(0);
  vector<Point2f> prev_cam1_points(0);

  for (const auto& item : *prev_features_ptr) {
    for (const auto& prev_feature : item.second) {
      prev_ids.push_back(prev_feature.id);
      prev_lifetime.push_back(prev_feature.lifetime);
      prev_cam0_points.push_back(prev_feature.cam0_point);
      prev_cam1_points.push_back(prev_feature.cam1_point);
    }
  }

  // Number of the features before tracking.
  before_tracking = prev_cam0_points.size();

  // Abort tracking if there is no features in
  // the previous frame.
  if (prev_ids.size() == 0) return;

  // Track features using LK optical flow method.
  vector<Point2f> curr_cam0_points(0);
  vector<unsigned char> track_inliers(0);

  predictFeatureTracking(prev_cam0_points,
      cam0_R_p_c, cam0_intrinsics, curr_cam0_points);

  calcOpticalFlowPyrLK(
      prev_cam0_pyramid_, curr_cam0_pyramid_,
      prev_cam0_points, curr_cam0_points,
      track_inliers, noArray(),
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
        processor_config.max_iteration,
        processor_config.track_precision),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  // Mark those tracked points out of the image region
  // as untracked.
  for (int i = 0; i < curr_cam0_points.size(); ++i) {
    if (track_inliers[i] == 0) continue;
    if (curr_cam0_points[i].y < 0 ||
        curr_cam0_points[i].y > cam0_curr_img_ptr->image.rows-1 ||
        curr_cam0_points[i].x < 0 ||
        curr_cam0_points[i].x > cam0_curr_img_ptr->image.cols-1)
      track_inliers[i] = 0;
  }

  // Collect the tracked points.
  vector<FeatureIDType> prev_tracked_ids(0);
  vector<int> prev_tracked_lifetime(0);
  vector<Point2f> prev_tracked_cam0_points(0);
  vector<Point2f> prev_tracked_cam1_points(0);
  vector<Point2f> curr_tracked_cam0_points(0);

  removeUnmarkedElements(
      prev_ids, track_inliers, prev_tracked_ids);
  removeUnmarkedElements(
      prev_lifetime, track_inliers, prev_tracked_lifetime);
  removeUnmarkedElements(
      prev_cam0_points, track_inliers, prev_tracked_cam0_points);
  removeUnmarkedElements(
      prev_cam1_points, track_inliers, prev_tracked_cam1_points);
  removeUnmarkedElements(
      curr_cam0_points, track_inliers, curr_tracked_cam0_points);

  // Number of features left after tracking.
  after_tracking = curr_tracked_cam0_points.size();


  // Outlier removal involves three steps, which forms a close
  // loop between the previous and current frames of cam0 (left)
  // and cam1 (right). Assuming the stereo matching between the
  // previous cam0 and cam1 images are correct, the three steps are:
  //
  // prev frames cam0 ----------> cam1
  //              |                |
  //              |ransac          |ransac
  //              |   stereo match |
  // curr frames cam0 ----------> cam1
  //
  // 1) Stereo matching between current images of cam0 and cam1.
  // 2) RANSAC between previous and current images of cam0.
  // 3) RANSAC between previous and current images of cam1.
  //
  // For Step 3, tracking between the images is no longer needed.
  // The stereo matching results are directly used in the RANSAC.

  // Step 1: stereo matching.
  vector<Point2f> curr_cam1_points(0);
  vector<unsigned char> match_inliers(0);
  stereoMatch(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

  vector<FeatureIDType> prev_matched_ids(0);
  vector<int> prev_matched_lifetime(0);
  vector<Point2f> prev_matched_cam0_points(0);
  vector<Point2f> prev_matched_cam1_points(0);
  vector<Point2f> curr_matched_cam0_points(0);
  vector<Point2f> curr_matched_cam1_points(0);

  removeUnmarkedElements(
      prev_tracked_ids, match_inliers, prev_matched_ids);
  removeUnmarkedElements(
      prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
  removeUnmarkedElements(
      prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
  removeUnmarkedElements(
      prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
  removeUnmarkedElements(
      curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
  removeUnmarkedElements(
      curr_cam1_points, match_inliers, curr_matched_cam1_points);

  // Number of features left after stereo matching.
  after_matching = curr_matched_cam0_points.size();

  // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
  vector<int> cam0_ransac_inliers(0);
  twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,
      cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, processor_config.ransac_threshold,
      0.99, cam0_ransac_inliers);

  vector<int> cam1_ransac_inliers(0);
  twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,
      cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, processor_config.ransac_threshold,
      0.99, cam1_ransac_inliers);

  // Number of features after ransac.
  after_ransac = 0;

  for (int i = 0; i < cam0_ransac_inliers.size(); ++i) {
    if (cam0_ransac_inliers[i] == 0 ||
        cam1_ransac_inliers[i] == 0) continue;
    int row = static_cast<int>(
        curr_matched_cam0_points[i].y / grid_height);
    int col = static_cast<int>(
        curr_matched_cam0_points[i].x / grid_width);
    int code = row*processor_config.grid_col + col;
    (*curr_features_ptr)[code].push_back(FeatureMetaData());

    FeatureMetaData& grid_new_feature = (*curr_features_ptr)[code].back();
    grid_new_feature.id = prev_matched_ids[i];
    grid_new_feature.lifetime = ++prev_matched_lifetime[i];
    grid_new_feature.cam0_point = curr_matched_cam0_points[i];
    grid_new_feature.cam1_point = curr_matched_cam1_points[i];

    ++after_ransac;
  }

  // Compute the tracking rate.
  int prev_feature_num = 0;
  for (const auto& item : *prev_features_ptr)
    prev_feature_num += item.second.size();

  int curr_feature_num = 0;
  for (const auto& item : *curr_features_ptr)
    curr_feature_num += item.second.size();

  RCLCPP_INFO(this->get_logger(),
      "\033[0;32m candidates: %d; track: %d; match: %d; ransac: %d/%d=%f\033[0m",
      before_tracking, after_tracking, after_matching,
      curr_feature_num, prev_feature_num,
      static_cast<double>(curr_feature_num)/
      (static_cast<double>(prev_feature_num)+1e-5));
  //printf(
  //    "\033[0;32m candidates: %d; raw track: %d; stereo match: %d; ransac: %d/%d=%f\033[0m\n",
  //    before_tracking, after_tracking, after_matching,
  //    curr_feature_num, prev_feature_num,
  //    static_cast<double>(curr_feature_num)/
  //    (static_cast<double>(prev_feature_num)+1e-5));

  return;
}

void ImageProcessor::stereoMatch(
    const vector<cv::Point2f>& cam0_points,
    vector<cv::Point2f>& cam1_points,
    vector<unsigned char>& inlier_markers) {

  if (cam0_points.size() == 0) return;

  if(cam1_points.size() == 0) {
    // Initialize cam1_points by projecting cam0_points to cam1 using the
    // rotation from stereo extrinsics
    const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
    vector<cv::Point2f> cam0_points_undistorted;
    undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model,
                    cam0_distortion_coeffs, cam0_points_undistorted,
                    R_cam0_cam1);
    cam1_points = distortPoints(cam0_points_undistorted, cam1_intrinsics,
                                cam1_distortion_model, cam1_distortion_coeffs);
  }

  // Track features using LK optical flow method.
  calcOpticalFlowPyrLK(curr_cam0_pyramid_, curr_cam1_pyramid_,
      cam0_points, cam1_points,
      inlier_markers, noArray(),
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                   processor_config.max_iteration,
                   processor_config.track_precision),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  // Mark those tracked points out of the image region
  // as untracked.
  // inlier_markers.resize(cam1_points.size(), 1);
  for (int i = 0; i < cam1_points.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    if (cam1_points[i].y < 0 ||
        cam1_points[i].y > cam1_curr_img_ptr->image.rows-1 ||
        cam1_points[i].x < 0 ||
        cam1_points[i].x > cam1_curr_img_ptr->image.cols-1)
      inlier_markers[i] = 0;
  }

  // Compute the relative rotation between the cam0
  // frame and cam1 frame.
  const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
  const cv::Vec3d t_cam0_cam1 = R_cam1_imu.t() * (t_cam0_imu-t_cam1_imu);
  // Compute the essential matrix.
  const cv::Matx33d t_cam0_cam1_hat(
      0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
      t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
      -t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
  const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

  // Further remove outliers based on the known
  // essential matrix.
  vector<cv::Point2f> cam0_points_undistorted(0);
  vector<cv::Point2f> cam1_points_undistorted(0);
  undistortPoints(
      cam0_points, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, cam0_points_undistorted);
  undistortPoints(
      cam1_points, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, cam1_points_undistorted);

  double norm_pixel_unit = 4.0 / (
      cam0_intrinsics[0]+cam0_intrinsics[1]+
      cam1_intrinsics[0]+cam1_intrinsics[1]);

  for (int i = 0; i < cam0_points_undistorted.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cv::Vec3d pt0(cam0_points_undistorted[i].x,
        cam0_points_undistorted[i].y, 1.0);
    cv::Vec3d pt1(cam1_points_undistorted[i].x,
        cam1_points_undistorted[i].y, 1.0);
    cv::Vec3d epipolar_line = E * pt0;
    double error = fabs((pt1.t() * epipolar_line)[0]) / sqrt(
        epipolar_line[0]*epipolar_line[0]+
        epipolar_line[1]*epipolar_line[1]);
    if (error > processor_config.stereo_threshold*norm_pixel_unit)
      inlier_markers[i] = 0;
  }

  return;
}
void ImageProcessor::addNewFeatures() {
  const Mat& curr_img = cam0_curr_img_ptr->image;

  // Size of each grid.
  static int grid_height =
    cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =
    cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  // Create a mask to avoid redetecting existing features.
  Mat mask(curr_img.rows, curr_img.cols, CV_8U, Scalar(1));

  for (const auto& features : *curr_features_ptr) {
    for (const auto& feature : features.second) {
      const int y = static_cast<int>(feature.cam0_point.y);
      const int x = static_cast<int>(feature.cam0_point.x);

      int up_lim = y-2, bottom_lim = y+3,
          left_lim = x-2, right_lim = x+3;
      if (up_lim < 0) up_lim = 0;
      if (bottom_lim > curr_img.rows) bottom_lim = curr_img.rows;
      if (left_lim < 0) left_lim = 0;
      if (right_lim > curr_img.cols) right_lim = curr_img.cols;

      Range row_range(up_lim, bottom_lim);
      Range col_range(left_lim, right_lim);
      mask(row_range, col_range) = 0;
    }
  }

  // Detect new features.
  vector<KeyPoint> new_features(0);
  detector_ptr->detect(curr_img, new_features, mask);

  // Collect the new detected features based on the grid.
  // Select the ones with top response within each grid afterwards.
  vector<vector<KeyPoint> > new_feature_sieve(
      processor_config.grid_row*processor_config.grid_col);
  for (const auto& feature : new_features) {
    int row = static_cast<int>(feature.pt.y / grid_height);
    int col = static_cast<int>(feature.pt.x / grid_width);
    new_feature_sieve[
      row*processor_config.grid_col+col].push_back(feature);
  }

  new_features.clear();
  for (auto& item : new_feature_sieve) {
    if (item.size() > processor_config.grid_max_feature_num) {
      std::sort(item.begin(), item.end(),
          &ImageProcessor::keyPointCompareByResponse);
      item.erase(
          item.begin()+processor_config.grid_max_feature_num, item.end());
    }
    new_features.insert(new_features.end(), item.begin(), item.end());
  }

  int detected_new_features = new_features.size();

  // Find the stereo matched points for the newly
  // detected features.
  vector<cv::Point2f> cam0_points(new_features.size());
  for (int i = 0; i < new_features.size(); ++i)
    cam0_points[i] = new_features[i].pt;

  vector<cv::Point2f> cam1_points(0);
  vector<unsigned char> inlier_markers(0);
  stereoMatch(cam0_points, cam1_points, inlier_markers);

  vector<cv::Point2f> cam0_inliers(0);
  vector<cv::Point2f> cam1_inliers(0);
  vector<float> response_inliers(0);
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cam0_inliers.push_back(cam0_points[i]);
    cam1_inliers.push_back(cam1_points[i]);
    response_inliers.push_back(new_features[i].response);
  }

  int matched_new_features = cam0_inliers.size();

  if (matched_new_features < 5 &&
      static_cast<double>(matched_new_features)/
      static_cast<double>(detected_new_features) < 0.1)
    RCLCPP_WARN(this->get_logger(), "Images at [%f] seems unsynced...",
        rclcpp::Time(cam0_curr_img_ptr->header.stamp).seconds());

  // Group the features into grids
  GridFeatures grid_new_features;
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code)
      grid_new_features[code] = vector<FeatureMetaData>(0);

  for (int i = 0; i < cam0_inliers.size(); ++i) {
    const cv::Point2f& cam0_point = cam0_inliers[i];
    const cv::Point2f& cam1_point = cam1_inliers[i];
    const float& response = response_inliers[i];

    int row = static_cast<int>(cam0_point.y / grid_height);
    int col = static_cast<int>(cam0_point.x / grid_width);
    int code = row*processor_config.grid_col + col;

    FeatureMetaData new_feature;
    new_feature.response = response;
    new_feature.cam0_point = cam0_point;
    new_feature.cam1_point = cam1_point;
    grid_new_features[code].push_back(new_feature);
  }

  // Sort the new features in each grid based on its response.
  for (auto& item : grid_new_features)
    std::sort(item.second.begin(), item.second.end(),
        &ImageProcessor::featureCompareByResponse);

  int new_added_feature_num = 0;
  // Collect new features within each grid with high response.
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
    vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

    if (features_this_grid.size() >=
        processor_config.grid_min_feature_num) continue;

    int vacancy_num = processor_config.grid_min_feature_num -
      features_this_grid.size();
    for (int k = 0;
        k < vacancy_num && k < new_features_this_grid.size(); ++k) {
      features_this_grid.push_back(new_features_this_grid[k]);
      features_this_grid.back().id = next_feature_id++;
      features_this_grid.back().lifetime = 1;

      ++new_added_feature_num;
    }
  }

  //printf("\033[0;33m detected: %d; matched: %d; new added feature: %d\033[0m\n",
  //    detected_new_features, matched_new_features, new_added_feature_num);

  return;
}

void ImageProcessor::pruneGridFeatures() {
  for (auto& item : *curr_features_ptr) {
    auto& grid_features = item.second;
    // Continue if the number of features in this grid does
    // not exceed the upper bound.
    if (grid_features.size() <=
        processor_config.grid_max_feature_num) continue;
    std::sort(grid_features.begin(), grid_features.end(),
        &ImageProcessor::featureCompareByLifetime);
    grid_features.erase(grid_features.begin()+
        processor_config.grid_max_feature_num,
        grid_features.end());
  }
  return;
}

void ImageProcessor::undistortPoints(
    const vector<cv::Point2f>& pts_in,  //cam0_points
    const cv::Vec4d& intrinsics,        //cam0_intrinsics
    const string& distortion_model,     //cam0_model
    const cv::Vec4d& distortion_coeffs, //cam0_coeffs
    vector<cv::Point2f>& pts_out,       //cam0_undistorte
    const cv::Matx33d &rectification_matrix,  //R_cam0_cam1
    const cv::Vec4d &new_intrinsics) {    //(1,1,0,0)

  if (pts_in.size() == 0) return;

  const cv::Matx33d K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);

  const cv::Matx33d K_new(
      new_intrinsics[0], 0.0, new_intrinsics[2],
      0.0, new_intrinsics[1], new_intrinsics[3],
      0.0, 0.0, 1.0);

  if (distortion_model == "radtan") {
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
  } else if (distortion_model == "equidistant") {
    cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                                 rectification_matrix, K_new);
  } else {
    RCLCPP_WARN_ONCE(this->get_logger(), "The model %s is unrecognized, use radtan instead...",
                  distortion_model.c_str());
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
  }

  return;
}

vector<cv::Point2f> ImageProcessor::distortPoints(
    const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs) {

  const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                      0.0, intrinsics[1], intrinsics[3],
                      0.0, 0.0, 1.0);

  vector<cv::Point2f> pts_out;
  if (distortion_model == "radtan") {
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);
  } else if (distortion_model == "equidistant") {
    cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
  } else {
    RCLCPP_WARN_ONCE(this->get_logger(), "The model %s is unrecognized, using radtan instead...",
                  distortion_model.c_str());
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);
  }

  return pts_out;
}

void ImageProcessor::integrateImuData(
    Matx33f& cam0_R_p_c, Matx33f& cam1_R_p_c) {
  // Find the start and the end limit within the imu msg buffer.
  auto begin_iter = imu_msg_buffer.begin();
  while (begin_iter != imu_msg_buffer.end()) {
    if (stamp2sec(stampMinus(begin_iter->header.stamp,
          cam0_prev_img_ptr->header.stamp) ) < -0.01)
      ++begin_iter;
    else
      break;
  }

  auto end_iter = begin_iter;
  while (end_iter != imu_msg_buffer.end()) {
    if (stamp2sec(stampMinus(end_iter->header.stamp,
          cam0_curr_img_ptr->header.stamp)) < 0.005)
      ++end_iter;
    else
      break;
  }

  // Compute the mean angular velocity in the IMU frame.
  Vec3f mean_ang_vel(0.0, 0.0, 0.0);
  for (auto iter = begin_iter; iter < end_iter; ++iter)
    mean_ang_vel += Vec3f(iter->angular_velocity.x,
        iter->angular_velocity.y, iter->angular_velocity.z);

  if (end_iter-begin_iter > 0)
    mean_ang_vel *= 1.0f / (end_iter-begin_iter);

  // Transform the mean angular velocity from the IMU
  // frame to the cam0 and cam1 frames.
  Vec3f cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;
  Vec3f cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

  // Compute the relative rotation.
  double dtime = stamp2sec(stampMinus(cam0_curr_img_ptr->header.stamp,
      cam0_prev_img_ptr->header.stamp));
  Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);
  Rodrigues(cam1_mean_ang_vel*dtime, cam1_R_p_c);
  cam0_R_p_c = cam0_R_p_c.t();
  cam1_R_p_c = cam1_R_p_c.t();

  // Delete the useless and used imu messages.
  imu_msg_buffer.erase(imu_msg_buffer.begin(), end_iter);
  return;
}

void ImageProcessor::rescalePoints(
    vector<Point2f>& pts1, vector<Point2f>& pts2,
    float& scaling_factor) {

  scaling_factor = 0.0f;

  for (int i = 0; i < pts1.size(); ++i) {
    scaling_factor += sqrt(pts1[i].dot(pts1[i]));
    scaling_factor += sqrt(pts2[i].dot(pts2[i]));
  }

  scaling_factor = (pts1.size()+pts2.size()) /
    scaling_factor * sqrt(2.0f);

  for (int i = 0; i < pts1.size(); ++i) {
    pts1[i] *= scaling_factor;
    pts2[i] *= scaling_factor;
  }

  return;
}

void ImageProcessor::twoPointRansac(
    const vector<Point2f>& pts1, const vector<Point2f>& pts2,
    const cv::Matx33f& R_p_c, const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    const double& inlier_error,
    const double& success_probability,
    vector<int>& inlier_markers) {

  // Check the size of input point size.
  if (pts1.size() != pts2.size())
    RCLCPP_ERROR(this->get_logger(), "Sets of different size (%lu and %lu) are used...",
        pts1.size(), pts2.size());

  double norm_pixel_unit = 2.0 / (intrinsics[0]+intrinsics[1]);
  int iter_num = static_cast<int>(
      ceil(log(1-success_probability) / log(1-0.7*0.7)));

  // Initially, mark all points as inliers.
  inlier_markers.clear();
  inlier_markers.resize(pts1.size(), 1);

  // Undistort all the points.
  vector<Point2f> pts1_undistorted(pts1.size());
  vector<Point2f> pts2_undistorted(pts2.size());
  undistortPoints(
      pts1, intrinsics, distortion_model,
      distortion_coeffs, pts1_undistorted);
  undistortPoints(
      pts2, intrinsics, distortion_model,
      distortion_coeffs, pts2_undistorted);

  // Compenstate the points in the previous image with
  // the relative rotation.
  for (auto& pt : pts1_undistorted) {
    Vec3f pt_h(pt.x, pt.y, 1.0f);
    //Vec3f pt_hc = dR * pt_h;
    Vec3f pt_hc = R_p_c * pt_h;
    pt.x = pt_hc[0];
    pt.y = pt_hc[1];
  }

  // Normalize the points to gain numerical stability.
  float scaling_factor = 0.0f;
  rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
  norm_pixel_unit *= scaling_factor;

  // Compute the difference between previous and current points,
  // which will be used frequently later.
  vector<Point2d> pts_diff(pts1_undistorted.size());
  for (int i = 0; i < pts1_undistorted.size(); ++i)
    pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];

  // Mark the point pairs with large difference directly.
  // BTW, the mean distance of the rest of the point pairs
  // are computed.
  double mean_pt_distance = 0.0;
  int raw_inlier_cntr = 0;
  for (int i = 0; i < pts_diff.size(); ++i) {
    double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
    // 25 pixel distance is a pretty large tolerance for normal motion.
    // However, to be used with aggressive motion, this tolerance should
    // be increased significantly to match the usage.
    if (distance > 50.0*norm_pixel_unit) {
      inlier_markers[i] = 0;
    } else {
      mean_pt_distance += distance;
      ++raw_inlier_cntr;
    }
  }
  mean_pt_distance /= raw_inlier_cntr;

  // If the current number of inliers is less than 3, just mark
  // all input as outliers. This case can happen with fast
  // rotation where very few features are tracked.
  if (raw_inlier_cntr < 3) {
    for (auto& marker : inlier_markers) marker = 0;
    return;
  }

  // Before doing 2-point RANSAC, we have to check if the motion
  // is degenerated, meaning that there is no translation between
  // the frames, in which case, the model of the RANSAC does not
  // work. If so, the distance between the matched points will
  // be almost 0.
  //if (mean_pt_distance < inlier_error*norm_pixel_unit) {
  if (mean_pt_distance < norm_pixel_unit) {
    //ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
    for (int i = 0; i < pts_diff.size(); ++i) {
      if (inlier_markers[i] == 0) continue;
      if (sqrt(pts_diff[i].dot(pts_diff[i])) >
          inlier_error*norm_pixel_unit)
        inlier_markers[i] = 0;
    }
    return;
  }

  // In the case of general motion, the RANSAC model can be applied.
  // The three column corresponds to tx, ty, and tz respectively.
  MatrixXd coeff_t(pts_diff.size(), 3);
  for (int i = 0; i < pts_diff.size(); ++i) {
    coeff_t(i, 0) = pts_diff[i].y;
    coeff_t(i, 1) = -pts_diff[i].x;
    coeff_t(i, 2) = pts1_undistorted[i].x*pts2_undistorted[i].y -
      pts1_undistorted[i].y*pts2_undistorted[i].x;
  }

  vector<int> raw_inlier_idx;
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] != 0)
      raw_inlier_idx.push_back(i);
  }

  vector<int> best_inlier_set;
  double best_error = 1e10;

  RandomGenerators random_generator;

  for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx) {
    // Randomly select two point pairs.
    // Although this is a weird way of selecting two pairs, but it
    // is able to efficiently avoid selecting repetitive pairs.
    int select_idx1 = random_generator.uniformInteger(0, raw_inlier_idx.size()-1);
    int select_idx_diff = random_generator.uniformInteger(1, raw_inlier_idx.size()-1);
    int select_idx2 = select_idx1+select_idx_diff<raw_inlier_idx.size() ?
      select_idx1+select_idx_diff :
      select_idx1+select_idx_diff-raw_inlier_idx.size();

    int pair_idx1 = raw_inlier_idx[select_idx1];
    int pair_idx2 = raw_inlier_idx[select_idx2];

    // Construct the model;
    Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
    Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
    Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
    vector<double> coeff_l1_norm(3);
    coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
    coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
    coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
    int base_indicator = min_element(coeff_l1_norm.begin(),
        coeff_l1_norm.end())-coeff_l1_norm.begin();

    Vector3d model(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      Matrix2d A;
      A << coeff_ty, coeff_tz;
      Vector2d solution = A.inverse() * (-coeff_tx);
      model(0) = 1.0;
      model(1) = solution(0);
      model(2) = solution(1);
    } else if (base_indicator ==1) {
      Matrix2d A;
      A << coeff_tx, coeff_tz;
      Vector2d solution = A.inverse() * (-coeff_ty);
      model(0) = solution(0);
      model(1) = 1.0;
      model(2) = solution(1);
    } else {
      Matrix2d A;
      A << coeff_tx, coeff_ty;
      Vector2d solution = A.inverse() * (-coeff_tz);
      model(0) = solution(0);
      model(1) = solution(1);
      model(2) = 1.0;
    }

    // Find all the inliers among point pairs.
    VectorXd error = coeff_t * model;

    vector<int> inlier_set;
    for (int i = 0; i < error.rows(); ++i) {
      if (inlier_markers[i] == 0) continue;
      if (std::abs(error(i)) < inlier_error*norm_pixel_unit)
        inlier_set.push_back(i);
    }

    // If the number of inliers is small, the current
    // model is probably wrong.
    if (inlier_set.size() < 0.2*pts1_undistorted.size())
      continue;

    // Refit the model using all of the possible inliers.
    VectorXd coeff_tx_better(inlier_set.size());
    VectorXd coeff_ty_better(inlier_set.size());
    VectorXd coeff_tz_better(inlier_set.size());
    for (int i = 0; i < inlier_set.size(); ++i) {
      coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
      coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
      coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
    }

    Vector3d model_better(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_ty_better, coeff_tz_better;
      Vector2d solution =
          (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
      model_better(0) = 1.0;
      model_better(1) = solution(0);
      model_better(2) = solution(1);
    } else if (base_indicator ==1) {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_tx_better, coeff_tz_better;
      Vector2d solution =
          (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
      model_better(0) = solution(0);
      model_better(1) = 1.0;
      model_better(2) = solution(1);
    } else {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_tx_better, coeff_ty_better;
      Vector2d solution =
          (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
      model_better(0) = solution(0);
      model_better(1) = solution(1);
      model_better(2) = 1.0;
    }

    // Compute the error and upate the best model if possible.
    VectorXd new_error = coeff_t * model_better;

    double this_error = 0.0;
    for (const auto& inlier_idx : inlier_set)
      this_error += std::abs(new_error(inlier_idx));
    this_error /= inlier_set.size();

    if (inlier_set.size() > best_inlier_set.size()) {
      best_error = this_error;
      best_inlier_set = inlier_set;
    }
  }

  // Fill in the markers.
  inlier_markers.clear();
  inlier_markers.resize(pts1.size(), 0);
  for (const auto& inlier_idx : best_inlier_set)
    inlier_markers[inlier_idx] = 1;

  //printf("inlier ratio: %lu/%lu\n",
  //    best_inlier_set.size(), inlier_markers.size());

  return;
}

void ImageProcessor::publish() {

  // Publish features.
  custom_msgs::msg::CameraMeasurement::SharedPtr feature_msg_ptr(new custom_msgs::msg::CameraMeasurement);
  feature_msg_ptr->header.stamp = cam0_curr_img_ptr->header.stamp;

  vector<FeatureIDType> curr_ids(0);
  vector<Point2f> curr_cam0_points(0);
  vector<Point2f> curr_cam1_points(0);

  for (const auto& grid_features : (*curr_features_ptr)) {
    for (const auto& feature : grid_features.second) {
      curr_ids.push_back(feature.id);
      curr_cam0_points.push_back(feature.cam0_point);
      curr_cam1_points.push_back(feature.cam1_point);
    }
  }

  vector<Point2f> curr_cam0_points_undistorted(0);
  vector<Point2f> curr_cam1_points_undistorted(0);

  undistortPoints(
      curr_cam0_points, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, curr_cam0_points_undistorted);
  undistortPoints(
      curr_cam1_points, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, curr_cam1_points_undistorted);

  for (int i = 0; i < curr_ids.size(); ++i) {
    feature_msg_ptr->features.push_back(custom_msgs::msg::FeatureMeasurement());
    feature_msg_ptr->features[i].id = curr_ids[i];
    feature_msg_ptr->features[i].u0 = curr_cam0_points_undistorted[i].x;
    feature_msg_ptr->features[i].v0 = curr_cam0_points_undistorted[i].y;
    feature_msg_ptr->features[i].u1 = curr_cam1_points_undistorted[i].x;
    feature_msg_ptr->features[i].v1 = curr_cam1_points_undistorted[i].y;
  }

  feature_pub->publish(feature_msg_ptr);

  // Publish tracking info.
  custom_msgs::msg::TrackingInfo::SharedPtr tracking_info_msg_ptr(new custom_msgs::msg::TrackingInfo());
  tracking_info_msg_ptr->header.stamp = cam0_curr_img_ptr->header.stamp;
  tracking_info_msg_ptr->before_tracking = before_tracking;
  tracking_info_msg_ptr->after_tracking = after_tracking;
  tracking_info_msg_ptr->after_matching = after_matching;
  tracking_info_msg_ptr->after_ransac = after_ransac;
  tracking_info_pub->publish(tracking_info_msg_ptr);

  return;
}

void ImageProcessor::drawFeaturesMono() {
  // Colors for different features.
  Scalar tracked(0, 255, 0);
  Scalar new_feature(0, 255, 255);

  static int grid_height =
    cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =
    cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  // Create an output image.
  int img_height = cam0_curr_img_ptr->image.rows;
  int img_width = cam0_curr_img_ptr->image.cols;
  Mat out_img(img_height, img_width, CV_8UC3);
  cvtColor(cam0_curr_img_ptr->image, out_img, CV_GRAY2RGB);

  // Draw grids on the image.
  for (int i = 1; i < processor_config.grid_row; ++i) {
    Point pt1(0, i*grid_height);
    Point pt2(img_width, i*grid_height);
    line(out_img, pt1, pt2, Scalar(255, 0, 0));
  }
  for (int i = 1; i < processor_config.grid_col; ++i) {
    Point pt1(i*grid_width, 0);
    Point pt2(i*grid_width, img_height);
    line(out_img, pt1, pt2, Scalar(255, 0, 0));
  }

  // Collect features ids in the previous frame.
  vector<FeatureIDType> prev_ids(0);
  for (const auto& grid_features : *prev_features_ptr)
    for (const auto& feature : grid_features.second)
      prev_ids.push_back(feature.id);

  // Collect feature points in the previous frame.
  map<FeatureIDType, Point2f> prev_points;
  for (const auto& grid_features : *prev_features_ptr)
    for (const auto& feature : grid_features.second)
      prev_points[feature.id] = feature.cam0_point;

  // Collect feature points in the current frame.
  map<FeatureIDType, Point2f> curr_points;
  for (const auto& grid_features : *curr_features_ptr)
    for (const auto& feature : grid_features.second)
      curr_points[feature.id] = feature.cam0_point;

  // Draw tracked features.
  for (const auto& id : prev_ids) {
    if (prev_points.find(id) != prev_points.end() &&
        curr_points.find(id) != curr_points.end()) {
      cv::Point2f prev_pt = prev_points[id];
      cv::Point2f curr_pt = curr_points[id];
      circle(out_img, curr_pt, 3, tracked);
      line(out_img, prev_pt, curr_pt, tracked, 1);

      prev_points.erase(id);
      curr_points.erase(id);
    }
  }

  // Draw new features.
  for (const auto& new_curr_point : curr_points) {
    cv::Point2f pt = new_curr_point.second;
    circle(out_img, pt, 3, new_feature, -1);
  }

  imshow("Feature", out_img);
  waitKey(5);
}

void ImageProcessor::drawFeaturesStereo() {

  if(debug_stereo_pub.getNumSubscribers() > 0)
  {
    // Colors for different features.
    Scalar tracked(0, 255, 0);
    Scalar new_feature(0, 255, 255);

    static int grid_height =
      cam0_curr_img_ptr->image.rows / processor_config.grid_row;
    static int grid_width =
      cam0_curr_img_ptr->image.cols / processor_config.grid_col;

    // Create an output image.
    int img_height = cam0_curr_img_ptr->image.rows;
    int img_width = cam0_curr_img_ptr->image.cols;
    Mat out_img(img_height, img_width*2, CV_8UC3);
    cvtColor(cam0_curr_img_ptr->image,
             out_img.colRange(0, img_width), CV_GRAY2RGB);
    cvtColor(cam1_curr_img_ptr->image,
             out_img.colRange(img_width, img_width*2), CV_GRAY2RGB);

    // Draw grids on the image.
    for (int i = 1; i < processor_config.grid_row; ++i) {
      Point pt1(0, i*grid_height);
      Point pt2(img_width*2, i*grid_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i) {
      Point pt1(i*grid_width, 0);
      Point pt2(i*grid_width, img_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i) {
      Point pt1(i*grid_width+img_width, 0);
      Point pt2(i*grid_width+img_width, img_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    vector<FeatureIDType> prev_ids(0);
    for (const auto& grid_features : *prev_features_ptr)
      for (const auto& feature : grid_features.second)
        prev_ids.push_back(feature.id);

    // Collect feature points in the previous frame.
    map<FeatureIDType, Point2f> prev_cam0_points;
    map<FeatureIDType, Point2f> prev_cam1_points;
    for (const auto& grid_features : *prev_features_ptr)
      for (const auto& feature : grid_features.second) {
        prev_cam0_points[feature.id] = feature.cam0_point;
        prev_cam1_points[feature.id] = feature.cam1_point;
      }

    // Collect feature points in the current frame.
    map<FeatureIDType, Point2f> curr_cam0_points;
    map<FeatureIDType, Point2f> curr_cam1_points;
    for (const auto& grid_features : *curr_features_ptr)
      for (const auto& feature : grid_features.second) {
        curr_cam0_points[feature.id] = feature.cam0_point;
        curr_cam1_points[feature.id] = feature.cam1_point;
      }

    // Draw tracked features.
    for (const auto& id : prev_ids) {
      if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
          curr_cam0_points.find(id) != curr_cam0_points.end()) {
        cv::Point2f prev_pt0 = prev_cam0_points[id];
        cv::Point2f prev_pt1 = prev_cam1_points[id] + Point2f(img_width, 0.0);
        cv::Point2f curr_pt0 = curr_cam0_points[id];
        cv::Point2f curr_pt1 = curr_cam1_points[id] + Point2f(img_width, 0.0);

        circle(out_img, curr_pt0, 3, tracked, -1);
        circle(out_img, curr_pt1, 3, tracked, -1);
        line(out_img, prev_pt0, curr_pt0, tracked, 1);
        line(out_img, prev_pt1, curr_pt1, tracked, 1);

        prev_cam0_points.erase(id);
        prev_cam1_points.erase(id);
        curr_cam0_points.erase(id);
        curr_cam1_points.erase(id);
      }
    }

    // Draw new features.
    for (const auto& new_cam0_point : curr_cam0_points) {
      cv::Point2f pt0 = new_cam0_point.second;
      cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
        Point2f(img_width, 0.0);

      circle(out_img, pt0, 3, new_feature, -1);
      circle(out_img, pt1, 3, new_feature, -1);
    }

    cv_bridge::CvImage debug_image(cam0_curr_img_ptr->header, "bgr8", out_img);
    debug_stereo_pub.publish(debug_image.toImageMsg());
  }
  //imshow("Feature", out_img);
  //waitKey(5);

  return;
}

void ImageProcessor::updateFeatureLifetime() {
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features = (*curr_features_ptr)[code];
    for (const auto& feature : features) {
      if (feature_lifetime.find(feature.id) == feature_lifetime.end())
        feature_lifetime[feature.id] = 1;
      else
        ++feature_lifetime[feature.id];
    }
  }

  return;
}

void ImageProcessor::featureLifetimeStatistics() {

  map<int, int> lifetime_statistics;
  for (const auto& data : feature_lifetime) {
    if (lifetime_statistics.find(data.second) ==
        lifetime_statistics.end())
      lifetime_statistics[data.second] = 1;
    else
      ++lifetime_statistics[data.second];
  }

  for (const auto& data : lifetime_statistics)
    cout << data.first << " : " << data.second << endl;

  return;
}

} // end namespace ros2_msckf

// #include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
// RCLCPP_COMPONENTS_REGISTER_NODE(ros2_msckf::ImageProcessor)