#include "ros2_msckf/msckf_vio.h"
#include "ros2_msckf/geo_mag_declination.h"

#include "Eigen/SVD"
#include "Eigen/QR"
#include "Eigen/SparseCore"
#include "Eigen/SPQRSupport"
// #include "eigen_conversions/"
#include "tf2_eigen/tf2_eigen.h"
#include "pcl_conversions/pcl_conversions.h"
// #include "tf2/convert.h"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>

#define K_rad 0.017453292519943295
#define G_CONSTANT 9.80665
#define FUSE_MAG
// #define DECLINATION_FUSE
// #define GPS_FUSE

using namespace std;
using namespace Eigen;

namespace ros2_msckf{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

// MsckfVio::MsckfVio(const rclcpp::NodeOptions& options):
MsckfVio::MsckfVio():
  Node("msckf_vio"), 
  tf_pub(this),
  is_gravity_set(false),
  is_first_img(true),
  ref_mag_strength(0.4),
  mag_strength_gate(0.4),
  mag_acc_gate(0.5),
  mag_yaw_rate_gate(0.25),
  mag_noise(0.05),
  yaw_declination(0.0){
  mag_filter.setAlpha(0.1);
  baro_filter.setAlpha(0.1);
  gps_init = false;
  has_gps = false;
  gps_correct_mag = false;

  acc_com<< 0.999495, 0.001916, 0.003466,
            -0.001772, 0.998882, -0.004759,
             -0.004731, 0.002897, 0.997182;
  acc_bias<<0.000136,0.000175, 0.010692;

  gyro_com<< 0.999746, 0.002423, 0.003045,
            -0.002252, 0.999325, -0.004288,
             -0.003556, 0.003661, 0.999141;
  gyro_bias<<-0.032213,-0.12088, 0.135487;

  gps_vel_innov_gate = 5.0;
  gps_pos_innov_gate = 5.0;
  // fout.open("/home/chao/Documents/traj/ros2_msckf/result.csv", ios::app);
  return;
}

bool MsckfVio::loadParameters() {
  // Frame id
  fixed_frame_id = this->declare_parameter<string>("fixed_frame_id", "world");
  child_frame_id = this->declare_parameter<string>("child_frame_id", "robot");
  publish_tf = this->declare_parameter<bool>("publish_tf", true);
  frame_rate = this->declare_parameter<double>("frame_rate", 40.0);
  position_std_threshold = this->declare_parameter<double>("position_std_threshold", 8.0);
  rotation_threshold = this->declare_parameter<double>("rotation_threshold", 0.2618);
  translation_threshold = this->declare_parameter<double>("translation_threshold", 0.4);
  tracking_rate_threshold = this->declare_parameter<double>("tracking_rate_threshold", 0.5);

  // Feature optimization parameters
  Feature::optimization_config.translation_threshold = 
  this->declare_parameter<double>("feature.config.translation_threshold", 0.2);

  // Noise related parameters
  IMUState::gyro_noise = 
  this->declare_parameter<double>("noise.gyro", 0.001);
  IMUState::acc_noise = 
  this->declare_parameter<double>("noise.acc", 0.01);
  IMUState::gyro_bias_noise = 
  this->declare_parameter<double>("noise.gyro_bias", 0.001);
  IMUState::acc_bias_noise = 
  this->declare_parameter<double>("noise.acc_bias", 0.01);
  Feature::observation_noise = 
  this->declare_parameter<double>("noise.feature", 0.01);

  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?
  state_server.imu_state.velocity[0] =
  this->declare_parameter<double>("initial_state.velocity.x", 0.0);
  state_server.imu_state.velocity[1] = 
  this->declare_parameter<double>("initial_state.velocity.y", 0.0);
  state_server.imu_state.velocity[2] = 
  this->declare_parameter<double>("initial_state.velocity.z", 0.0);

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  double velocity_cov = 
  this->declare_parameter<double>("initial_covariance.velocity", 1e-4);
  double  gyro_bias_cov= 
  this->declare_parameter<double>("initial_covariance.gyro_bias", 1e-2);
  double acc_bias_cov = 
  this->declare_parameter<double>("initial_covariance.acc_bias", 0.25);

  double extrinsic_rotation_cov = 
  this->declare_parameter<double>("initial_covariance.extrinsic_rotation_cov", 3.0462e-4);
  double extrinsic_translation_cov = 
  this->declare_parameter<double>("initial_covariance.extrinsic_translation_cov", 1e-4);

  state_server.state_cov = MatrixXd::Zero(29, 29);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Transformation offsets between the frames involved.
  Isometry3d T_cam0_imu = utils::getTransformEigen(shared_from_this(), "cam0.T_cam_imu");
  // Isometry3d T_cam0_imu = T_imu_cam0.inverse();

  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
  CAMState::T_cam0_cam1 =
    utils::getTransformEigen(shared_from_this(), "cam1.T_cn_cnm1");
  IMUState::T_imu_body =
    utils::getTransformEigen(shared_from_this(), "T_imu_body").inverse();

  float yaw = this->declare_parameter<float>("mag_extrinsic", -0.1067);
  fuse_mag = this->declare_parameter<bool>("fuse_mag", false);
  fuse_3d_mag = this->declare_parameter<bool>("fuse_3d_mag", false);
  fuse_d = this->declare_parameter<bool>("fuse_d", false);
  fuse_gps = this->declare_parameter<bool>("fuse_gps", false);
  mag_heading_noise = this->declare_parameter<double>("mag_heading_noise", 0.5);
  mag_noise = this->declare_parameter<double>("mag_noise", 0.05);
  result_path = this->declare_parameter<string>("save_path","/home/chao/Documents/traj/ros2_msckf/result.csv");
  fout.open(result_path, ios::app);
  Matrix3d R_imu_cam;
  R_imu_cam<<0.00033976, 0.99998887, -0.00470619,
             -0.64505359, 0.00381533, 0.76412781,
             0.76413726, 0.00277613, 0.6450477;
  R_mag_imu = AngleAxisd(yaw, Vector3d(0,0,1.0)).matrix();
  R_mag_imu = T_cam0_imu.linear()*R_imu_cam*R_mag_imu;

  // Maximum number of camera states to be stored
  max_cam_state_size = 30;
  this->declare_parameter<int>("max_cam_state_size", max_cam_state_size);

  RCLCPP_INFO(get_logger(), "===========================================");
  RCLCPP_INFO(get_logger(), "fixed frame id: %s", fixed_frame_id.c_str());
  RCLCPP_INFO(get_logger(), "child frame id: %s", child_frame_id.c_str());
  RCLCPP_INFO(get_logger(), "publish tf: %d", publish_tf);
  RCLCPP_INFO(get_logger(), "frame rate: %f", frame_rate);
  RCLCPP_INFO(get_logger(), "position std threshold: %f", position_std_threshold);
  RCLCPP_INFO(get_logger(), "Keyframe rotation threshold: %f", rotation_threshold);
  RCLCPP_INFO(get_logger(), "Keyframe translation threshold: %f", translation_threshold);
  RCLCPP_INFO(get_logger(), "Keyframe tracking rate threshold: %f", tracking_rate_threshold);
  RCLCPP_INFO(get_logger(), "gyro noise: %.10f", IMUState::gyro_noise);
  RCLCPP_INFO(get_logger(), "gyro bias noise: %.10f", IMUState::gyro_bias_noise);
  RCLCPP_INFO(get_logger(), "acc noise: %.10f", IMUState::acc_noise);
  RCLCPP_INFO(get_logger(), "acc bias noise: %.10f", IMUState::acc_bias_noise);
  RCLCPP_INFO(get_logger(), "observation noise: %.10f", Feature::observation_noise);
  RCLCPP_INFO(get_logger(), "initial velocity: %f, %f, %f",
      state_server.imu_state.velocity(0),
      state_server.imu_state.velocity(1),
      state_server.imu_state.velocity(2));
  RCLCPP_INFO(get_logger(), "initial gyro bias cov: %f", gyro_bias_cov);
  RCLCPP_INFO(get_logger(), "initial acc bias cov: %f", acc_bias_cov);
  RCLCPP_INFO(get_logger(), "initial velocity cov: %f", velocity_cov);
  RCLCPP_INFO(get_logger(), "initial extrinsic rotation cov: %f",
      extrinsic_rotation_cov);
  RCLCPP_INFO(get_logger(), "initial extrinsic translation cov: %f",
      extrinsic_translation_cov);

  cout << T_cam0_imu.linear().transpose() << endl;
  cout << T_cam0_imu.translation() << endl;

  RCLCPP_INFO(get_logger(), "max camera state #: %d", max_cam_state_size);
  RCLCPP_INFO(get_logger(), "fuse mag #: %d", fuse_mag);
  RCLCPP_INFO(get_logger(), "===========================================");
  return true;
}

bool MsckfVio::createRosIO() {
  odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
  feature_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "feature_point_cloud", 10);

  reset_srv = this->create_service<std_srvs::srv::Trigger>("reset",
      std::bind(&MsckfVio::resetCallback, this, std::placeholders::_1, std::placeholders::_2, 
      std::placeholders::_3));

  imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", 10,
      std::bind(&MsckfVio::imuCallback, this, std::placeholders::_1));
  feature_sub = this->create_subscription<custom_msgs::msg::CameraMeasurement>("features", 40,
      std::bind(&MsckfVio::featureCallback, this, std::placeholders::_1));

  mocap_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("mocap_odom", 10,
      std::bind(&MsckfVio::mocapOdomCallback, this, std::placeholders::_1));
  mocap_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("gt_odom", 1);

  mag_sub = this->create_subscription<px4_msgs::msg::VehicleMagnetometer>("VehicleMagnetometer_PubSubTopic", 20,
            std::bind(&MsckfVio::magCallback, this, std::placeholders::_1));

  baro_sub = this->create_subscription<px4_msgs::msg::VehicleAirData>("VehicleAirData_PubSubTopic", 10,
              std::bind(&MsckfVio::baroCallback, this, std::placeholders::_1));

  gps_sub = this->create_subscription<px4_msgs::msg::VehicleGpsPosition>("VehicleGpsPosition_PubSubTopic", 10,
              std::bind(&MsckfVio::gpsCallback, this, std::placeholders::_1));

  return true;
}

bool MsckfVio::initialize() {
  if (!loadParameters()) return false;
  RCLCPP_INFO(get_logger(), "Finish loading ROS parameters...");

  // Initialize state server
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  }

  if (!createRosIO()) return false;
  RCLCPP_INFO(get_logger(), "Finish creating ROS IO...");

  return true;
}

void MsckfVio::imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg) {

  // IMU msgs are pushed backed into a buffer instead of
  // being processed immediately. The IMU msgs are processed
  // when the next image is available, in which way, we can
  // easily handle the transfer delay.
  imu_msg_buffer.push_back(*msg);
  Eigen::Vector3d cur_acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  acc_filter.update(cur_acc);
  Eigen::Vector3d cur_gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  gyro_filter.update(cur_gyro);
  // RCLCPP_INFO(get_logger(), "get imu msg..");
  if (!is_gravity_set) {
    if (imu_msg_buffer.size() < 200) return;
    //if (imu_msg_buffer.size() < 10) return;
    
    is_gravity_set = initializeGravityAndBias();
    // cout<<"initial rotation: "<<state_server.imu_state.orientation.transpose()<<endl;
    // if(is_gravity_set)
      // LOG(INFO) << "initial rotation: "<< state_server.imu_state.orientation.transpose();
  }

  return;
}

bool MsckfVio::initializeGravityAndBias() {

    // Initialize gravity and gyro bias.
    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    for (const auto& imu_msg : imu_msg_buffer) {
      Vector3d angular_vel = Vector3d::Zero();
      Vector3d linear_acc = Vector3d::Zero();

      utils::fromMsg(imu_msg.angular_velocity, angular_vel);
      utils::fromMsg(imu_msg.linear_acceleration, linear_acc);
      // angular_vel = gyro_com*angular_vel + gyro_bias;
      // linear_acc = acc_com*linear_acc + acc_bias;
       
      sum_angular_vel += angular_vel;
      sum_linear_acc += linear_acc;
    }

    state_server.imu_state.gyro_bias =
      sum_angular_vel / imu_msg_buffer.size();
    //IMUState::gravity =
    //  -sum_linear_acc / imu_msg_buffer.size();
    // This is the gravity in the IMU frame.
    Vector3d gravity_imu =
      sum_linear_acc / imu_msg_buffer.size();

    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    double gravity_norm = gravity_imu.norm();
    if(fabs(gravity_norm - G_CONSTANT) > 1.5 )
    {
      if(gps_buffer.size()<2)
      {
        RCLCPP_WARN(this->get_logger(), "acceleration is unstable: %f, and no gps data, failed to initialize!", gravity_norm);
        return false;
      }
      auto last_gps = gps_buffer.end() - 1;
      auto second_last_gps = gps_buffer.end() - 2;
      double dt = stamp2sec((last_gps->timestamp - second_last_gps->timestamp));
      Vector3d mov_acc(last_gps->vel_n_m_s - second_last_gps->vel_n_m_s,
                       last_gps->vel_e_m_s - second_last_gps->vel_e_m_s,
                       last_gps->vel_d_m_s - second_last_gps->vel_d_m_s);
      mov_acc /= dt;
      gravity_imu -= mov_acc;
      gravity_norm = gravity_imu.norm();
      gps_ori_height = last_gps->alt;
      ref_pos_lat = last_gps->lat*K_rad;
      ref_pos_lon = last_gps->lon*K_rad;
      ref_sin_lat = sin(ref_pos_lat);
      ref_cos_lat = cos(ref_pos_lat);
      state_server.state_cov.block<2,2>(27,27) = Matrix2d::Identity()*(last_gps->eph);

      state_server.imu_state.velocity = Vector3d(last_gps->vel_n_m_s,last_gps->vel_e_m_s,last_gps->vel_d_m_s);
      state_server.state_cov.block<3,3>(6,6) = Matrix3d::Identity()*last_gps->s_variance_m_s;

      gps_init = true;
      gps_correct_mag = true;
    }
    IMUState::gravity = Vector3d(0.0, 0.0, gravity_norm);

    Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
      gravity_imu, -IMUState::gravity);
    
    Matrix3d R_i_w = q0_i_w.toRotationMatrix();
    
    //初始化磁力计
#ifdef FUSE_MAG
      if(mag_buffer.size() == 0)
        return false;

      Vector3d Mag_w = R_i_w*R_mag_imu*mag_filter.getState();
      double theta_yaw = -atan2(Mag_w(1), Mag_w(0));
      if(has_gps)
      {
        LOG(INFO)<<"init Mag with yaw_declination, "<< yaw_declination;
        theta_yaw += yaw_declination;
        gps_correct_mag = true;
      }
      AngleAxisd yaw_rot(theta_yaw, Vector3d(0,0,1));
      R_i_w = yaw_rot.toRotationMatrix()*R_i_w;
      state_server.imu_state.mag_ned = yaw_rot.toRotationMatrix()*Mag_w;
      RCLCPP_INFO(this->get_logger(), "mag initialize: %f", theta_yaw);

#endif

    state_server.imu_state.orientation =    //i -> w 旋转，w -> i 坐标变换 q:[x,y,z,w]
      rotationToQuaternion(R_i_w.transpose());
    // cout<<state_server.imu_state.orientation<<endl;
    return true;
}

double MsckfVio::getDeclination()
{
  return gps_correct_mag? yaw_declination:0;
  // return 0.0;
}

void MsckfVio::resetCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {

	// RCLCPP_WARN(get_logger(), "Start resetting msckf vio...");
	// Temporarily shutdown the subscribers to prevent the
	// state from updating.
  (void)request_header;
	feature_sub.reset();
	imu_sub.reset();

	// Reset the IMU state.
	IMUState& imu_state = state_server.imu_state;
	imu_state.time = 0.0;
	imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
	imu_state.position = Vector3d::Zero();
	imu_state.velocity = Vector3d::Zero();
	imu_state.gyro_bias = Vector3d::Zero();
	imu_state.acc_bias = Vector3d::Zero();
	imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
	imu_state.position_null = Vector3d::Zero();
	imu_state.velocity_null = Vector3d::Zero();

	// Remove all existing camera states.
	state_server.cam_states.clear();

	// Reset the state covariance.
	double gyro_bias_cov = 1e-4;
	double acc_bias_cov = 1e-2;
	double velocity_cov = 0.25;
	this->declare_parameter<double>("initial_covariance/velocity", velocity_cov);
	this->declare_parameter<double>("initial_covariance/gyro_bias",gyro_bias_cov);
	this->declare_parameter<double>("initial_covariance/acc_bias", acc_bias_cov);

	double extrinsic_rotation_cov = 3.0462e-4;
	double extrinsic_translation_cov = 1e-4;
	this->declare_parameter<double>("initial_covariance/extrinsic_rotation_cov",
		extrinsic_rotation_cov);
	this->declare_parameter<double>("initial_covariance/extrinsic_translation_cov",
		extrinsic_translation_cov);

	state_server.state_cov = MatrixXd::Zero(29, 29);
	for (int i = 3; i < 6; ++i)
		state_server.state_cov(i, i) = gyro_bias_cov;
	for (int i = 6; i < 9; ++i)
		state_server.state_cov(i, i) = velocity_cov;
	for (int i = 9; i < 12; ++i)
		state_server.state_cov(i, i) = acc_bias_cov;
	for (int i = 15; i < 18; ++i)
		state_server.state_cov(i, i) = extrinsic_rotation_cov;
	for (int i = 18; i < 21; ++i)
		state_server.state_cov(i, i) = extrinsic_translation_cov;

	// Clear all exsiting features in the map.
	map_server.clear();

	// Clear the IMU msg buffer.
	imu_msg_buffer.clear();

	// Reset the starting flags.
	is_gravity_set = false;
	is_first_img = true;

	// Restart the subscribers.
	imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", 10,
		std::bind(&MsckfVio::imuCallback, this, std::placeholders::_1));
	feature_sub = this->create_subscription<custom_msgs::msg::CameraMeasurement>("features", 10 ,
		std::bind(&MsckfVio::featureCallback, this, std::placeholders::_1));

	// TODO: When can the reset fail?
	res->success = true;
	RCLCPP_WARN(get_logger(), "Resetting msckf vio completed...");
	// return true;
}

void MsckfVio::magCallback(const px4_msgs::msg::VehicleMagnetometer::SharedPtr msg)
{
  Vector3d mag(msg->magnetometer_ga[0], msg->magnetometer_ga[1], msg->magnetometer_ga[2]);
  MagMsg mag_data;
  mag_data.magnetometer_ga = mag;
  mag_data.timestamp = stamp2sec(msg->timestamp);
  mag_buffer.emplace_back(mag_data);
  mag_filter.update(mag);
}

void MsckfVio::baroCallback(const px4_msgs::msg::VehicleAirData::SharedPtr msg)
{
  double height = msg->baro_alt_meter;
  baro_filter.update(height);
  if(is_gravity_set && fabs(baro_ori_height)<1e-6)
  {
    baro_ori_height = msg->baro_alt_meter - state_server.imu_state.position(2);
  }
}


bool MsckfVio::checkGpsGood(px4_msgs::msg::VehicleGpsPosition::SharedPtr msg)
{
  bool gps_sats_status = msg->satellites_used>11;
  bool gps_fix_status = msg->fix_type>2;
  bool gps_eph_status = msg->eph < 3;
  bool gps_sacc_status = msg->s_variance_m_s < 0.5;

  LOG(INFO)<<"gps: "<<msg->timestamp<<" "<<int(msg->satellites_used)<<" "<<int(msg->fix_type)<<" "<<msg->eph<<" "<<msg->s_variance_m_s;
  // return true;
  return (gps_sats_status && 
          gps_fix_status && 
          gps_eph_status &&
          gps_sacc_status);
}

void MsckfVio::gpsCallback(const px4_msgs::msg::VehicleGpsPosition::SharedPtr msg)
{
  if(!has_gps && !gps_correct_mag && msg->satellites_used>6)
  {
    double lat = msg->lat*1e-7;
    double lon = msg->lon*1e-7;
    yaw_declination = get_mag_declination_radians(lat, lon);
    mag_inclination = get_mag_inclination_radians(lat, lon);
    ref_mag_strength = get_mag_strength_gauss(lat, lon);
    LOG(INFO)<<"get yaw_declination, "<<yaw_declination;
    has_gps = true;
  }

  if(checkGpsGood(msg))
  {
    if(!gps_init && is_gravity_set)
    {
      gps_init = true;
      gps_ori_height = msg->alt - state_server.imu_state.position(2);
      ref_pos_lat = (msg->lat*1e-7)*K_rad;
      ref_pos_lon = (msg->lon*1e-7)*K_rad;
      ref_sin_lat = sin(ref_pos_lat);
      ref_cos_lat = cos(ref_pos_lat);
      map_projection_reproject();
      // state_server.state_cov.block<2,2>(27,27) = Matrix2d::Identity()*msg->eph;
    }
    else
    {
      gps_buffer.emplace_back(*msg);
    }
  }
}
// p_projection_reproject(&_pos_ref, -_state.pos(0), -_state.pos(1), &est_lat, &est_lon);
int MsckfVio::map_projection_reproject()
{
  Vector2d pos = state_server.imu_state.position.head<2>();
	const double x_rad = -pos(0) / CONSTANTS_RADIUS_OF_EARTH;
	const double y_rad = -pos(1) / CONSTANTS_RADIUS_OF_EARTH;
	const double c = sqrt(x_rad * x_rad + y_rad * y_rad);
  double new_pos_lat;
  double new_pos_lon;
	if (fabs(c) > 0) {
		const double sin_c = sin(c);
		const double cos_c = cos(c);

		const double new_pos_lat = asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c);
		const double new_pos_lon = (ref_pos_lon + atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c));

    ref_pos_lon = new_pos_lon;
    ref_pos_lat = new_pos_lat;

    ref_sin_lat = sin(ref_pos_lat);
    ref_cos_lat = cos(ref_pos_lat);

	}

	return 0;
}

void MsckfVio::gpsUpdate(px4_msgs::msg::VehicleGpsPosition& gps_sample)
{
  //限制观测误差幅度
  Vector2d measure_pos(0,0);
  //经纬度投影到NED系
	const double lat_rad = (gps_sample.lat*1e-7)*K_rad;
	const double lon_rad = (gps_sample.lon*1e-7)*K_rad;

	const double sin_lat = sin(lat_rad);
	const double cos_lat = cos(lat_rad);

	const double cos_d_lon = cos(lon_rad - ref_pos_lon);

	const double arg = constrain(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0,  1.0);
	const double c = acos(arg);

	double k = 1.0;

	if (fabs(c) > 0) {
		k = (c / sin(c));
	}

	measure_pos(0) = static_cast<float>(k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * CONSTANTS_RADIUS_OF_EARTH);
	measure_pos(1) = static_cast<float>(k * cos_lat * sin(lon_rad - ref_pos_lon) * CONSTANTS_RADIUS_OF_EARTH);

  Vector3d measure_v(gps_sample.vel_n_m_s, gps_sample.vel_e_m_s, gps_sample.vel_d_m_s);

  double pos_cov = max(double(gps_sample.eph), 0.5);
  double vxy_cov = max(double(gps_sample.s_variance_m_s), 0.3);
  double vz_cov = pow(1.5,2)*vxy_cov;
  LOG(INFO)<<"gps_pos: "<<gps_sample.timestamp<<" "<<measure_pos(0)<<" "<<measure_pos(1);
  LOG(INFO)<<"gps_V: "<<gps_sample.timestamp<<" "<<measure_v(0)<<" "<<measure_v(1)<<" "<<measure_v(2);
  
  Matrix<double, 5, 1> r = Matrix<double, 5, 1>::Zero();
  r.block<2,1>(0,0) = measure_pos - state_server.imu_state.position.head<2>();
  r.block<3,1>(2,0) = measure_v - state_server.imu_state.velocity;

  LOG(INFO)<<"gps_r: "<<gps_sample.timestamp<<" "<<r(0,0)<<" "<<r(1,0)<<" "<<r(2,0)<<" "<<r(3,0)<<" "<<r(4,0);
  
  MatrixXd &s_cov = state_server.state_cov;
  Vector3d innor_v_var(s_cov(6,6)+vxy_cov, s_cov(7,7)+vxy_cov, s_cov(8,8)+vz_cov);
  Vector2d p_innor_var(s_cov(12, 12) + pos_cov, s_cov(12,12)+pos_cov);
  double v_test_ratio = max(pow(r(2),2)/(pow(gps_vel_innov_gate,2)*innor_v_var(0)),
                          max(pow(r(3),2)/(pow(gps_vel_innov_gate,2)*innor_v_var(1)),
                              pow(r(4),2)/(pow(gps_vel_innov_gate,2)*innor_v_var(2))));
  double p_test_ratio  = max(pow(r(0),2)/(pow(gps_pos_innov_gate,2)*p_innor_var(0)),
                             pow(r(1),2)/(pow(gps_pos_innov_gate,2)*p_innor_var(1)));

  double test_ratio = max(v_test_ratio, p_test_ratio);

  bool update = 0;

  if(fuse_gps && test_ratio<=1.0)
  {
    update = 1;
    RCLCPP_INFO(this->get_logger(), "update GPS !!");
    MatrixXd H_gps = MatrixXd::Zero(5, 29+state_server.cam_states.size());
    H_gps.block<2,2>(0,12) = Matrix2d::Identity();
    // H_gps.block<2,2>(0,27) = Matrix2d::Identity()*(-1);
    H_gps.block<3,3>(2, 6) = Matrix3d::Identity();

    Matrix<double, 5, 5> n_gps = Matrix<double, 5, 5>::Zero();
    n_gps.block<2,2>(0,0) = pos_cov*Matrix2d::Identity();
    n_gps.block<2,2>(2,2) = vxy_cov*Matrix2d::Identity();
    n_gps(4,4) = vz_cov;

    // auto &state_cov = state_server.state_cov;
    // MatrixXd Q = H_gps*stat_cov*H_gps.transpose();
    // Matrix<double, 5, 1> r = Matrix<double, 5, 1>::Zero();
    // r.block<2,1>(0,0) = measure_pos - state_server.imu_state.position.head<2>();
    // r.block<3,1>(2,0) = measure_v - state_server.imu_state.velocity;

    comMeasurementUpdate(H_gps, r, n_gps);
    onlineReset(2);
  }

  LOG(INFO)<<"gps_update: "<<gps_sample.timestamp<<" "<<update;
  //计算K r更新，对r进行限制
	// return 0;
}

void MsckfVio::reinitMag()
{
  LOG(INFO)<<"Enter reinit Mag";
  Matrix3d R_M_ned  = AngleAxisd(yaw_declination, Vector3d(0, 0, 1)).toRotationMatrix();
  const Vector4d dq(cos(yaw_declination/2.0), 0, 0, sin(yaw_declination/2.0));
  state_server.imu_state.orientation = quaternionMultiplication(
      dq, state_server.imu_state.orientation);

  state_server.imu_state.velocity = R_M_ned*state_server.imu_state.velocity;

  state_server.imu_state.position = R_M_ned*state_server.imu_state.position;

  state_server.imu_state.mag_ned = R_M_ned*state_server.imu_state.mag_ned;
  auto &gps_bias = state_server.imu_state.gps_bias;
  Vector3d pos_bias = R_M_ned*Vector3d(gps_bias(0), gps_bias(1), 0);
  gps_bias = pos_bias.head<2>();

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq, cam_state_iter->second.orientation);
    cam_state_iter->second.position = R_M_ned*cam_state_iter->second.position;
  }
}

void MsckfVio::featureCallback(const custom_msgs::msg::CameraMeasurement::SharedPtr msg) 
{

	// Return if the gravity vector has not been set.
  // RCLCPP_INFO(get_logger(), "feature callback...");
	if (!is_gravity_set) return;

	// Start the system if the first image is received.
	// The frame where the first image is received will be
	// the origin.
	if (is_first_img) {
		is_first_img = false;
    double msg_time = stamp2sec(msg->header.stamp);
		state_server.imu_state.time = msg_time;
    last_mag_time = msg_time;
	}

	static double max_processing_time = 0.0;
	static int critical_time_cntr = 0;
	double processing_start_time = clocker.now().seconds();

	// Propogate the IMU state.
	// that are received before the image msg.
	rclcpp::Time start_time = clocker.now();
	batchImuProcessing(stamp2sec(msg->header.stamp));
	double imu_processing_time = (clocker.now() - start_time).seconds();
  // cout.precision(10)
  // LOG(INFO)<<"predict_P: "<<state_server.imu_state.time<<" "
  //          <<state_server.imu_state.position[0]<<" "
  //          <<state_server.imu_state.position[1]<<" "
  //          <<state_server.imu_state.position[2];
  // // IOFormat nosetwidth(StreamPrecision, DontAlignCols);   //设置Eigen output not align or the float number will bi： 00-00.0232
  // LOG(INFO)<<"predict_V: "<<state_server.imu_state.time<<" "
  //          <<state_server.imu_state.velocity[0]<<" "
  //          <<state_server.imu_state.velocity[1]<<" "
  //          <<state_server.imu_state.velocity[2];

	// Augment the state vector.
	start_time = clocker.now();
	stateAugmentation(stamp2sec(msg->header.stamp));
	double state_augmentation_time = (clocker.now()-start_time).seconds();

	// Add new observations for existing features or new
	// features in the map server.
	start_time = clocker.now();
	addFeatureObservations(msg);
	double add_observations_time = (clocker.now()-start_time).seconds();

	// Perform measurement update if necessary.
	start_time = clocker.now();
	removeLostFeatures();
	double remove_lost_features_time = (clocker.now()-start_time).seconds();
  LOG(INFO)<<"correct_P: "<<state_server.imu_state.time<<" "
           <<state_server.imu_state.position[0]<<" "
           <<state_server.imu_state.position[1]<<" "
           <<state_server.imu_state.position[2];

  LOG(INFO)<<"correct_V: "<<state_server.imu_state.time<<" "
           <<state_server.imu_state.velocity[0]<<" "
           <<state_server.imu_state.velocity[1]<<" "
           <<state_server.imu_state.velocity[2];

	start_time = clocker.now();
	pruneCamStateBuffer();
	double prune_cam_states_time = (clocker.now()-start_time).seconds();

  if((!gps_correct_mag)&& has_gps)
  {
    reinitMag();
    gps_correct_mag = true;
  }
  auto &s = state_server.imu_state;
  fout.setf(ios::fixed, ios::floatfield);
  fout.precision(5);  //1e9
  fout<<state_server.imu_state.time<<" "
      <<s.position(0)<<" "
      <<s.position(1)<<" "
      <<s.position(2)<<" "
      <<s.orientation.x()<<" "
      <<s.orientation.y()<<" "
      <<s.orientation.z()<<" "
      <<s.orientation.w()<<endl;

	// Publish the odometry.
	start_time = clocker.now();
	publish(msg->header.stamp);
	double publish_time = (clocker.now()-start_time).seconds();

	// Reset the system if necessary.
	onlineReset(1);

	double processing_end_time = clocker.now().seconds();
	double processing_time =
		processing_end_time - processing_start_time;
	if (processing_time > 1.0/frame_rate) {
		++critical_time_cntr;
		RCLCPP_INFO(get_logger(), "\033[1;31mTotal processing time %f/%d...\033[0m",
			processing_time, critical_time_cntr);
		//printf("IMU processing time: %f/%f\n",
		//    imu_processing_time, imu_processing_time/processing_time);
		//printf("State augmentation time: %f/%f\n",
		//    state_augmentation_time, state_augmentation_time/processing_time);
		//printf("Add observations time: %f/%f\n",
		//    add_observations_time, add_observations_time/processing_time);
		printf("Remove lost features time: %f/%f\n",
			remove_lost_features_time, remove_lost_features_time/processing_time);
		printf("Remove camera states time: %f/%f\n",
			prune_cam_states_time, prune_cam_states_time/processing_time);
		//printf("Publish time: %f/%f\n",
		//    publish_time, publish_time/processing_time);
	}

	return;
}

void MsckfVio::mocapOdomCallback(
	const nav_msgs::msg::Odometry::SharedPtr msg) {
	static bool first_mocap_odom_msg = true;

	// If this is the first mocap odometry messsage, set
	// the initial frame.
	if (first_mocap_odom_msg) {
		Quaterniond orientation;
		Vector3d translation;
		tf2::fromMsg(
			msg->pose.pose.position, translation);
		tf2::fromMsg(
			msg->pose.pose.orientation, orientation);
		//tf::vectorMsgToEigen(
		//    msg->transform.translation, translation);
		//tf::quaternionMsgToEigen(
		//    msg->transform.rotation, orientation);
		mocap_initial_frame.linear() = orientation.toRotationMatrix();
		mocap_initial_frame.translation() = translation;
		first_mocap_odom_msg = false;
	}

	// Transform the ground truth.
	Quaterniond orientation;
	Vector3d translation;
	//tf::vectorMsgToEigen(
	//    msg->transform.translation, translation);
	//tf::quaternionMsgToEigen(
	//    msg->transform.rotation, orientation);
	tf2::fromMsg(
		msg->pose.pose.position, translation);
	tf2::fromMsg(
		msg->pose.pose.orientation, orientation);

	Eigen::Isometry3d T_b_v_gt;
	T_b_v_gt.linear() = orientation.toRotationMatrix();
	T_b_v_gt.translation() = translation;
	Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

	//Eigen::Vector3d body_velocity_gt;
	//tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
	//body_velocity_gt = mocap_initial_frame.linear().transpose() *
	//  body_velocity_gt;

	// Ground truth tf.
	if (publish_tf) {
		geometry_msgs::msg::TransformStamped T_b_w_gt_tf;
		T_b_w_gt_tf = tf2::eigenToTransform(T_b_w_gt);
		T_b_w_gt_tf.header.frame_id = fixed_frame_id;
		T_b_w_gt_tf.header.stamp = msg->header.stamp;
		T_b_w_gt_tf.child_frame_id = child_frame_id+"_mocap";
		tf_pub.sendTransform(T_b_w_gt_tf);
	}

	// Ground truth odometry.
	nav_msgs::msg::Odometry mocap_odom_msg;
	mocap_odom_msg.header.stamp = msg->header.stamp;
	mocap_odom_msg.header.frame_id = fixed_frame_id;
	mocap_odom_msg.child_frame_id = child_frame_id+"_mocap";

	mocap_odom_msg.pose.pose = tf2::toMsg(T_b_w_gt);
	//tf::vectorEigenToMsg(body_velocity_gt,
	//    mocap_odom_msg.twist.twist.linear);

	mocap_odom_pub->publish(mocap_odom_msg);
	return;
}

bool MsckfVio::getMagData(MagMsg &mag_sample, const double &time_bound)
{
  Vector3d sum_mag = Vector3d::Zero();
  double sum_time = 0.0;
  int valid_count = 0;
  int used_count = 0;
  double &state_timestamp = state_server.imu_state.time;
  for(auto &mag : mag_buffer)
  {
    if(mag.timestamp< state_timestamp)
    {
      used_count++;
      continue;
    }
    if(mag.timestamp>time_bound)
      break;
    sum_mag += mag.magnetometer_ga;
    sum_time += mag.timestamp;
    valid_count ++;
    used_count++;
  }

  mag_buffer.erase(mag_buffer.begin(), mag_buffer.begin()+used_count);

  if(valid_count == 0)
    return false;

  mag_sample.timestamp = sum_time/valid_count;
  mag_sample.magnetometer_ga  = sum_mag/valid_count;
  RCLCPP_INFO(this->get_logger(), "get mag data count: %d", valid_count);
  return true;
}


void MsckfVio::magFusionControl(MagMsg &mag_sample, bool is_data_ready)
{
  if(!is_data_ready)
  {
    mag_fusion_mode = 0;
    RCLCPP_WARN(this->get_logger(), "do not fuse mag!");
    return;
  }
  Vector3d &data = mag_sample.magnetometer_ga;
  double mag_strength = sqrt((data[0]*data[0])+data[1]*data[1]+data[2]*data[2]);
  if(fabs(mag_strength-ref_mag_strength)>mag_strength_gate)
  {
    mag_fusion_mode = 0;
    RCLCPP_WARN(this->get_logger(), "do not fuse mag!, mag strength: %f", mag_strength);
    return;
  }
  Eigen::Matrix3d R_i_w = quaternionToRotation(state_server.imu_state.orientation).transpose();

  auto lpf_acc = R_i_w *acc_filter.getState();
  auto lpf_gyro = R_i_w*gyro_filter.getState();
  double norm_acc_NE = sqrt(lpf_acc[0]*lpf_acc[0] + lpf_acc[1]*lpf_acc[1]);
  yaw_angle_observable = yaw_angle_observable
				? norm_acc_NE > mag_acc_gate //0.5
				: norm_acc_NE > 2.0f * mag_acc_gate;

  if (!mag_bias_observable && (fabs(lpf_gyro(2)) > mag_yaw_rate_gate)) {
		// initial yaw motion is detected
		mag_bias_observable = true;

	} else if (mag_bias_observable) {
		// require sustained yaw motion of 50% the initial yaw rate threshold 0.25
		// const float yaw_dt = 1e-6f * (float)(mag_sample.timestamp - last_mag_time);
		// const float min_yaw_change_req =  0.5f * mag_yaw_rate_gate * yaw_dt;
		// mag_bias_observable = fabsf() > min_yaw_change_req;
    mag_bias_observable = fabs(lpf_gyro[2])>0.5*mag_yaw_rate_gate;
	}

	// _yaw_delta_ef = 0.0f;
	// _time_yaw_started = _imu_sample_delayed.time_us;

  if(fuse_3d_mag && gps_correct_mag && (mag_bias_observable || yaw_angle_observable))
    mag_fusion_mode = 2;
  else
    mag_fusion_mode = 1;
}

void MsckfVio::fuseMag2D(MagMsg &mag_sample)
{
  double n_yaw = pow(mag_heading_noise, 2);
  Vector3d mag_earth;
  Vector3d mag_data = mag_sample.magnetometer_ga;
  Eigen::Matrix3d R_i_w = quaternionToRotation(state_server.imu_state.orientation).transpose();

  double predict_yaw;   //表示I系旋转到W系的yaw？
  double measure_yaw;
  double roll;
  double pitch;
  Eigen::Matrix3d R_i_h;
  if(fabs(R_i_w(2,0)) < fabs(R_i_w(2,1)))
  {
    Eigen::Vector3d euler321 = Euler321(R_i_w);
    predict_yaw = euler321(0);
    // euler321(2) = 0;
    roll = euler321(2);
    pitch = euler321(1);
    // roll = Eigen::AngleAxisd(euler321(0), Vector3d(1,0,0));
    // pitch = Eigen::AngleAxisd(euler321(1), Vector3d(0,1,0));
    R_i_h = Eigen::AngleAxisd(pitch, Vector3d(0,1,0)).toRotationMatrix()*Eigen::AngleAxisd(roll, Vector3d(1,0,0)).toRotationMatrix();
  }
  else
  {
    predict_yaw = wrap_pi(-atan2(R_i_w(0,1), R_i_w(1,1))); //body -> world 角度

    roll = asin(R_i_w(2,1));
    pitch = atan2(-R_i_w(2,0), R_i_w(2,2));
    R_i_h = Eigen::AngleAxisd(roll, Vector3d(1,0,0)).toRotationMatrix()*Eigen::AngleAxisd(pitch, Vector3d(0,1,0)).toRotationMatrix();
  }

  // double ear_gav = R_i_h.row(2)*curr_acc;
  // double grav_x = R_i_h.row(0)*curr_acc;
  // double grav_y = R_i_h.row(1)*curr_acc;
  mag_earth = R_i_h*R_mag_imu*(mag_data - state_server.imu_state.mag_bias);
  measure_yaw = atan2(-mag_earth(1), mag_earth(0)) + getDeclination();
  measure_yaw = wrap_pi(measure_yaw);
  // double measure_yaw_o = wrap_pi(atan2(mag_data[1], mag_data[0]));
  // double acc_norm = curr_acc.norm();
  double kdeg = 180/M_PI;

  LOG(INFO)<<"measure_yaw: "<<state_server.imu_state.time<<" "<<measure_yaw*kdeg<<" "<<predict_yaw*kdeg;
  // LOG(INFO)<<"measure_gav: "<<state_server.imu_state.time<<" "<<grav_x<<" "<<grav_y<<" "<<ear_gav;

  if(fuse_mag)
  {
    double H_z = 0;
    double H_y = 0;
    double R10_2 = pow(R_i_w(1,0),2);
    double R00_2 = pow(R_i_w(0,0), 2);
    if(R00_2>R10_2 && R00_2>1e-6f)
    {
      double R00_inv = 1/R00_2;
      double x_div = 1/(1+R00_inv*R10_2);
      H_z = x_div*R00_inv*(R_i_w(1,1)*R_i_w(0,0) - R_i_w(0,1)*R_i_w(1,0));
      H_y = x_div*R00_inv*(R_i_w(0,2)*R_i_w(1,0) - R_i_w(1,2)*R_i_w(0,0));
    }
    else if(R10_2 > 1E-6f)
    {
      double R10_inv = 1/R10_2;
      double x_div = 1/(1+R10_inv*R00_2);
      H_z = x_div*R10_inv*(R_i_w(1,1)*R_i_w(0,0) - R_i_w(0,1)*R_i_w(1,0));
      H_y = x_div*R10_inv*(R_i_w(0,2)*R_i_w(1,0) - R_i_w(1,2)*R_i_w(0,0));
    }
    // 计算K H = [(0,0,1), 0 ..... 0]
    // double Q = state_server.state_cov(2,2) + n_yaw;
    MatrixXd & state_cov = state_server.state_cov;
    double Q = H_y*H_y*state_cov(1,1)+H_y*H_z*state_cov(1,2)+H_z*H_y*state_cov(2,1)+H_z*H_z*state_cov(2,2)+n_yaw;
    
    // LOG(INFO)<<"Q: "<<state_server.imu_state.time<<" "<<Q;
    // Eigen::MatrixXd K = state_server.state_cov.col(2);
    Eigen::MatrixXd K = H_y*state_cov.col(1) + H_z*state_cov.col(2);
    if(Q< 1e-6)
    {
      RCLCPP_WARN(this->get_logger(), "yaw convarience too small!");
      return;
    }
    K = K*1.0/(Q);
    // cout<<"K: "<<endl;
    // cout<<K<<endl;
    //update..
    VectorXd delta_x = K*wrap_pi(measure_yaw - predict_yaw);

    const Vector4d dq_imu =
      smallAngleQuaternion(delta_x.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
        dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x.segment<3>(3);
    state_server.imu_state.velocity += delta_x.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x.segment<3>(9);
    state_server.imu_state.position += delta_x.segment<3>(12);

    const Vector4d dq_extrinsic =
      smallAngleQuaternion(delta_x.segment<3>(15));
    state_server.imu_state.R_imu_cam0 = quaternionToRotation(
        dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x.segment<3>(18);

    state_server.imu_state.mag_ned += delta_x.segment<3>(21);
    state_server.imu_state.mag_bias += delta_x.segment<3>(24);
    state_server.imu_state.gps_bias += delta_x.segment<2>(27);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size();
        ++i, ++cam_state_iter) {
      const VectorXd& delta_x_cam = delta_x.segment<6>(29+i*6);
      const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
      cam_state_iter->second.orientation = quaternionMultiplication(
          dq_cam, cam_state_iter->second.orientation);
      cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    MatrixXd H_heading = MatrixXd::Zero(1, 29+state_server.cam_states.size()*6);
    H_heading(0, 2) = H_z;
    H_heading(0,1) = H_y;
    // H_heading(0, 2) = 1;
      // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_heading.cols()) - K*H_heading;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH*state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;
  }
  // double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  // double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  // double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  // LOG(INFO)<<"pos_cov: "<<state_server.imu_state.time<<" "<<position_x_std<<" "<<position_y_std<<" "<<position_z_std;


}

void MsckfVio::fuseMag3D(MagMsg &mag_sample)
{
  //fuseMag
  MatrixXd H_mag = MatrixXd::Zero(3, 29+state_server.cam_states.size()*6);
  Matrix3d R_w_i = quaternionToRotation(state_server.imu_state.orientation);
  Matrix3d R_i_m = R_mag_imu.transpose();
  H_mag.block<3,3>(0,0) = R_i_m*skewSymmetric(R_w_i*state_server.imu_state.mag_ned);
  H_mag.block<3,3>(0,21) = R_i_m*R_w_i;
  H_mag.block<3,3>(0,24) = Eigen::Matrix3d::Identity();

  Vector3d predict_mag = R_i_m*R_w_i*state_server.imu_state.mag_ned + state_server.imu_state.mag_bias;
  Vector3d measure_mag = mag_sample.magnetometer_ga; 
  Vector3d r = measure_mag - predict_mag;

  if(fuse_mag)
  {
    //对齐测量值的z轴后不进行融合z(2) = 0;
    // Compute the Kalman gain.
    const MatrixXd& P = state_server.state_cov;
    Matrix3d S = H_mag*P*H_mag.transpose() +
        pow(mag_noise,2)*Matrix3d::Identity();
    //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    MatrixXd K_transpose = S.ldlt().solve(H_mag*P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r;

    // Update the IMU state.
    const VectorXd& delta_x_imu = delta_x.head<29>();
    // LOG(INFO)<<"delta_P: "<<state_server.imu_state.time<<" "
    //          <<delta_x_imu[12]<<" "
    //          <<delta_x_imu[13]<<" "
    //          <<delta_x_imu[14];
    // LOG(INFO)<<"delta_V: "<<state_server.imu_state.time<<" "
    //          <<delta_x_imu[6]<<" "
    //          <<delta_x_imu[7]<<" "
    //          <<delta_x_imu[8];

    if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
        //delta_x_imu.segment<3>(3).norm() > 0.15 ||
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        //delta_x_imu.segment<3>(9).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0) {
      printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
      printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
      RCLCPP_WARN(get_logger(), "Update change is too large.");
      //return;
    }

    const Vector4d dq_imu =
      smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
        dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    const Vector4d dq_extrinsic =
      smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_imu_cam0 = quaternionToRotation(
        dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

    state_server.imu_state.mag_ned += delta_x_imu.segment<3>(21);
    state_server.imu_state.mag_bias += delta_x_imu.segment<3>(24);
    state_server.imu_state.gps_bias += delta_x_imu.segment<2>(27);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size();
        ++i, ++cam_state_iter) {
      const VectorXd& delta_x_cam = delta_x.segment<6>(29+i*6);
      const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
      cam_state_iter->second.orientation = quaternionMultiplication(
          dq_cam, cam_state_iter->second.orientation);
      cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_mag.cols()) - K*H_mag;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH*state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    fuseDeclination(0.5);    //初始 0.02为何？
  }

  //fuse_declination:

  return;
}

// void Ekf::limitDeclination()
// {
// 	// get a reference value for the earth field declinaton and minimum plausible horizontal field strength
// 	// set to 50% of the horizontal strength from geo tables if location is known
// 	float decl_reference;
// 	float h_field_min = 0.001f;
// 	if (_params.mag_declination_source & MASK_USE_GEO_DECL) {
// 		// use parameter value until GPS is available, then use value returned by geo library
// 		if (_NED_origin_initialised) {
// 			decl_reference = _mag_declination_gps;
// 			h_field_min = fmaxf(h_field_min , 0.5f * _mag_strength_gps * cosf(_mag_inclination_gps));
// 		} else {
// 			decl_reference = math::radians(_params.mag_declination_deg);
// 		}
// 	} else {
// 		// always use the parameter value
// 		decl_reference = math::radians(_params.mag_declination_deg);
// 	}

// 	// do not allow the horizontal field length to collapse - this will make the declination fusion badly conditioned
// 	// and can result in a reversal of the NE field states which the filter cannot recover from
// 	// apply a circular limit
// 	float h_field = sqrtf(_state.mag_I(0)*_state.mag_I(0) + _state.mag_I(1)*_state.mag_I(1));
// 	if (h_field < h_field_min) {
// 		if (h_field > 0.001f * h_field_min) {
// 			float h_scaler = h_field_min / h_field;
// 			_state.mag_I(0) *= h_scaler;
// 			_state.mag_I(1) *= h_scaler;
// 		} else {
// 			// too small to scale radially so set to expected value
// 			float mag_declination = getMagDeclination();
// 			_state.mag_I(0) = 2.0f * h_field_min * cosf(mag_declination);
// 			_state.mag_I(1) = 2.0f * h_field_min * sinf(mag_declination);
// 		}
// 		h_field = h_field_min;
// 	}

// 	// do not allow the declination estimate to vary too much relative to the reference value
// 	constexpr float decl_tolerance = 0.5f;
// 	const float decl_max = decl_reference + decl_tolerance;
// 	const float decl_min = decl_reference - decl_tolerance;
// 	const float decl_estimate = atan2f(_state.mag_I(1) , _state.mag_I(0));
// 	if (decl_estimate > decl_max)  {
// 		_state.mag_I(0) = h_field * cosf(decl_max);
// 		_state.mag_I(1) = h_field * sinf(decl_max);
// 	} else if (decl_estimate < decl_min)  {
// 		_state.mag_I(0) = h_field * cosf(decl_min);
// 		_state.mag_I(1) = h_field * sinf(decl_min);
// 	}
// }

//z轴磁力计数据和参考值对齐，这样就不进行融合了
//前提有GPS数据，获取到了参考表
// float Ekf::calculate_synthetic_mag_z_measurement(const Vector3f& mag_meas, const Vector3f& mag_earth_predicted)
// {
// 	// theoretical magnitude of the magnetometer Z component value given X and Y sensor measurement and our knowledge
// 	// of the earth magnetic field vector at the current location
// 	const float mag_z_abs = sqrtf(math::max(sq(mag_earth_predicted.length()) - sq(mag_meas(0)) - sq(mag_meas(1)), 0.0f));

// 	// calculate sign of synthetic magnetomter Z component based on the sign of the predicted magnetomer Z component
// 	const float mag_z_body_pred = mag_earth_predicted.dot(_R_to_earth.slice<3,1>(0,2));

// 	return mag_z_body_pred < 0 ? -mag_z_abs : mag_z_abs;
// }

// 磁偏角：ned系到地磁北旋转角度，观测偏航角：NED到IMU系偏航角度
void MsckfVio::fuseDeclination(double noise)
{
  auto &B_ned = state_server.imu_state.mag_ned;
  double predict_d = atan2(B_ned(1), B_ned(0));
  
  if(fuse_d)
  {
    double B_x2 = pow(B_ned(0), 2);
    double B_y2 = pow(B_ned(1), 2);
    double H_d_x = -B_ned(1)/(B_x2+B_y2);
    double H_d_y = B_ned(0)/(B_x2 + B_y2);

    MatrixXd H_d = MatrixXd::Zero(1, 29);
    H_d(0,21) = H_d_x;
    H_d(0,22) = H_d_y;

    double n_d = pow(noise, 2);
    auto &state_cov = state_server.state_cov;
    double Q = H_d_x*H_d_x*state_cov(21,21)+H_d_x*H_d_y*state_cov(21,22)+H_d_y*H_d_x*state_cov(22,21)+H_d_y*H_d_y*state_cov(22,22)+n_d;

    if(Q < 1e-6)
      return;
    MatrixXd K = (state_cov.col(21)*H_d(0,21) + state_cov.col(22)*H_d(0,22))/Q;
    //NED和参考偏角对齐后不进行融合？
    VectorXd delta_x = K*wrap_pi(yaw_declination - predict_d);

    const Vector4d dq_imu =
      smallAngleQuaternion(delta_x.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
        dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x.segment<3>(3);
    state_server.imu_state.velocity += delta_x.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x.segment<3>(9);
    state_server.imu_state.position += delta_x.segment<3>(12);

    const Vector4d dq_extrinsic =
      smallAngleQuaternion(delta_x.segment<3>(15));
    state_server.imu_state.R_imu_cam0 = quaternionToRotation(
        dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x.segment<3>(18);

    state_server.imu_state.mag_ned += delta_x.segment<3>(21);
    state_server.imu_state.mag_bias += delta_x.segment<3>(24);
    state_server.imu_state.gps_bias += delta_x.segment<2>(27);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size();
        ++i, ++cam_state_iter) {
      const VectorXd& delta_x_cam = delta_x.segment<6>(29+i*6);
      const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
      cam_state_iter->second.orientation = quaternionMultiplication(
          dq_cam, cam_state_iter->second.orientation);
      cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // H_heading(0, 2) = 1;
      // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_d.cols()) - K*H_d;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH*state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;
  }

  //limitDeclination()限制融合后NED下磁力计值

}

void MsckfVio::fuseMag(MagMsg &mag_sample)
{
  if(mag_fusion_mode == 1)
  {
    fuseMag2D(mag_sample);
    RCLCPP_INFO(this->get_logger(), "fuse 2d mag");
  }
  else
  {
    //第一次 3d融合，先融declination
    fuseMag3D(mag_sample);
    RCLCPP_INFO(this->get_logger(), "fuse 3d mag");
  }
}

void MsckfVio::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  // RCLCPP_INFO(get_logger(), "enter batchImuProcessing");
  int used_imu_msg_cntr = 0;

#ifdef FUSE_MAG
  MagMsg mag_sample;
  bool mag_data_ready = getMagData(mag_sample, time_bound);
  magFusionControl(mag_sample, mag_data_ready);
#endif
  bool gps_data_ready = false;
  px4_msgs::msg::VehicleGpsPosition gps_sample;
  double gps_time_d;
  int count_gps = 0;
  for(auto &per_gps : gps_buffer)
  {
    count_gps++;
    gps_time_d = stamp2sec(per_gps.timestamp);
    if(gps_time_d>state_server.imu_state.time && gps_time_d<time_bound)
    {
      gps_sample = per_gps;
      gps_data_ready = true;
      break;
    }
  }
  gps_buffer.erase(gps_buffer.begin(), gps_buffer.begin()+count_gps);

  for (const auto& imu_msg : imu_msg_buffer) {
    double imu_time = stamp2sec(imu_msg.header.stamp);
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    utils::fromMsg(imu_msg.angular_velocity, m_gyro);
    utils::fromMsg(imu_msg.linear_acceleration, m_acc);
    // m_gyro = gyro_com*m_gyro + gyro_bias;
    // m_acc = acc_com*m_acc + acc_bias;

#ifdef FUSE_MAG
    // RCLCPP_INFO(this->get_logger(), "fusion mode: %d, magtime %f, imu_time %f", mag_fusion_mode, mag_sample.timestamp, imu_time);
    if(mag_fusion_mode>0 && mag_sample.timestamp<imu_time)
    {
      curr_acc = m_acc;
      fuseMag(mag_sample);
      mag_fusion_mode = 0;
    }
#endif

    if(gps_data_ready && gps_time_d<imu_time)
    {
      gpsUpdate(gps_sample);
      gps_data_ready = false;
    }

    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}

void MsckfVio::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  double dtime = time - imu_state.time;

  // Compute discrete transition and noise covariance matrix
  Matrix<double, 29, 29> F = Matrix<double, 29, 29>::Zero();
  Matrix<double, 29, 12> G = Matrix<double, 29, 12>::Zero();

  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(
      imu_state.orientation).transpose()*skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F.block<3, 3>(12, 6) = Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(
      imu_state.orientation).transpose();
  G.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order, 连续微分方程->离散
  // which can be considered to be accurate enough assuming
  // dtime is within 0.01s.
  Matrix<double, 29, 29> Fdt = F * dtime;
  Matrix<double, 29, 29> Fdt_square = Fdt * Fdt;
  Matrix<double, 29, 29> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 29, 29> Phi = Matrix<double, 29, 29>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

  // Propogate the state using 4th order Runge-Kutta
  predictNewState(dtime, gyro, acc);

  // Modify the transition matrix， 可观性相关？
  Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
  Phi.block<3, 3>(0, 0) =
    quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

  Vector3d u = R_kk_1 * IMUState::gravity;
  RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

  Matrix3d A1 = Phi.block<3, 3>(6, 0);
  Vector3d w1 = skewSymmetric(
      imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
  Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

  Matrix3d A2 = Phi.block<3, 3>(12, 0);
  Vector3d w2 = skewSymmetric(
      dtime*imu_state.velocity_null+imu_state.position_null-
      imu_state.position) * IMUState::gravity;
  Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

  // Propogate the state covariance matrix.
  Matrix<double, 29, 29> Q = Phi*G*state_server.continuous_noise_cov*
    G.transpose()*Phi.transpose()*dtime;
  state_server.state_cov.block<29, 29>(0, 0) =
    Phi*state_server.state_cov.block<29, 29>(0, 0)*Phi.transpose() + Q;

  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(
        0, 29, 29, state_server.state_cov.cols()-29) =
      Phi * state_server.state_cov.block(
        0, 29, 29, state_server.state_cov.cols()-29);
    state_server.state_cov.block(
        29, 0, state_server.state_cov.rows()-29, 29) =
      state_server.state_cov.block(
        29, 0, state_server.state_cov.rows()-29, 29) * Phi.transpose();
  }

  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state correspondes to null space.
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}

void MsckfVio::predictNewState(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +
    IMUState::gravity;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  return;
}

void MsckfVio::stateAugmentation(const double& time) {
  // RCLCPP_INFO(get_logger(), "enter stateAugmentation");

  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;  //imu 下坐标 -> cam0 下坐标
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;  //cam0 在imu下坐标

  // Add a new camera state to the state server.
  Matrix3d R_w_i = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i;
  Vector3d t_c_w = state_server.imu_state.position +
    R_w_i.transpose()*t_c_i;

  //add new camera state every time comes a new img msg
  state_server.cam_states[state_server.imu_state.id] =  //cam_state: map< >
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  cam_state.position = t_c_w;

  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
  Matrix<double, 6, 29> J = Matrix<double, 6, 29>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  J.block<3, 3>(3, 12) = Matrix3d::Identity();
  J.block<3, 3>(3, 18) = Matrix3d::Identity();

  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 29, 29>& P11 =
    state_server.state_cov.block<29, 29>(0, 0);
  const MatrixXd& P12 =
    state_server.state_cov.block(0, 29, 29, old_cols-29);

  // Fill in the augmented state covariance.
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

void MsckfVio::addFeatureObservations(
    const custom_msgs::msg::CameraMeasurement::SharedPtr& msg) {
  // RCLCPP_INFO(get_logger(), "enter addFeatureObservations");
  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : msg->features) {
    if (map_server.find(feature.id) == map_server.end()) {
      // This is a new feature.
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
    } else {
      // This is an old feature.
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}

void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {
  // RCLCPP_INFO(get_logger(), "enter measurementJacobian");
  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose.
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // Cam1 pose.
  Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
  Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
  Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector4d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

  // Compute the Jacobians.
  Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

  Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
  dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
  Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
  H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

  // Modifty the measurement Jacobian to ensure
  // observability constrain.
  Matrix<double, 4, 6> A = H_x;
  Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  u.block<3, 1>(0, 0) = quaternionToRotation(
      cam_state.orientation_null) * IMUState::gravity;
  u.block<3, 1>(3, 0) = skewSymmetric(
      p_w-cam_state.position_null) * IMUState::gravity;
  H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  H_f = -H_x.block<4, 3>(0, 3);

  // Compute the residual.
  r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
      p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

  return;
}

void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {
  // RCLCPP_INFO(get_logger(), "enter featureJacobian");
  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 4 * valid_cam_state_ids.size();

  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      29+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
    Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
    Vector4d r_i = Vector4d::Zero();
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<4, 6>(stack_cntr, 29+6*cam_state_cntr) = H_xi;
    H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<4>(stack_cntr) = r_i;
    stack_cntr += 4;
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {
  // RCLCPP_INFO(get_logger(), "enter measurementUpdate");
  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(29+state_server.cam_states.size()*6);
    r_thin = r_temp.head(29+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<29>();
  LOG(INFO)<<"delta_P: "<<state_server.imu_state.time<<" "
           <<delta_x_imu[12]<<" "
           <<delta_x_imu[13]<<" "
           <<delta_x_imu[14];
  LOG(INFO)<<"delta_V: "<<state_server.imu_state.time<<" "
           <<delta_x_imu[6]<<" "
           <<delta_x_imu[7]<<" "
           <<delta_x_imu[8];

  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
    RCLCPP_WARN(get_logger(), "Update change is too large.");
    //return;
  }

  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  state_server.imu_state.position += delta_x_imu.segment<3>(12);

  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  state_server.imu_state.mag_ned += delta_x_imu.segment<3>(21);
  state_server.imu_state.mag_bias += delta_x_imu.segment<3>(24);
  state_server.imu_state.gps_bias += delta_x_imu.segment<2>(27);

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(29+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

void MsckfVio::comMeasurementUpdate(
    const MatrixXd& H, const VectorXd& r, const MatrixXd &noise) {
  // RCLCPP_INFO(get_logger(), "enter measurementUpdate");
  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  H_thin = H;
  r_thin = r;

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() + noise;
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();
  // cout<<"GPS K:"<<endl;
  // cout<<K<<endl;

  MatrixXd KHP = K*H_thin*P;

  bool helthy_P = true;

  for(int i=0; i<29+6*state_server.cam_states.size(); i++)
  {
    if(P(i,i) < KHP(i,i))
    {
      helthy_P = false;
    }
  }

  if(helthy_P)
  {
    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    const VectorXd& delta_x_imu = delta_x.head<29>();
    // LOG(INFO)<<"delta_P: "<<state_server.imu_state.time<<" "
    //          <<delta_x_imu[12]<<" "
    //          <<delta_x_imu[13]<<" "
    //          <<delta_x_imu[14];
    // LOG(INFO)<<"delta_V: "<<state_server.imu_state.time<<" "
    //          <<delta_x_imu[6]<<" "
    //          <<delta_x_imu[7]<<" "
    //          <<delta_x_imu[8];

    if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
        //delta_x_imu.segment<3>(3).norm() > 0.15 ||
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        //delta_x_imu.segment<3>(9).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0) {
      printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
      printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
      RCLCPP_WARN(get_logger(), "Update change is too large.");
      //return;
    }

    const Vector4d dq_imu =
      smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
        dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    const Vector4d dq_extrinsic =
      smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_imu_cam0 = quaternionToRotation(
        dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

    state_server.imu_state.mag_ned += delta_x_imu.segment<3>(21);
    state_server.imu_state.mag_bias += delta_x_imu.segment<3>(24);
    state_server.imu_state.gps_bias += delta_x_imu.segment<2>(27);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size();
        ++i, ++cam_state_iter) {
      const VectorXd& delta_x_cam = delta_x.segment<6>(29+i*6);
      const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
      cam_state_iter->second.orientation = quaternionMultiplication(
          dq_cam, cam_state_iter->second.orientation);
      cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov -= KHP;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;
  }

  return;
}

bool MsckfVio::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  //cout << dof << " " << gamma << " " <<
  //  chi_squared_test_table[dof] << " ";

  if (gamma < chi_squared_test_table[dof]) {
    //cout << "passed" << endl;
    return true;
  } else {
    //cout << "failed" << endl;
    return false;
  }
}

void MsckfVio::removeLostFeatures() {
  // RCLCPP_INFO(get_logger(), "enter removeLostFeatures");

  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> processed_feature_ids(0);

  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // Pass the features that are still being tracked.
    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*feature.observations.size() - 3;
    processed_feature_ids.push_back(feature.id);
  }

  //cout << "invalid/processed feature #: " <<
  //  invalid_feature_ids.size() << "/" <<
  //  processed_feature_ids.size() << endl;
  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  if (processed_feature_ids.size() == 0) return;

  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      29+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  LOG(INFO)<<"process_update: "<<state_server.imu_state.time<<" "
           <<processed_feature_ids.size();
  // Process the features which lose track.
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 1500) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}

void MsckfVio::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids) {
  // RCLCPP_INFO(get_logger(), "enter findRedundantCamStates");
  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  auto first_cam_state_iter = state_server.cam_states.begin();

  // Pose of the key camera state.
  const Vector3d key_position =
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}

void MsckfVio::pruneCamStateBuffer() {
  // RCLCPP_INFO(get_logger(), "enter pruneCamStateBuffer");
  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  findRedundantCamStates(rm_cam_state_ids);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }

    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      29+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);

  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 29 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  }

  return;
}

void MsckfVio::onlineReset(int type_r) {

	// Never perform online reset if position std threshold
	// is non-positive.
  if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

	// Check the uncertainty of positions to determine if
	// the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  if (position_x_std < position_std_threshold &&
    position_y_std < position_std_threshold &&
    position_z_std < position_std_threshold)
    {
      LOG(INFO)<<"reset: "<<state_server.imu_state.time<<" "<<0;
      return;
    }

  LOG(INFO)<<"reset: "<<state_server.imu_state.time<<" "<<type_r;
  RCLCPP_WARN(get_logger(), "Start %lld online reset procedure...",
    ++online_reset_counter);
  RCLCPP_INFO(get_logger(), "Stardard deviation in xyz: %f, %f, %f",
    position_x_std, position_y_std, position_z_std);

	// Remove all existing camera states.
  state_server.cam_states.clear();

	// Clear all exsiting features in the map.
	map_server.clear();

	// Reset the state covariance.
	double gyro_bias_cov = 1e-4;
	double acc_bias_cov = 1e-2;
	double velocity_cov = 0.25;
  rclcpp::Parameter velocity_cov_parm("initial_covariance.velocity", velocity_cov);
  this->set_parameter(velocity_cov_parm);
	// this->declare_parameter<double>("initial_covariance/velocity",
	// 	velocity_cov);
  rclcpp::Parameter gyro_bias_cov_parm("initial_covariance.gyro_bias", gyro_bias_cov);
  this->set_parameter(gyro_bias_cov_parm);

  rclcpp::Parameter acc_bias_cov_parm("initial_covariance.acc_bias", acc_bias_cov);
  this->set_parameter(acc_bias_cov_parm);
	// this->declare_parameter<double>(,
	// 	gyro_bias_cov);
	// this->declare_parameter<double>("initial_covariance/acc_bias",
	// 	acc_bias_cov);

	double extrinsic_rotation_cov = 3.0462e-4;
	double extrinsic_translation_cov = 1e-4;
  rclcpp::Parameter extrinsic_rotation_cov_parm("initial_covariance.extrinsic_rotation_cov", extrinsic_rotation_cov);
	this->set_parameter(extrinsic_rotation_cov_parm);

  rclcpp::Parameter extrinsic_translation_cov_parm("initial_covariance.extrinsic_translation_cov", extrinsic_translation_cov);
  this->set_parameter(extrinsic_translation_cov_parm);
  // this->declare_parameter<double>(,
	// 	extrinsic_rotation_cov);
	// this->declare_parameter<double>("initial_covariance/extrinsic_translation_cov",
	// 	extrinsic_translation_cov);

	state_server.state_cov = MatrixXd::Zero(29, 29);
	for (int i = 3; i < 6; ++i)
		state_server.state_cov(i, i) = gyro_bias_cov;
	for (int i = 6; i < 9; ++i)
		state_server.state_cov(i, i) = velocity_cov;
	for (int i = 9; i < 12; ++i)
		state_server.state_cov(i, i) = acc_bias_cov;
	for (int i = 15; i < 18; ++i)
		state_server.state_cov(i, i) = extrinsic_rotation_cov;
	for (int i = 18; i < 21; ++i)
		state_server.state_cov(i, i) = extrinsic_translation_cov;
  state_server.state_cov(2,2) = pow(mag_heading_noise, 2);

	RCLCPP_WARN(get_logger(), "%lld online reset complete...", online_reset_counter);
	return;
}

void MsckfVio::publish(const rclcpp::Time& time) {
  // RCLCPP_INFO(get_logger(), "enter publish");
	// Convert the IMU frame to the body frame.
	const IMUState& imu_state = state_server.imu_state;
	Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  Matrix3d ts = Matrix3d::Identity();
  ts(0, 0) = -1;
  ts(2,2) = -1;
	T_i_w.linear() = ts*quaternionToRotation(
		imu_state.orientation).transpose();
	T_i_w.translation() = Vector3d(-imu_state.position[0], imu_state.position[1], -imu_state.position[2]);


	Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
		IMUState::T_imu_body.inverse();
	Eigen::Vector3d body_velocity =
		IMUState::T_imu_body.linear() * imu_state.velocity;

	// Publish tf
	if (publish_tf) {
		geometry_msgs::msg::TransformStamped T_b_w_tf;
		T_b_w_tf = tf2::eigenToTransform(T_b_w);
    T_b_w_tf.header.frame_id = fixed_frame_id;
		T_b_w_tf.header.stamp = time;
		T_b_w_tf.child_frame_id = child_frame_id;
		tf_pub.sendTransform(T_b_w_tf);
	}
  // RCLCPP_INFO(get_logger(), "finish transform publish");
	// Publish the odometry
	nav_msgs::msg::Odometry odom_msg;
	odom_msg.header.stamp = time;
	odom_msg.header.frame_id = fixed_frame_id;
	odom_msg.child_frame_id = child_frame_id;

	odom_msg.pose.pose = tf2::toMsg(T_b_w);
	odom_msg.twist.twist.linear = tf2::toMsg2(body_velocity);

	// Convert the covariance.
	Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
	Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
	Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
	Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
	Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
	P_imu_pose << P_pp, P_po, P_op, P_oo;

	Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
	H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
	H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
	Matrix<double, 6, 6> P_body_pose = H_pose *
		P_imu_pose * H_pose.transpose();

	for (int i = 0; i < 6; ++i)
		for (int j = 0; j < 6; ++j)
		odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);

	// Construct the covariance for the velocity.
	Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
	Matrix3d H_vel = IMUState::T_imu_body.linear();
	Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
		odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

	odom_pub->publish(odom_msg);
  // RCLCPP_INFO(get_logger(), "finish odometry publish");
	// Publish the 3D positions of the features that
	// has been initialized.
	pcl::PointCloud<pcl::PointXYZ> pcl_feature;
	sensor_msgs::msg::PointCloud2 feature_msg;
	 
	pcl_feature.header.frame_id = fixed_frame_id;
	pcl_feature.height = 1;
	for (const auto& item : map_server) {
		const auto& feature = item.second;
		if (feature.is_initialized) {
			Vector3d feature_position = IMUState::T_imu_body.linear() * feature.position;
			pcl_feature.points.push_back(pcl::PointXYZ(
				-feature_position(0), feature_position(1), -feature_position(2)));
		}
	}
	pcl_feature.width = pcl_feature.points.size();
  // RCLCPP_INFO(get_logger(), "before feature msg transfer");
	pcl::toROSMsg(pcl_feature, feature_msg);
  // RCLCPP_INFO(get_logger(), "before feature publish");
	feature_pub->publish(feature_msg);
  // RCLCPP_INFO(get_logger(), "finish publish");
	return;
}

} // namespace msckf_vio

// #include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
// RCLCPP_COMPONENTS_REGISTER_NODE(ros2_msckf::MsckfVio)