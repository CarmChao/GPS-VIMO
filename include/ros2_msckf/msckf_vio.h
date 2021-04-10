#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"
#include "ros2_msckf/math_utils.hpp"
#include "ros2_msckf/utils.h"
#include "custom_msgs/msg/camera_measurement.hpp"
#include "visibility_control.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "px4_msgs/msg/vehicle_magnetometer.hpp"
#include "px4_msgs/msg/vehicle_gps_position.hpp"
#include "px4_msgs/msg/vehicle_air_data.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "std_srvs/srv/trigger.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glog/logging.h"

#include <vector>
#include <set>
#include <string>
#include <map>
#include <memory>

// #define MAG_FUSE
// #define DECLINATION_FUSE
// #define GPS_FUSE

namespace ros2_msckf {
/*
 * @brief MsckfVio Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */
class MsckfVio : public rclcpp::Node
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    // COMPOSITION_PUBLIC
    // explicit MsckfVio(const rclcpp::NodeOptions& options);
    MsckfVio();
    // Disable copy and assign constructor
    MsckfVio(const MsckfVio&) = delete;
    MsckfVio operator=(const MsckfVio&) = delete;

    // Destructor
    ~MsckfVio() {}

    /*
    * @brief initialize Initialize the VIO.
    */
    bool initialize();

    /*
    * @brief reset Resets the VIO to initial status.
    */
    void reset();

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

    static double stamp2sec(const uint64_t &timestamp)
    {
        double sec = timestamp/1000000;
        return sec+static_cast<double>((timestamp-sec*1e6)/1e6);
    }

    static uint64_t stamp2sec(const double &time)
    {

    }
    static double wrap(double x, double low, double high)
    {
    // already in range
        if (low <= x && x < high) {
            return x;
        }

            const double range = high - low;
            const double inv_range = 1.0 / range; // should evaluate at compile time, multiplies below at runtime
            const double num_wraps = floor((x - low) * inv_range);
            return x - range * num_wraps;
    }

    static double wrap_pi(double x)
    {
        return wrap(x, -M_PI, M_PI);
    }

static Eigen::Vector3d Euler321(const Eigen::Matrix3d &R)
    {
        double roll = atan2(R(2, 1), R(2, 2));
        double pitch = asin(-R(2, 0));
        double yaw = atan2(R(1, 0), R(0, 0));

        if (fabs(pitch - M_PI/2.0) < 1.0e-3) {
            roll = 0;
            yaw = atan2(R(1, 2), R(0, 2));

        } else if (fabs(pitch + M_PI /2) < 1.0e-3) {
            roll = 0;
            yaw = atan2(-R(1, 2), -R(0, 2));
        }

        return Eigen::Vector3d(yaw, pitch, roll);
    }

    typedef std::shared_ptr<MsckfVio> Ptr;
    typedef std::shared_ptr<const MsckfVio> ConstPtr;

private:
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer {
        IMUState imu_state;
        CamStateServer cam_states;

        // State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };


    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    /*
     * @brief createRosIO
     *    Create ros publisher and subscirbers.
     */
    bool createRosIO();

    /*
     * @brief imuCallback
     *    Callback function for the imu message.
     * @param msg IMU msg.
     */
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
     */
    void featureCallback(const custom_msgs::msg::CameraMeasurement::SharedPtr msg);

    void magCallback(const px4_msgs::msg::VehicleMagnetometer::SharedPtr msg);

    void baroCallback(const px4_msgs::msg::VehicleAirData::SharedPtr msg);

    void gpsCallback(const px4_msgs::msg::VehicleGpsPosition::SharedPtr msg);

    bool getMagData(MagMsg &mag_sample, const double &time_bound);

    void magFusionControl(MagMsg &mag_sample, bool is_data_ready);
    
    void fuseMag(MagMsg &mag_sample);
    
    void fuseMag2D(MagMsg &mag_sample);
    
    void fuseMag3D(MagMsg &mag_sample);
    
    void fuseDeclination(double noise);
    /*
     * @brief publish Publish the results of VIO.
     * @param time The time stamp of output msgs.
     */
    void publish(const rclcpp::Time& time);

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
     */
    bool initializeGravityAndBias();

    /*
     * @biref resetCallback
     *    Callback function for the reset service.
     *    Note that this is NOT anytime-reset. This function should
     *    only be called before the sensor suite starts moving.
     *    e.g. while the robot is still on the ground.
     */
    void resetCallback(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const double& time_bound);
    void processModel(const double& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc);
    void predictNewState(const double& dt,
        const Eigen::Vector3d& gyro,
        const Eigen::Vector3d& acc);

    // Measurement update
    void stateAugmentation(const double& time);
    void addFeatureObservations(const custom_msgs::msg::CameraMeasurement::SharedPtr& msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType& cam_state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 4, 6>& H_x,
        Eigen::Matrix<double, 4, 3>& H_f,
        Eigen::Vector4d& r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    void measurementUpdate(const Eigen::MatrixXd& H,
        const Eigen::VectorXd& r);
    bool gatingTest(const Eigen::MatrixXd& H,
        const Eigen::VectorXd&r, const int& dof);
    void removeLostFeatures();
    void findRedundantCamStates(
        std::vector<StateIDType>& rm_cam_state_ids);
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();

    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    // Features used
    MapServer map_server;

    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<sensor_msgs::msg::Imu> imu_msg_buffer;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Ros node handle

    // Subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Subscription<custom_msgs::msg::CameraMeasurement>::SharedPtr feature_sub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr feature_pub;
    tf2_ros::TransformBroadcaster tf_pub;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv;
    rclcpp::Subscription<px4_msgs::msg::VehicleMagnetometer>::SharedPtr mag_sub;
    rclcpp::Subscription<px4_msgs::msg::VehicleAirData>::SharedPtr baro_sub;
    rclcpp::Subscription<px4_msgs::msg::VehicleGpsPosition>::SharedPtr gps_sub;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Whether to publish tf or not.
    bool publish_tf;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;

    // bool FUSE_MAG = false;

    LPF<Eigen::Vector3d> acc_filter;
    LPF<Eigen::Vector3d> gyro_filter;

    Eigen::Matrix3d R_mag_imu;
    uint8_t mag_fusion_mode;
    LPF<Eigen::Vector3d> mag_filter;
    std::vector<MagMsg> mag_buffer;

    Eigen::Vector3d curr_acc;
    
    double last_mag_time;
    double ref_mag_strength;
    double mag_strength_gate;
    bool mag_bias_observable;
    bool yaw_angle_observable;
    double mag_acc_gate;        //0.5
    double mag_yaw_rate_gate;   //0.25
    double mag_heading_noise;   //0.3
    double mag_noise;           //0.05
    double yaw_declination;
    //GPS
    bool has_gps;

    bool fuse_mag;

    // Debugging variables and functions
    void mocapOdomCallback(
        const nav_msgs::msg::Odometry::SharedPtr msg);

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr mocap_odom_sub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr mocap_odom_pub;
    geometry_msgs::msg::TransformStamped raw_mocap_odom_msg;
    Eigen::Isometry3d mocap_initial_frame;

    rclcpp::Clock clocker;
};

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;

} // namespace ros2_msckf

#endif