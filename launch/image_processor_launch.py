from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()
    indemind = 'indemind-camimu.yaml'
    indemind_topic = ['/imu', '/left_img', '/right_img']
    euroc = 'camchain-imucam-euroc.yaml'
    euroc_topic = ['/imu0', '/cam0/image_raw', 'cam1/image_raw']
    configs = []
    configs.append(indemind)
    configs.append(euroc)
    topics = []
    topics.append(indemind_topic)
    topics.append(euroc_topic)
    src_index = 0
    
    config = os.path.join(get_package_share_directory('ros2_msckf'), 'config', configs[src_index])
    image_node = Node(
        package='ros2_msckf',
        node_name='image_processor',
        node_executable='image_processor',
        parameters=[config,
            {'grid_row': 4},
            {'grid_col': 5},
            {'grid_min_feature_num': 3},
            {'grid_max_feature_num': 4},
            {'pyramid_levels': 3},
            {'patch_size': 15},
            {'fast_threshold': 10},
            {'max_iteration': 30},
            {'track_precision': 0.01},
            {'ransac_threshold': 3.0},
            {'stereo_threshold': 5.0}],
        remappings=[
            ('imu', topics[src_index][0]),
            ('cam0_image', topics[src_index][1]),
            ('cam1_image', topics[src_index][2])],
        output='screen'
    )
    vio_node = Node(
        package='ros2_msckf',
        node_name='msckf_vio',
        node_executable='msckf_vio',
        parameters=[config,
            {"publish_tf": "true"},
            {"frame_rate": 20.0},
            {"fixed_frame_id": "world"},
            {"child_frame_id": "odom"},
            {"max_cam_state_size": 20},
            {"position_std_threshold": 8.0},
            {"rotation_threshold": 0.2618},
            {"translation_threshold": 0.4},
            {"tracking_rate_threshold": 0.5},
            {"feature.config.translation_threshold": -1.0},
            {"noise.gyro": 0.005},
            {"noise.acc": 0.05},
            {"noise.gyro_bias": 0.001},
            {"noise.acc_bias": 0.01},
            {"noise.feature": 0.035},
            {"initial_covariance.velocity": 0.25},
            {"initial_covariance.gyro_bias":0.01},
            {"initial_covariance.acc_bias": 0.01},
            {"initial_covariance.extrinsic_rotation_cov": 3.0462e-4},
            {"initial_covariance.extrinsic_translation_cov": 2.5e-5},
            {"initial_state.velocity.x": 0.0},
            {"initial_state.velocity.y": 0.0},
            {"initial_state.velocity.z": 0.0},
            {"mag_extrinsic": -0.1067},
            {"fuse_mag": "true"},
            {"mag_heading_noise": 0.3}
        ],
        remappings=[
            ('imu', topics[src_index][0])
        ],
        output='screen'
    )
    ld.add_action(image_node)
    ld.add_action(vio_node)

    return ld
