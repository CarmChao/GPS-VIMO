from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(get_package_share_directory('ros2_msckf'), 'config', 'camchain-imucam-euroc.yaml')
    # node1 = ComposableNode(
    #             package='ros2_msckf',
    #             node_plugin='ros2_msckf::ImageProcessor',
    #             node_name='image_processor',
    #             parameters=[config]
    #         )
    # node1.parameters
    container = ComposableNodeContainer(
        node_name ="ros2_msckf_container",
        node_namespace ='',
        package='rclcpp_components',
        node_executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='ros2_msckf',
                node_plugin='ros2_msckf::ImageProcessor',
                node_name='image_processor',
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
                    {'ransac_threshold': 3},
                    {'stereo_threshold': 5}],
                remappings=[
                    ('/imu0', '~/imu'),
                    ('/cam0/image_raw', '~/cam0_image'),
                    ('/cam1/image_raw', '~/cam1_image')
                ]
            )
        ],
        output='screen',
    )

    return launch.LaunchDescription([container])