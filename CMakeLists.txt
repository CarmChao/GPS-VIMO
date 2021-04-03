cmake_minimum_required(VERSION 3.5)
project(ros2_msckf)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_MODULE_PATH &{CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(std_srvs REQUIRED)
# find_package(pcl_ros REQUIRED)
# find_package(rosidl_default_generators REQUIRED)
find_package(custom_msgs  REQUIRED)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(PCL REQUIRED)
find_package(Glog REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(node_plugins "")

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/CameraMeasurement.msg"
#   "msg/FeatureMeasurement.msg"
#   "msg/TrackingInfo.msg"
#   DEPENDENCIES std_msgs
# )
# ament_export_dependencies(rosidl_default_runtime)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${BOOST_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

# add_library(image_processor SHARED
#   src/image_processor.cpp
#   src/utils.cpp)
# target_compile_definitions(image_processor
#   PRIVATE "COMPOSITION_BUILDING_DLL")
add_executable(image_processor 
    src/test.cpp
    src/image_processor.cpp
    src/utils.cpp)
ament_target_dependencies(image_processor
  Boost
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "cv_bridge"
  "image_transport"
  "message_filters"
  "custom_msgs"
  )
# rosidl_target_interfaces(image_processor ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(image_processor ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES})
# rclcpp_components_register_nodes(image_processor "ros2_msckf::ImageProcessor")
# set(node_plugins "${node_plugins}ros2_msckf::ImageProcessor;$<TARGET_FILE:image_processor>\n")

# add_library(msckf_vio SHARED
#   src/msckf_vio.cpp)
# # add_dependencies(msckf_vio ${PROJECT_NAME})
# target_compile_definitions(msckf_vio
#   PRIVATE "COMPOSITION_BUILDING_DLL")
add_executable(msckf_vio
  src/test_vio.cpp
  src/msckf_vio.cpp
  src/utils.cpp)
ament_target_dependencies(msckf_vio
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "nav_msgs"
  "tf2_ros"
  "std_srvs"
  "tf2_eigen"
  # "tf2"
  "pcl_conversions"
  "custom_msgs")
# rosidl_target_interfaces(msckf_vio ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(msckf_vio 
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${GLOG_LIBRARIES})
# rclcpp_components_register_nodes(msckf_vio "ros2_msckf::msckf_vio")
# set(node_plugins "${node_plugins}ros2_msckf::msckf_vio;$<TARGET_FILE:msckf_vio>\n")

if(NOT WIN32)
  ament_environment_hooks(
    "${ament_cmake_package_templates_ENVIRONMENT_HOOK_LIBRARY_PATH}")
endif()

# install(TARGETS
#   image_processor
#   msckf_vio
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin)
install(TARGETS
  image_processor
  msckf_vio
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
 config
 DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
 launch
 DESTINATION share/${PROJECT_NAME}
)

ament_package()
