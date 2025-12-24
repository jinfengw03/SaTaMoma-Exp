# Install script for directory: /home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/install/pal_gripper_description")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/jiachenli/miniconda3/envs/ros39/bin/x86_64-conda-linux-gnu-objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE DIRECTORY FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/gazebo")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE DIRECTORY FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/meshes")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE DIRECTORY FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/robots")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE DIRECTORY FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/urdf")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE DIRECTORY FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/ros2_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/pal_gripper_description")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/pal_gripper_description")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description/environment" TYPE FILE FILES "/opt/ros/foxy/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description/environment" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description/environment" TYPE FILE FILES "/opt/ros/foxy/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description/environment" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_index/share/ament_index/resource_index/packages/pal_gripper_description")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description/cmake" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description/cmake" TYPE FILE FILES
    "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_core/pal_gripper_descriptionConfig.cmake"
    "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/ament_cmake_core/pal_gripper_descriptionConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pal_gripper_description" TYPE FILE FILES "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client/urdf/pal_gripper_description/build/pal_gripper_description/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
