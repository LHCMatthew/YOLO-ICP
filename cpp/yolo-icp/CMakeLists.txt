cmake_minimum_required(VERSION 3.30)  # Updated for compatibility; adjust max to your CMake version if needed
project(pcd_write)


find_package(Boost COMPONENTS system REQUIRED)  # Uncommented; PCL needs this
find_package(PCL REQUIRED)  # Specify minimal components to avoid visualization/VTK/Qt6

find_package(OpenCV 4.10.0 REQUIRED)   # Not needed for basic PCD write
find_package(JsonCpp REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(${PCL_DEFINITIONS})

set(MAIN_SRC
  main.cpp
  goicp/jly_goicp.cpp
  goicp/jly_3ddt.cpp
  goicp/matrix.cpp
  goicp/ConfigMap.cpp
  goicp/StringTokenizer.cpp
)

# Create executable
add_executable(pcd_write ${MAIN_SRC})

get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)

# Include directories
target_include_directories(pcd_write PRIVATE
  ${PCL_INCLUDE_DIRS}
  goicp
  ${JSONCPP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(pcd_write
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  jsoncpp_lib
)




# Linux
# cmake_minimum_required(VERSION 3.20)  # Updated for compatibility; adjust max to your CMake version if needed
# project(pcd_write)


# find_package(Boost COMPONENTS system REQUIRED)  # Uncommented; PCL needs this
# find_package(PCL REQUIRED)  # Specify minimal components to avoid visualization/VTK/Qt6

# find_package(OpenCV 4.10.0 REQUIRED)   # Not needed for basic PCD write

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_definitions(${PCL_DEFINITIONS})

# set(MAIN_SRC
#   main.cpp
#   goicp/jly_goicp.cpp
#   goicp/jly_3ddt.cpp
#   goicp/matrix.cpp
#   goicp/ConfigMap.cpp
#   goicp/StringTokenizer.cpp
# )

# # Create executable
# add_executable(pcd_write ${MAIN_SRC})

# # Include directories
# target_include_directories(pcd_write PRIVATE
#   ${PCL_INCLUDE_DIRS}
#   goicp
#   ${OpenCV_INCLUDE_DIRS}
#   ${JSONCPP_INCLUDE_DIRS}
# )

# # Link libraries
# target_link_libraries(pcd_write
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBS}
#   /usr/lib/x86_64-linux-gnu/libjsoncpp.so.1.9.5
# )