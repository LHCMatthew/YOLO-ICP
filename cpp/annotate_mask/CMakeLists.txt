cmake_minimum_required(VERSION 3.30)  # Updated for compatibility; adjust max to your CMake version if needed
project(annotate_mask)


find_package(Boost COMPONENTS system REQUIRED)  # Uncommented; PCL needs this
find_package(PCL REQUIRED)  # Specify minimal components to avoid visualization/VTK/Qt6
find_package(JsonCpp REQUIRED)

find_package(OpenCV REQUIRED)   # Not needed for basic PCD write

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(${PCL_DEFINITIONS})

set(MAIN_SRC
    annotate_mask.cpp
)

# Create executable
add_executable(annotate_mask ${MAIN_SRC})

get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${JSON_INC_PATH})

# Include directories
target_include_directories(annotate_mask PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(annotate_mask
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    # ${JSONCPP_LIBRARIES}
    jsoncpp_lib
)





# Linux
# cmake_minimum_required(VERSION 3.20)  # Updated for compatibility; adjust max to your CMake version if needed
# project(annotate_mask)


# find_package(Boost COMPONENTS system REQUIRED)  # Uncommented; PCL needs this
# find_package(PCL REQUIRED)  # Specify minimal components to avoid visualization/VTK/Qt6
# #find_package(JsonCpp REQUIRED)

# find_package(OpenCV REQUIRED)   # Not needed for basic PCD write

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_definitions(${PCL_DEFINITIONS})

# set(MAIN_SRC
#     annotate_mask.cpp
# )

# # Create executable
# add_executable(annotate_mask ${MAIN_SRC})

# #get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
# include_directories(${JSON_INCLUDE_PATH})

# # Include directories
# target_include_directories(annotate_mask PRIVATE
#     ${PCL_INCLUDE_DIRS}
#     ${OpenCV_INCLUDE_DIRS}
#     ${JSONCPP_INCLUDE_DIRS}
# )

# # Link libraries
# target_link_libraries(annotate_mask
#     ${PCL_LIBRARIES}
#     ${OpenCV_LIBS}
#     # ${JSONCPP_LIBRARIES}
#     /usr/lib/x86_64-linux-gnu/libjsoncpp.so.1.9.5
# )