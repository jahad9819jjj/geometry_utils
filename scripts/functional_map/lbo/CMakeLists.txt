cmake_minimum_required(VERSION 3.10)
project(LaplaceBeltramiOperator)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(lbo src/lbo_image.cpp)
target_link_libraries(lbo ${OpenCV_LIBS})