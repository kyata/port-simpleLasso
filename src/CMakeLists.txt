cmake_minimum_required(VERSION 2.8)
# find_package(Boost 1.67 COMPONENTS python36 REQUIRED)
# find_package(PythonLibs 3 REQUIRED)
find_package(Eigen3 REQUIRED)

# Debug Build
set(CMAKE_BUILD_TYPE Debug)

# include path settings
set(INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    # ${Boost_INCLUDE_DIRS}
    # ${PYTHON_INCLUDE_DIRS}
)
include_directories(${INCLUDE_DIR})

# sources settings
set(LIB_SOURCES
    cd.cc
)

# Enable to C++11 building
add_definitions(-std=c++11)

# create library
add_library(simpleLasso STATIC
    ${LIB_SOURCES}
    # ${Boost_LIBRARIES}
    # ${PYTHON_LIBRARY}
)

add_executable(main main.cc)
target_link_libraries(main simpleLasso)

# Debug
set(CMAKE_BUILD_TYPE Debug)