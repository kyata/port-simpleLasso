cmake_minimum_required(VERSION 2.8)

find_package(PkgConfig REQUIRED)
if(APPLE)
    pkg_search_module(GTEST REQUIRED gtest_main)
    set(GTEST_LIBS ${GTEST_LDFLAGS})
else(APPLE)
    find_package(GTest REQUIRED)
    set(GTEST_LIBS ${GTEST_BOTH_LIBRARIES} pthread)
endif(APPLE)
find_package(Eigen3 REQUIRED)

include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

set(TEST_SOURCES
    test_cd.cc
)

# Enable to C++11 building
add_definitions(-std=c++11)

# create executable files
add_executable(test_cd ${TEST_SOURCES})

link_directories(${CMAKE_BINARY_DIR}/src)
target_link_libraries(test_cd
    simpleLasso
    ${GTEST_LIBS}
)
target_compile_options(test_cd PUBLIC ${GTEST_CFLAGS})
add_test(NAME AllTests COMMAND test_cd)