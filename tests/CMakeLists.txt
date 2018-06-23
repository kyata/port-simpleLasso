cmake_minimum_required(VERSION 2.8)

find_package(PkgConfig REQUIRED)
#find_package(GTest REQUIRED)
pkg_search_module(GTEST REQUIRED gtest_main)

include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

set(TEST_SOURCES
    test_cd.cc
)

add_executable(test_cd ${TEST_SOURCES})
target_link_libraries(test_cd 
    simpleLasso
    ${GTEST_LDFLAGS}
)
target_compile_options(test_cd PUBLIC ${GTEST_CFLAGS})
add_test(NAME AllTests COMMAND test_cd)