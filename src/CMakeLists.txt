cmake_minimum_required(VERSION 2.8)

# include path settings
set(INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/include
)
include_directories(${INCLUDE_DIR})

# sources settings
set(LIB_SOURCES
    cd.cc
)

# Enable to C++11 building
add_definitions(-std=c++11)

# create library
add_library(simpleLasso STATIC ${LIB_SOURCES} )
