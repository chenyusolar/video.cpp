cmake_minimum_required(VERSION 3.20)
project(video.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DVIDEO_DEBUG=1)
endif()

find_package(PkgConfig REQUIRED)

set(RUST_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(CORE_TARGET_DIR ${RUST_CORE_DIR}/target)
set(CORE_RELEASE_DIR ${CORE_TARGET_DIR}/release)

set(VIDEO_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/video.h
)

set(VIDEO_SOURCES
    cli/main.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(video ${VIDEO_SOURCES})

if(EXISTS ${CORE_RELEASE_DIR}/libvideo_core.a)
    target_link_libraries(video ${CORE_RELEASE_DIR}/libvideo_core.a)
    target_link_libraries(video m)
    target_link_libraries(video pthread)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(video dl)
    endif()
    message(STATUS "Linked with static library: ${CORE_RELEASE_DIR}/libvideo_core.a")
elseif(EXISTS ${CORE_RELEASE_DIR}/libvideo_core.so)
    target_link_libraries(video ${CORE_RELEASE_DIR}/libvideo_core.so)
    message(STATUS "Linked with shared library: ${CORE_RELEASE_DIR}/libvideo_core.so")
elseif(EXISTS ${CORE_RELEASE_DIR}/libvideo_core.dylib)
    target_link_libraries(video ${CORE_RELEASE_DIR}/libvideo_core.dylib)
    message(STATUS "Linked with dynamic library: ${CORE_RELEASE_DIR}/libvideo_core.dylib")
else()
    message(WARNING "No Rust core library found. Please build the Rust core first:
    cd core && cargo build --release
")
endif()

target_compile_definitions(video PRIVATE
    VIDEO_VERSION_MAJOR=0
    VIDEO_VERSION_MINOR=1
    VIDEO_VERSION_PATCH=0
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(video PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(video PRIVATE -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(video PRIVATE /O2 /Ob2)
endif()

install(TARGETS video DESTINATION bin)

message(STATUS "==========================================")
message(STATUS " video.cpp Configuration")
message(STATUS "==========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==========================================")
