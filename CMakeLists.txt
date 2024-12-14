cmake_minimum_required(VERSION 3.20)
project(playground LANGUAGES CXX)

include(FetchContent)

set(BUILD_EXAMPLES
    OFF
    CACHE INTERNAL "")
set(BUILD_SHARED_LIBS
    OFF
    CACHE INTERNAL "")
set(BUILD_TESTS
    OFF
    CACHE INTERNAL "")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

# FetchContent_Declare(libconfig GIT_REPOSITORY
# https://github.com/hyperrealm/libconfig)
#
# FetchContent_MakeAvailable(libconfig)
#
# FetchContent_Declare( serdepp GIT_REPOSITORY https://github.com/injae/serdepp
# GIT_TAG "origin/main") FetchContent_MakeAvailable(serdepp)

FetchContent_Declare(
  sdl3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL
  GIT_TAG "origin/main")
FetchContent_MakeAvailable(sdl3)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG "11.0.2")
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
  quill
  GIT_REPOSITORY https://github.com/odygrd/quill
  GIT_TAG "v7.5.0")
FetchContent_MakeAvailable(quill)

find_package(Vulkan REQUIRED COMPONENTS glslang shaderc_combined)

# include(FetchContent) FetchContent_Declare( tomlplusplus GIT_REPOSITORY
# https://github.com/marzer/tomlplusplus.git GIT_TAG v3.2.0)

# FetchContent_MakeAvailable(tomlplusplus)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/small_vector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/small_vector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/glad.cc)
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${CMAKE_PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(
  ${CMAKE_PROJECT_NAME} PRIVATE SDL3::SDL3-static Vulkan::shaderc_combined
                                fmt::fmt quill::quill)
