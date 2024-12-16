cmake_minimum_required(VERSION 3.20)
project(playground LANGUAGES C CXX)

include(cmake/CPM.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

# FetchContent_Declare(libconfig GIT_REPOSITORY
# https://github.com/hyperrealm/libconfig)
#
# FetchContent_MakeAvailable(libconfig)
#
# FetchContent_Declare( serdepp GIT_REPOSITORY https://github.com/injae/serdepp
# GIT_TAG "origin/main") FetchContent_MakeAvailable(serdepp)

find_package(Vulkan REQUIRED COMPONENTS glslang shaderc_combined)

cpmaddpackage("gh:libsdl-org/SDL#main")
cpmaddpackage("gh:fmtlib/fmt#11.0.2")
cpmaddpackage("gh:odygrd/quill@7.5.0")
cpmaddpackage("gh:g-truc/glm#bf71a834948186f4097caa076cd2663c69a10e1e")
cpmaddpackage("gh:TartanLlama/expected@1.1.0")
cpmaddpackage("gh:TartanLlama/optional@1.1.0")

add_subdirectory(third_party/stb)
add_subdirectory(third_party/tinygltf)

set(nk_files "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nuklear/nuklear.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nuklear/nuklear.c")
add_library(nuklear STATIC ${nk_files})

target_include_directories(
  nuklear PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nuklear")

target_compile_definitions(
  nuklear
  PUBLIC NK_INCLUDE_FIXED_TYPES
         NK_INCLUDE_STANDARD_IO
         NK_INCLUDE_STANDARD_VARARGS
         NK_INCLUDE_DEFAULT_ALLOCATOR
         NK_INCLUDE_VERTEX_BUFFER_OUTPUT
         NK_INCLUDE_FONT_BAKING
         NK_INCLUDE_DEFAULT_FONT
         NK_UINT_DRAW_INDEX)

add_custom_target(
  copy_data ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data
          ${CMAKE_BINARY_DIR}/data
  COMMENT "copy data directory"
  VERBATIM)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/small_vector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/small_vector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/glad.cc)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
add_dependencies(${CMAKE_PROJECT_NAME} copy_data)

target_include_directories(${CMAKE_PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  PRIVATE SDL3::SDL3-static
          Vulkan::shaderc_combined
          fmt::fmt
          quill::quill
          glm::glm
          tl::expected
          tl::optional
          stb
          nuklear
          tinygltf)
