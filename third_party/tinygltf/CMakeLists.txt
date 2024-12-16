cmake_minimum_required(VERSION 3.20)
project(tinygltf LANGUAGES C CXX)

set(source_files
    ${PROJECT_SOURCE_DIR}/tiny_gltf.h ${PROJECT_SOURCE_DIR}/json.hpp
    ${PROJECT_SOURCE_DIR}/tiny_gltf.cc)

add_library(${PROJECT_NAME} STATIC ${source_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC stb)
