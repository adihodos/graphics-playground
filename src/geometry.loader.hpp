#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <span>

#include <tl/optional.hpp>
#include <tl/expected.hpp>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>

#include <tiny_gltf.h>

#include "renderer.common.hpp"
#include "error.hpp"

struct GeometryNode
{
    tl::optional<uint32_t> parent{};
    std::string name{};
    glm::mat4 transform{ glm::identity<glm::mat4>() };
    // bbox
    uint32_t vertex_offset{};
    uint32_t index_offset{};
    uint32_t index_count{};
};

struct GeometryVertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec4 color;
    glm::vec4 tangent;
    glm::vec2 uv;
    uint32_t pbr_buf_id;
};

template<typename T>
struct FormatDescriptors;

template<>
struct FormatDescriptors<GeometryVertex>
{
    static constexpr const VertexFormatDescriptor descriptors[] = {
        { .size = 3, .type = GL_FLOAT, .offset = offsetof(GeometryVertex, pos), .normalized = false },
        { .size = 3, .type = GL_FLOAT, .offset = offsetof(GeometryVertex, normal), .normalized = false },
        { .size = 4, .type = GL_FLOAT, .offset = offsetof(GeometryVertex, color), .normalized = false },
        { .size = 4, .type = GL_FLOAT, .offset = offsetof(GeometryVertex, tangent), .normalized = false },
        { .size = 2, .type = GL_FLOAT, .offset = offsetof(GeometryVertex, uv), .normalized = false },
        { .size = 1, .type = GL_UNSIGNED_INT, .offset = offsetof(GeometryVertex, pbr_buf_id), .normalized = false },
    };
};

struct MaterialDefinition
{
    std::string name;
    uint32_t base_color_src;
    uint32_t metallic_src;
    uint32_t normal_src;
    float metallic_factor;
    float roughness_factor;
    glm::vec4 base_color_factor;
};

struct LoadedGeometry
{
    std::vector<GeometryNode> nodes{};
    std::unique_ptr<tinygltf::Model> gltf{ std::make_unique<tinygltf::Model>() };

    LoadedGeometry() = default;
    LoadedGeometry(LoadedGeometry&&) = default;
    LoadedGeometry(const LoadedGeometry&) = delete;
    LoadedGeometry& operator=(const LoadedGeometry&) = delete;

    static tl::expected<LoadedGeometry, GenericProgramError> from_file(const std::filesystem::path& path);
    static tl::expected<LoadedGeometry, GenericProgramError> from_memory(const std::span<const uint8_t> bytes);

    glm::uvec2 extract_data(void* vertex_buffer, void* index_buffer, const glm::uvec2 offsets);
    glm::uvec2 extract_single_node_data(void* vertex_buffer,
                                        void* index_buffer,
                                        const glm::uvec2 offsets,
                                        const tinygltf::Node& node,
                                        const tl::optional<uint32_t> parent);

    glm::uvec2 compute_vertex_index_count() const;
    glm::uvec2 compute_node_vertex_index_count(const tinygltf::Node& node) const;
};
