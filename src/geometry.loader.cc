#include "geometry.loader.hpp"

#include <mio/mmap.hpp>
#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <glad/glad.h>

extern quill::Logger* g_logger;
using namespace std;

tl::expected<LoadedGeometry, GenericProgramError>
LoadedGeometry::from_file(const filesystem::path& path)
{
    error_code e{};
    mio::mmap_source mmaped_file = mio::make_mmap_source(path.string(), e);
    if (e) {
        return tl::make_unexpected<GenericProgramError>(SystemError{ e });
    }
    return from_memory(
        span<const uint8_t>{ reinterpret_cast<const uint8_t*>(mmaped_file.data()), mmaped_file.mapped_length() });
}

tl::expected<LoadedGeometry, GenericProgramError>
LoadedGeometry::from_memory(const std::span<const uint8_t> bytes)
{
    LoadedGeometry geometry{};
    tinygltf::TinyGLTF loader;

    string error;
    string warning;
    if (!loader.LoadBinaryFromMemory(geometry.gltf.get(), &error, &warning, bytes.data(), bytes.size_bytes())) {
        return tl::make_unexpected(GenericProgramError{ GLTFError{ error, warning } });
    }

    return tl::expected<LoadedGeometry, GenericProgramError>{ std::move(geometry) };
}

glm::uvec2
LoadedGeometry::compute_node_vertex_index_count(const tinygltf::Node& node) const
{
    glm::uvec2 result{ 0, 0 };
    for (const int child_idx : node.children) {
        result += compute_node_vertex_index_count(gltf->nodes[child_idx]);
    }

    if (node.mesh != -1) {
        const tinygltf::Mesh* mesh = &gltf->meshes[node.mesh];
        for (const tinygltf::Primitive& prim : mesh->primitives) {
            assert(prim.mode == TINYGLTF_MODE_TRIANGLES);

            auto attr_itr = prim.attributes.find("POSITION");
            if (attr_itr == cend(prim.attributes)) {
                LOG_ERROR(g_logger, "Missing POSITION attribute");
                continue;
            }

            if (prim.indices == -1) {
                LOG_ERROR(g_logger, "Unindexed geometry is not supported.");
                continue;
            }

            const tinygltf::Accessor& pos_accessor = gltf->accessors[attr_itr->second];
            assert(pos_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
            assert(pos_accessor.type == TINYGLTF_TYPE_VEC3);

            result.x += pos_accessor.count;

            const tinygltf::Accessor& index_accesor = gltf->accessors[prim.indices];
            assert(index_accesor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ||
                   index_accesor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
            assert(index_accesor.type == TINYGLTF_TYPE_SCALAR);

            result.y += index_accesor.count;
        }
    }

    return result;
}

glm::uvec2
LoadedGeometry::compute_vertex_index_count() const
{
    glm::uvec2 result{ 0, 0 };
    for (const tinygltf::Scene& s : gltf->scenes) {
        for (const int node_idx : s.nodes) {
            result += compute_node_vertex_index_count(gltf->nodes[node_idx]);
        }
    }

    return result;
}

glm::uvec2
LoadedGeometry::extract_data(void* vertex_buffer, void* index_buffer, const glm::uvec2 offsets)
{
    glm::uvec2 acc_offsets{ offsets };
    for (const tinygltf::Scene& s : gltf->scenes) {
        for (const int node_idx : s.nodes) {
            acc_offsets +=
                extract_single_node_data(vertex_buffer, index_buffer, acc_offsets, gltf->nodes[node_idx], tl::nullopt);
        }
    }

    return acc_offsets;
}

glm::uvec2
LoadedGeometry::extract_single_node_data(void* vertex_buffer,
                                         void* index_buffer,
                                         const glm::uvec2 offsets,
                                         const tinygltf::Node& node,
                                         const tl::optional<uint32_t> parent)
{
    const glm::mat4 node_transform = [n = &node]() {
        if (!n->matrix.empty()) {
            return glm::mat4{ glm::make_mat4(n->matrix.data()) };
        }

        const glm::mat4 scale = [n]() {
            const glm::mat4 m{ glm::identity<glm::mat4>() };
            return n->scale.empty() ? m : glm::scale(m, glm::vec3{ glm::make_vec3(n->scale.data()) });
        }();

        const glm::mat4 rotation = [n]() {
            const glm::mat4 m{ glm::identity<glm::mat4>() };
            return n->rotation.empty() ? m : glm::mat4{ glm::toMat4(glm::make_quat(n->rotation.data())) };
        }();

        const glm::mat4 translate = [n]() {
            const glm::mat4 m{ glm::identity<glm::mat4>() };
            return n->translation.empty() ? m : glm::translate(m, glm::vec3{ glm::make_vec3(n->translation.data()) });
        }();

        return translate * rotation * scale;
    }();

    const uint32_t node_id = static_cast<uint32_t>(nodes.size());
    this->nodes.emplace_back(parent, node.name, glm::mat4{ node_transform }, 0, 0, 0);

    glm::uvec2 acc_offsets{ offsets };
    for (const int child_idx : node.children) {
        acc_offsets += extract_single_node_data(
            vertex_buffer, index_buffer, acc_offsets, gltf->nodes[child_idx], tl::optional<uint32_t>{ node_id });
    }

    this->nodes[node_id].vertex_offset = acc_offsets.x;
    this->nodes[node_id].index_offset = acc_offsets.y;

    uint32_t vertex_count{};
    uint32_t index_count{};

    if (node.mesh != -1) {
        tl::optional<uint32_t> ancestor{ parent };
        glm::mat4 transform{ node_transform };

        while (ancestor) {
            const GeometryNode* a = &nodes[*ancestor];
            transform = a->transform * transform;
            ancestor = a->parent;
        }

        nodes[node_id].transform = transform;
        const glm::mat4 normals_matrix = glm::transpose(glm::inverse(transform));

        const tinygltf::Mesh* mesh = &gltf->meshes[node.mesh];
        for (const tinygltf::Primitive& prim : mesh->primitives) {
            assert(prim.mode == TINYGLTF_MODE_TRIANGLES);

            if (prim.indices == -1) {
                LOG_ERROR(g_logger, "Unindexed geometry is not supported.");
                continue;
            }

            GeometryVertex* dst_data{ reinterpret_cast<GeometryVertex*>(vertex_buffer) + acc_offsets.x };

            uint32_t primitive_vertices{};
            if (auto attr_itr = prim.attributes.find("POSITION"); attr_itr != cend(prim.attributes)) {
                const tinygltf::Accessor& accessor = gltf->accessors[attr_itr->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC3);

                assert(accessor.bufferView != -1);
                const tinygltf::BufferView& buffer_view = gltf->bufferViews[accessor.bufferView];

                assert(buffer_view.target == TINYGLTF_TARGET_ARRAY_BUFFER);
                assert(buffer_view.buffer != -1);
                const tinygltf::Buffer& buffer = gltf->buffers[buffer_view.buffer];

                span<const glm::vec3> src_data{ reinterpret_cast<const glm::vec3*>(
                                                    buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset),
                                                static_cast<size_t>(accessor.count) };

                for (size_t vtx = 0; vtx < static_cast<size_t>(accessor.count); ++vtx) {
                    dst_data[vtx].pos = transform * glm::vec4{ src_data[vtx], 1.0f };
                    dst_data[vtx].color = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
                }

                primitive_vertices = accessor.count;
            } else {
                LOG_ERROR(g_logger, "Missing POSITION attribute");
                continue;
            }

            if (auto attr_normals = prim.attributes.find("NORMAL"); attr_normals != cend(prim.attributes)) {
                const tinygltf::Accessor& accessor = gltf->accessors[attr_normals->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC3);

                assert(accessor.bufferView != -1);
                const tinygltf::BufferView& buffer_view = gltf->bufferViews[accessor.bufferView];
                assert(buffer_view.target == TINYGLTF_TARGET_ARRAY_BUFFER);
                assert(buffer_view.buffer != -1);

                const tinygltf::Buffer& buffer = gltf->buffers[buffer_view.buffer];

                span<const glm::vec3> src_data{ reinterpret_cast<const glm::vec3*>(
                                                    buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset),
                                                static_cast<size_t>(accessor.count) };

                for (size_t vtx = 0; vtx < static_cast<size_t>(accessor.count); ++vtx) {
                    dst_data[vtx].normal = glm::normalize(normals_matrix * glm::vec4(src_data[vtx], 0.0f));
                }
            }

            if (auto attr_texcoords = prim.attributes.find("TEXCOORD_0"); attr_texcoords != cend(prim.attributes)) {
                const tinygltf::Accessor& accessor = gltf->accessors[attr_texcoords->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC2);

                assert(accessor.bufferView != -1);
                const tinygltf::BufferView& buffer_view = gltf->bufferViews[accessor.bufferView];
                assert(buffer_view.target == TINYGLTF_TARGET_ARRAY_BUFFER);
                assert(buffer_view.buffer != -1);

                const tinygltf::Buffer& buffer = gltf->buffers[buffer_view.buffer];

                span<const glm::vec2> src_data{ reinterpret_cast<const glm::vec2*>(
                                                    buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset),
                                                static_cast<size_t>(accessor.count) };

                for (size_t vtx = 0; vtx < static_cast<size_t>(accessor.count); ++vtx) {
                    dst_data[vtx].uv = src_data[vtx];
                }
            }

            if (auto attr_tangent = prim.attributes.find("TANGENT"); attr_tangent != cend(prim.attributes)) {
                const tinygltf::Accessor& accessor = gltf->accessors[attr_tangent->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC4);

                assert(accessor.bufferView != -1);
                const tinygltf::BufferView& buffer_view = gltf->bufferViews[accessor.bufferView];
                assert(buffer_view.target == TINYGLTF_TARGET_ARRAY_BUFFER);
                assert(buffer_view.buffer != -1);

                const tinygltf::Buffer& buffer = gltf->buffers[buffer_view.buffer];

                span<const glm::vec4> src_data{ reinterpret_cast<const glm::vec4*>(
                                                    buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset),
                                                static_cast<size_t>(accessor.count) };

                for (size_t vtx = 0; vtx < static_cast<size_t>(accessor.count); ++vtx) {
                    dst_data[vtx].tangent = src_data[vtx];
                }
            }

            if (auto attr_color = prim.attributes.find("COLOR_0"); attr_color != cend(prim.attributes)) {
                const tinygltf::Accessor& accessor = gltf->accessors[attr_color->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC4);

                assert(accessor.bufferView != -1);
                const tinygltf::BufferView& buffer_view = gltf->bufferViews[accessor.bufferView];
                assert(buffer_view.target == TINYGLTF_TARGET_ARRAY_BUFFER);
                assert(buffer_view.buffer != -1);

                const tinygltf::Buffer& buffer = gltf->buffers[buffer_view.buffer];

                span<const glm::vec4> src_data{ reinterpret_cast<const glm::vec4*>(
                                                    buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset),
                                                static_cast<size_t>(accessor.count) };

                for (size_t vtx = 0; vtx < static_cast<size_t>(accessor.count); ++vtx) {
                    dst_data[vtx].tangent = src_data[vtx];
                }
            }

            //
            // indices
            {
                const tinygltf::Accessor& accessor = gltf->accessors[prim.indices];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ||
                       accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
                assert(accessor.type == TINYGLTF_TYPE_SCALAR);

                assert(accessor.bufferView != -1);
                const tinygltf::BufferView& buffer_view = gltf->bufferViews[accessor.bufferView];
                assert(buffer_view.buffer != -1);
                assert(buffer_view.byteStride == 0);
                const tinygltf::Buffer& buffer = gltf->buffers[buffer_view.buffer];

                uint32_t* dst_data{ reinterpret_cast<uint32_t*>(index_buffer) + acc_offsets.y };

                if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    span<const uint16_t> src_data{ reinterpret_cast<const uint16_t*>(buffer.data.data() +
                                                                                     buffer_view.byteOffset +
                                                                                     accessor.byteOffset),
                                                   static_cast<size_t>(accessor.count) };

                    for (size_t idx = 0; idx < static_cast<size_t>(accessor.count); ++idx) {
                        dst_data[idx] = src_data[idx];
                    }
                } else {
                    span<const uint32_t> src_data{ reinterpret_cast<const uint32_t*>(buffer.data.data() +
                                                                                     buffer_view.byteOffset +
                                                                                     accessor.byteOffset),
                                                   static_cast<size_t>(accessor.count) };

                    memcpy(dst_data, src_data.data(), src_data.size_bytes());
                }

                //
                // correct indices to account for the vertex offset
                for (size_t idx = 0; idx < static_cast<size_t>(accessor.count); ++idx) {
                    dst_data[idx] += acc_offsets.x;
                }

                vertex_count += primitive_vertices;
                acc_offsets.x += primitive_vertices;

                index_count += accessor.count;
                acc_offsets.y += accessor.count;
            }
        }
    }

    this->nodes[node_id].index_count = index_count;

    LOG_INFO(g_logger,
             "Node {}, offsets {} {}, vertices {}, indices {}",
             nodes[node_id].name,
             nodes[node_id].vertex_offset,
             nodes[node_id].index_offset,
             vertex_count,
             index_count);

    return glm::uvec2{ vertex_count, index_count };
}
