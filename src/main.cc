#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <ranges>

#include <fmt/core.h>
#include <mio/mmap.hpp>

#include "fn.hpp"
#include "small_vector.hpp"

#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>
#include <quill/Logger.h>
#include <quill/sinks/ConsoleSink.h>
#include <quill/std/Vector.h>

#include <shaderc/shaderc.hpp>

#include <tl/expected.hpp>
#include <tl/optional.hpp>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_projection.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <stb_image.h>
#include <nuklear.h>
#include <tiny_gltf.h>

#include "arcball.camera.hpp"
#include "renderer.common.hpp"
#include "geometry.loader.hpp"
#include "error.hpp"

#define PL_STRINGIZE(x) #x

namespace fn = rangeless::fn;
using fn::operators::operator%;
using fn::operators::operator%=;

using namespace std;

quill::Logger* g_logger{};

struct UIContext
{
    struct nk_context* ctx;
};

template<typename error_func, typename called_function, typename... called_func_args>
auto
func_call_wrapper(const char* function_name,
                  error_func err_func,
                  called_function function,
                  called_func_args&&... func_args) -> std::invoke_result_t<called_function, called_func_args&&...>
    requires std::is_invocable_v<decltype(err_func)>
{
    if constexpr (std::is_same_v<std::invoke_result_t<called_function, called_func_args&&...>, void>) {
        std::invoke(function, std::forward<called_func_args&&>(func_args)...);
    } else {
        const auto func_result = std::invoke(function, std::forward<called_func_args&&>(func_args)...);
        if (!func_result) {
            LOG_ERROR(g_logger, "{} error: {}", function_name, err_func());
        }
        return func_result;
    }
}

#define CHECKED_SDL(sdl_func, ...) func_call_wrapper(PL_STRINGIZE(sdl_func), SDL_GetError, sdl_func, ##__VA_ARGS__)
#define CHECKED_OPENGL(opengl_func, ...)                                                                               \
    func_call_wrapper(PL_STRINGIZE(opengl_func), glGetError, opengl_func, ##__VA_ARGS__)

template<typename T>
tuple<GLenum, uint32_t, bool>
vertex_array_attrib_from_type()
{
    if constexpr (is_same_v<T, glm::vec2> || is_same_v<remove_cvref_t<T>, float[2]>) {
        return { GL_FLOAT, 2, false };
    } else if constexpr (is_same_v<T, glm::vec3> || is_same_v<remove_cvref_t<T>, float[3]>) {
        return { GL_FLOAT, 3, false };
    } else if constexpr (is_same_v<T, glm::vec4> || is_same_v<remove_cvref_t<T>, float[4]>) {
        return { GL_FLOAT, 4, false };
    } else if constexpr (is_same_v<remove_cvref_t<T>, nk_byte[4]>) {
        return { GL_UNSIGNED_BYTE, 4, true };
    }
}

template<typename T>
void
vertex_array_append_attrib(const GLuint vao, const uint32_t idx, const uint32_t offset)
{
    glEnableVertexArrayAttrib(vao, idx);
    const auto [attr_type, attr_count, attr_normalized] = vertex_array_attrib_from_type<T>();
    glVertexArrayAttribFormat(vao, idx, attr_count, attr_type, attr_normalized, offset);
    glVertexArrayAttribBinding(vao, idx, 0);
}

using glsl_preprocessor_define = pair<string_view, string_view>;

tl::expected<pair<GLenum, shaderc_shader_kind>, SystemError>
classify_shader_file(const filesystem::path& fspath)
{
    assert(fspath.has_extension());

    const filesystem::path ext = fspath.extension();
    if (ext == ".vert")
        return { pair{ GL_VERTEX_SHADER, shaderc_vertex_shader } };
    if (ext == ".frag")
        return { pair{ GL_FRAGMENT_SHADER, shaderc_fragment_shader } };

    return tl::make_unexpected(SystemError{ make_error_code(errc::not_supported) });
}

struct shader_log_tag
{};
struct program_log_tag
{};

template<typename log_tag>
void
print_log_tag(const GLuint obj)
{
    char temp_buff[1024];
    GLsizei log_size{};

    if constexpr (is_same_v<log_tag, shader_log_tag>) {
        glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &log_size);
    } else if constexpr (is_same_v<log_tag, program_log_tag>) {
        glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &log_size);
    }

    if (log_size <= 0) {
        return;
    }

    log_size = min<GLsizei>(log_size, size(temp_buff));

    if constexpr (is_same_v<log_tag, shader_log_tag>) {
        glGetShaderInfoLog(obj, log_size - 1, &log_size, temp_buff);
    } else if constexpr (is_same_v<log_tag, program_log_tag>) {
        glGetProgramInfoLog(obj, log_size - 1, &log_size, temp_buff);
    }

    temp_buff[log_size] = 0;
    LOG_ERROR(g_logger, "[program|shader] {} error:\n{}", obj, temp_buff);
}

tl::expected<GLuint, GenericProgramError>
create_gpu_program_from_memory(const GLenum shader_kind_gl,
                               const shaderc_shader_kind shader_kind_sc,
                               const string_view input_filename,
                               const string_view src_code,
                               const string_view entry_point,
                               const span<const glsl_preprocessor_define> preprocessor_defines,
                               const bool optimize = false)
{
    shaderc::Compiler compiler{};
    shaderc::CompileOptions compile_options{};

    for (const auto [macro_name, macro_val] : preprocessor_defines) {
        compile_options.AddMacroDefinition(
            macro_name.data(), macro_name.length(), macro_val.data(), macro_val.length());
    }

    compile_options.SetOptimizationLevel(optimize ? shaderc_optimization_level_performance
                                                  : shaderc_optimization_level_zero);
    compile_options.SetTargetEnvironment(shaderc_target_env_opengl, 0);

    shaderc::PreprocessedSourceCompilationResult preprocessing_result = compiler.PreprocessGlsl(
        src_code.data(), src_code.size(), shader_kind_sc, input_filename.data(), compile_options);

    if (preprocessing_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_ERROR(
            g_logger, "Shader {} preprocessing failure:\n{}", input_filename, preprocessing_result.GetErrorMessage());
        return tl::make_unexpected(ShadercError{ preprocessing_result.GetErrorMessage() });
    }

    const string_view preprocessed_source{ preprocessing_result.begin(), preprocessing_result.end() };
    LOG_INFO(g_logger, "Preprocessed shader:\n{}", preprocessed_source);

    shaderc::SpvCompilationResult compilation_result = compiler.CompileGlslToSpv(preprocessed_source.data(),
                                                                                 preprocessed_source.length(),
                                                                                 shader_kind_sc,
                                                                                 input_filename.data(),
                                                                                 compile_options);

    if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_ERROR(g_logger,
                  "Shader [[ {} ]] compilation:: {} error(s)\n{}",
                  input_filename,
                  compilation_result.GetNumErrors(),
                  compilation_result.GetErrorMessage());
        return tl::make_unexpected(ShadercError{ compilation_result.GetErrorMessage() });
    }

    span<const uint32_t> spirv_bytecode{ compilation_result.begin(),
                                         static_cast<size_t>(compilation_result.end() - compilation_result.begin()) };

    const GLuint shader_handle{ glCreateShader(shader_kind_gl) };
    glShaderBinary(1,
                   &shader_handle,
                   GL_SHADER_BINARY_FORMAT_SPIR_V,
                   spirv_bytecode.data(),
                   static_cast<GLsizei>(spirv_bytecode.size_bytes()));
    glSpecializeShader(shader_handle, entry_point.data(), 0, nullptr, nullptr);

    GLint compile_status{};
    glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &compile_status);

    if (compile_status != GL_TRUE) {
        print_log_tag<shader_log_tag>(shader_handle);
        glDeleteShader(shader_handle);
        return tl::make_unexpected(OpenGLError{ glGetError() });
    }

    const GLuint program_handle{ glCreateProgram() };
    glProgramParameteri(program_handle, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(program_handle, shader_handle);
    glLinkProgram(program_handle);

    GLint link_status{};
    glGetProgramiv(program_handle, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) {
        print_log_tag<program_log_tag>(program_handle);
    }

    glDetachShader(program_handle, shader_handle);
    glDeleteShader(shader_handle);

    if (!link_status) {
        glDeleteProgram(program_handle);
        return tl::make_unexpected(OpenGLError{ glGetError() });
    }

    return program_handle;
}

tl::expected<GLuint, GenericProgramError>
create_gpu_program_from_file(const filesystem::path& source_file,
                             const string_view entry_point,
                             const span<const glsl_preprocessor_define> preprocessor_defines,
                             const bool optimize = false)
{
    error_code e{};
    const auto file_size = filesystem::file_size(source_file, e);
    if (e) {
        LOG_ERROR(g_logger, "FS error {}", e.message());
        return tl::make_unexpected(SystemError{ e });
    }

    string shader_code{};
    shader_code.reserve(static_cast<size_t>(file_size) + 1);

    ifstream f{ source_file };
    if (!f) {
        LOG_ERROR(g_logger, "Can't open file {}", source_file.string());
        return tl::make_unexpected(SystemError{ make_error_code(errc::io_error) });
    }

    shader_code.assign(istreambuf_iterator<char>{ f }, istreambuf_iterator<char>{});

    return classify_shader_file(source_file).and_then([&](auto p) {
        const auto [shader_kind_gl, shader_kind_shaderc] = p;
        return create_gpu_program_from_memory(
            shader_kind_gl, shader_kind_shaderc, source_file.string(), shader_code, entry_point, preprocessor_defines);
    });
}

char monka_mega_scratch_msg_buffer[4 * 1024 * 1024];

void
gl_debug_callback(GLenum source,
                  GLenum type,
                  GLuint id,
                  GLenum severity,
                  GLsizei length,
                  const GLchar* message,
                  const void* userParam)
{
    const char* dbg_src_desc = [source]() {
        switch (source) {
            case GL_DEBUG_SOURCE_API:
                return "OpenGL API";
                break;

            case GL_DEBUG_SOURCE_SHADER_COMPILER:
                return "OpenGL Shader Compiler";
                break;

            case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
                return "Windowing system";
                break;

            case GL_DEBUG_SOURCE_THIRD_PARTY:
                return "Third party";
                break;

            case GL_DEBUG_SOURCE_APPLICATION:
                return "Application";
                break;

            case GL_DEBUG_SOURCE_OTHER:
            default:
                return "Other";
        }
    }();

#define DESC_TABLE_ENTRY(glid, desc)                                                                                   \
    case glid:                                                                                                         \
        return desc;                                                                                                   \
        break

    const char* msg_type_desc = [type]() {
        switch (type) {
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_ERROR, "error");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, "deprecated behavior");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, "undefined behavior");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_PERFORMANCE, "performance");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_PORTABILITY, "portability");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_MARKER, "marker");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_PUSH_GROUP, "push group");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_POP_GROUP, "pop group");
            DESC_TABLE_ENTRY(GL_DEBUG_TYPE_OTHER, "other");
            default:
                return "other";
                break;
        }
    }();

    const char* severity_desc = [severity]() {
        switch (severity) {
            DESC_TABLE_ENTRY(GL_DEBUG_SEVERITY_HIGH, "high");
            DESC_TABLE_ENTRY(GL_DEBUG_SEVERITY_MEDIUM, "medium");
            DESC_TABLE_ENTRY(GL_DEBUG_SEVERITY_LOW, "low");
            DESC_TABLE_ENTRY(GL_DEBUG_SEVERITY_NOTIFICATION, "notification");
            default:
                return "unknown";
                break;
        }
    }();

    auto result = fmt::format_to(monka_mega_scratch_msg_buffer,
                                 "[OpenGL debug]\nsource: {}\ntype: {}\nid {}({:#0x})\n{}",
                                 dbg_src_desc,
                                 msg_type_desc,
                                 id,
                                 id,
                                 message ? message : "no message");
    *result.out = 0;

    if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM) {
        LOG_ERROR(g_logger, "{}", monka_mega_scratch_msg_buffer);
    } else {
        // LOG_DEBUG(g_logger, "{}", monka_mega_scratch_msg_buffer);
    }
}

struct BufferMapping
{
    GLuint handle;
    GLintptr offset;
    GLsizei length;
    void* mapped_addr;

    static tl::expected<BufferMapping, OpenGLError> create(const GLuint buffer,
                                                           const GLintptr offset,
                                                           const GLbitfield access,
                                                           const GLsizei mapping_len = 0);

    BufferMapping(const BufferMapping&) = delete;
    BufferMapping& operator=(const BufferMapping&) = delete;

    BufferMapping(const GLuint buf, const GLintptr off, const GLsizei len, void* mem) noexcept
        : handle{ buf }
        , offset{ off }
        , length{ len }
        , mapped_addr{ mem }
    {
    }

    ~BufferMapping()
    {
        if (mapped_addr) {
            // CHECKED_OPENGL(glFlushMappedNamedBufferRange, handle, offset, length);
            CHECKED_OPENGL(glUnmapNamedBuffer, handle);
        }
    }

    BufferMapping(BufferMapping&& rhs) noexcept
    {
        memcpy(this, &rhs, sizeof(*this));
        memset(&rhs, 0, sizeof(rhs));
    }
};

tl::expected<BufferMapping, OpenGLError>
BufferMapping::create(const GLuint buffer, const GLintptr offset, const GLbitfield access, const GLsizei mapping_len)
{
    GLsizei maplength{ mapping_len };
    if (maplength == 0) {
        glGetNamedBufferParameteriv(buffer, GL_BUFFER_SIZE, &maplength);
    }

    void* mapped_addr = CHECKED_OPENGL(glMapNamedBufferRange, buffer, offset, maplength, access);
    if (!mapped_addr) {
        return tl::unexpected{ OpenGLError{ glGetError() } };
    }

    return tl::expected<BufferMapping, OpenGLError>{ BufferMapping{ buffer, offset, maplength, mapped_addr } };
}

struct Texture
{
    GLuint handle{};
    GLenum internal_fmt{};
    int32_t width{};
    int32_t height{};
    int32_t depth{ 1 };

    static tl::expected<Texture, GenericProgramError> from_file(const filesystem::path& path);
    static tl::expected<Texture, GenericProgramError> from_memory(
        const void* pixels,
        const int32_t width,
        const int32_t height,
        const int32_t channels,
        const tl::optional<uint32_t> mip_levels = tl::nullopt);

    void release() noexcept
    {
        if (handle)
            glDeleteTextures(1, &handle);
    }
};

tl::expected<Texture, GenericProgramError>
Texture::from_memory(const void* pixels,
                     const int32_t width,
                     const int32_t height,
                     const int32_t channels,
                     const tl::optional<uint32_t> mip_levels)
{
    constexpr const pair<GLenum, GLenum> gl_format_pairs[] = {
        { GL_R8, GL_RED }, { GL_RG8, GL_RG }, { GL_RGB8, GL_RGB }, { GL_RGBA8, GL_RGBA }
    };
    assert(channels >= 1 && channels <= 4);

    Texture texture{};
    texture.internal_fmt = gl_format_pairs[channels - 1].first;
    texture.width = width;
    texture.height = height;

    glCreateTextures(GL_TEXTURE_2D, 1, &texture.handle);
    glTextureStorage2D(texture.handle, mip_levels.value_or(1), texture.internal_fmt, width, height);
    glTextureSubImage2D(
        texture.handle, 0, 0, 0, width, height, gl_format_pairs[channels - 1].second, GL_UNSIGNED_BYTE, pixels);

    mip_levels.map([texid = texture.handle](const uint32_t) { glGenerateTextureMipmap(texid); });

    return tl::expected<Texture, GenericProgramError>{ texture };
}

tl::expected<Texture, GenericProgramError>
Texture::from_file(const filesystem::path& path)
{
    const string s_path = path.string();
    int32_t width{};
    int32_t height{};
    int32_t channels{};

    unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{ stbi_load(s_path.c_str(), &width, &height, &channels, 0),
                                                            &stbi_image_free };

    if (!pixels) {
        return tl::make_unexpected(SystemError{ make_error_code(errc::io_error) });
    }

    return from_memory(pixels.get(), width, height, channels, tl::optional<uint32_t>{ 8 });
}

struct nk_sdl_vertex
{
    float position[2];
    float uv[2];
    nk_byte col[4];
};

struct BackendUI
{
    SDL_Window* window{};
    struct UIDeviceData
    {
        nk_buffer cmds;
        nk_draw_null_texture tex_null;
        GLuint buffers[3]{}; // vertex + index + uniform
        GLuint vao{};
        GLuint gpu_programs[2]{};
        GLuint prog_pipeline{};
        Texture font_atlas{};
        GLuint sampler;
    } gl_state{};
    nk_context ctx;
    nk_font_atlas atlas;
    nk_font* default_font{};
    uint64_t time_of_last_frame{};

    static constexpr const uint32_t MAX_VERTICES = 8192;
    static constexpr const uint32_t MAX_INDICES = 65535;

    BackendUI()
    {
        nk_buffer_init_default(&gl_state.cmds);
        nk_init_default(&ctx, nullptr);
        nk_font_atlas_init_default(&atlas);
    }

    ~BackendUI();

    BackendUI(const BackendUI&) = delete;
    BackendUI& operator=(const BackendUI&) = delete;

    BackendUI(BackendUI&& rhs) noexcept
    {
        memcpy(this, &rhs, sizeof(*this));
        rhs.window = nullptr;
        memset(&rhs.gl_state, 0, sizeof(rhs.gl_state));
        nk_buffer_init_default(&rhs.gl_state.cmds);
        nk_init_default(&rhs.ctx, nullptr);
        nk_font_atlas_init_default(&rhs.atlas);
    }

    static tl::expected<BackendUI, GenericProgramError> create(SDL_Window* win);
    bool handle_event(const SDL_Event* e);

    auto new_frame() -> UIContext { return UIContext{ .ctx = &ctx }; }
    void input_begin() { nk_input_begin(&ctx); }
    void input_end();
    void render(const DrawParams& dp);
};

BackendUI::~BackendUI()
{
    gl_state.font_atlas.release();
    glDeleteBuffers(size(gl_state.buffers), gl_state.buffers);
    for (const auto prg : gl_state.gpu_programs) {
        glDeleteProgram(prg);
    }
    glDeleteProgramPipelines(1, &gl_state.prog_pipeline);
    glDeleteVertexArrays(1, &gl_state.vao);
    glDeleteSamplers(1, &gl_state.sampler);

    nk_buffer_free(&gl_state.cmds);
    nk_font_atlas_clear(&atlas);

    nk_free(&ctx);
}

tl::expected<BackendUI, GenericProgramError>
BackendUI::create(SDL_Window* win)
{
    BackendUI backend;
    backend.window = win;

    UIDeviceData& dev = backend.gl_state;

    const uint32_t buffer_sizes[] = { MAX_VERTICES * sizeof(nk_sdl_vertex), MAX_INDICES * sizeof(uint32_t), 1024 };
    glCreateBuffers(size(dev.buffers), dev.buffers);
    for (size_t idx = 0; idx < size(dev.buffers); ++idx) {
        glNamedBufferStorage(dev.buffers[idx], buffer_sizes[idx], nullptr, GL_MAP_WRITE_BIT);
    }

    glCreateVertexArrays(1, &dev.vao);
    vertex_array_append_attrib<decltype(nk_sdl_vertex{}.position)>(dev.vao, 0, offsetof(nk_sdl_vertex, position));
    vertex_array_append_attrib<decltype(nk_sdl_vertex{}.uv)>(dev.vao, 1, offsetof(nk_sdl_vertex, uv));
    vertex_array_append_attrib<decltype(nk_sdl_vertex{}.col)>(dev.vao, 2, offsetof(nk_sdl_vertex, col));

    glVertexArrayVertexBuffer(dev.vao, 0, dev.buffers[0], 0, sizeof(nk_sdl_vertex));
    glVertexArrayElementBuffer(dev.vao, dev.buffers[1]);

    static constexpr const char* UI_VERTEX_SHADER = R"#(
    #version 450 core
    layout (location = 0) in vec2 pos;
    layout (location = 1) in vec2 texcoord;
    layout (location = 2) in vec4 color;

    layout (binding = 0) uniform GlobalParams {
        mat4 WorldViewProj;
    };

    layout (location = 0) out gl_PerVertex {
        vec4 gl_Position;
    };

    layout (location = 0) out VS_OUT_FS_IN {
        vec2 uv;
        vec4 color;
    } vs_out;

    void main() {
        vs_out.uv = texcoord;
        vs_out.color = color;
        gl_Position = WorldViewProj * vec4(pos, 0.0f, 1.0f);
    }
    )#";

    constexpr const char* const UI_FRAGMENT_SHADER = R"#(
    #version 450 core

    layout (binding = 0) uniform sampler2D FontAtlas;
    layout (location = 0) in VS_OUT_FS_IN {
        vec2 uv;
        vec4 color;
    } fs_in;
    layout (location = 0) out vec4 FinalFragColor;

    void main() {
        FinalFragColor = fs_in.color * texture(FontAtlas, fs_in.uv);
    }
    )#";

    glCreateProgramPipelines(1, &dev.prog_pipeline);

    constexpr const tuple<const char*, const char*, GLbitfield, GLenum, shaderc_shader_kind, const char*>
        shader_create_data[] = { { UI_VERTEX_SHADER,
                                   "main",
                                   GL_VERTEX_SHADER_BIT,
                                   GL_VERTEX_SHADER,
                                   shaderc_vertex_shader,
                                   "ui_vertex_shader" },
                                 { UI_FRAGMENT_SHADER,
                                   "main",
                                   GL_FRAGMENT_SHADER_BIT,
                                   GL_FRAGMENT_SHADER,
                                   shaderc_fragment_shader,
                                   "ui_fragment_shader" } };

    size_t idx{};
    for (auto [shader_code, entry_point, shader_stage, shader_type, shaderc_kind, shader_id] : shader_create_data) {
        auto shader_prog =
            create_gpu_program_from_memory(shader_type, shaderc_kind, shader_id, shader_code, entry_point, {});
        if (!shader_prog)
            return tl::make_unexpected(shader_prog.error());

        dev.gpu_programs[idx++] = *shader_prog;
        glUseProgramStages(dev.prog_pipeline, shader_stage, *shader_prog);
    }

    {
        nk_font_atlas_begin(&backend.atlas);

        nk_font* zed =
            nk_font_atlas_add_from_file(&backend.atlas, "data/fonts/ZedMonoNerdFontMono-Medium.ttf", 20.0f, nullptr);

        nk_font* default_font = zed;

        /*struct nk_font *droid = nk_font_atlas_add_from_file(atlas, "../../../extra_font/DroidSans.ttf", 14, 0);*/
        /*struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "../../../extra_font/Roboto-Regular.ttf", 16,
         * 0);*/
        /*struct nk_font *future = nk_font_atlas_add_from_file(atlas, "../../../extra_font/kenvector_future_thin.ttf",
         * 13, 0);*/
        /*struct nk_font *clean = nk_font_atlas_add_from_file(atlas, "../../../extra_font/ProggyClean.ttf", 12, 0);*/
        /*struct nk_font *tiny = nk_font_atlas_add_from_file(atlas, "../../../extra_font/ProggyTiny.ttf", 10, 0);*/
        /*struct nk_font *cousine = nk_font_atlas_add_from_file(atlas, "../../../extra_font/Cousine-Regular.ttf", 13,
         * 0);*/
        int atlas_width{}, atlas_height{};
        const void* atlas_pixels =
            nk_font_atlas_bake(&backend.atlas, &atlas_width, &atlas_height, NK_FONT_ATLAS_RGBA32);

        auto maybe_texture = Texture::from_memory(atlas_pixels, atlas_width, atlas_height, 4);
        if (!maybe_texture)
            return tl::make_unexpected(maybe_texture.error());

        backend.gl_state.font_atlas = *maybe_texture;
        nk_font_atlas_end(
            &backend.atlas, nk_handle_id((int)backend.gl_state.font_atlas.handle), &backend.gl_state.tex_null);

        // if (backend.atlas.default_font)
        //     nk_style_set_font(&backend.ctx, &backend.atlas.default_font->handle);
        nk_style_set_font(&backend.ctx, &default_font->handle);

        glCreateSamplers(1, &dev.sampler);
        glSamplerParameteri(dev.sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glSamplerParameteri(dev.sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    backend.time_of_last_frame = SDL_GetTicks();
    return tl::expected<BackendUI, GenericProgramError>{ std::move(backend) };
}

bool
BackendUI::handle_event(const SDL_Event* evt)
{
    struct nk_context* ctx = &this->ctx;

    switch (evt->type) {
        case SDL_EVENT_KEY_UP: /* KEYUP & KEYDOWN share same routine */
        case SDL_EVENT_KEY_DOWN: {
            const bool down = evt->type == SDL_EVENT_KEY_DOWN;
            const bool* state = SDL_GetKeyboardState(0);
            switch (evt->key.key) {
                case SDLK_RSHIFT: /* RSHIFT & LSHIFT share same routine */
                case SDLK_LSHIFT:
                    nk_input_key(ctx, NK_KEY_SHIFT, down);
                    break;
                case SDLK_DELETE:
                    nk_input_key(ctx, NK_KEY_DEL, down);
                    break;
                case SDLK_RETURN:
                    nk_input_key(ctx, NK_KEY_ENTER, down);
                    break;
                case SDLK_TAB:
                    nk_input_key(ctx, NK_KEY_TAB, down);
                    break;
                case SDLK_BACKSPACE:
                    nk_input_key(ctx, NK_KEY_BACKSPACE, down);
                    break;
                case SDLK_HOME:
                    nk_input_key(ctx, NK_KEY_TEXT_START, down);
                    nk_input_key(ctx, NK_KEY_SCROLL_START, down);
                    break;
                case SDLK_END:
                    nk_input_key(ctx, NK_KEY_TEXT_END, down);
                    nk_input_key(ctx, NK_KEY_SCROLL_END, down);
                    break;
                case SDLK_PAGEDOWN:
                    nk_input_key(ctx, NK_KEY_SCROLL_DOWN, down);
                    break;
                case SDLK_PAGEUP:
                    nk_input_key(ctx, NK_KEY_SCROLL_UP, down);
                    break;
                case SDLK_Z:
                    nk_input_key(ctx, NK_KEY_TEXT_UNDO, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_R:
                    nk_input_key(ctx, NK_KEY_TEXT_REDO, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_C:
                    nk_input_key(ctx, NK_KEY_COPY, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_V:
                    nk_input_key(ctx, NK_KEY_PASTE, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_X:
                    nk_input_key(ctx, NK_KEY_CUT, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_B:
                    nk_input_key(ctx, NK_KEY_TEXT_LINE_START, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_E:
                    nk_input_key(ctx, NK_KEY_TEXT_LINE_END, down && state[SDL_SCANCODE_LCTRL]);
                    break;
                case SDLK_UP:
                    nk_input_key(ctx, NK_KEY_UP, down);
                    break;
                case SDLK_DOWN:
                    nk_input_key(ctx, NK_KEY_DOWN, down);
                    break;
                case SDLK_LEFT:
                    if (state[SDL_SCANCODE_LCTRL])
                        nk_input_key(ctx, NK_KEY_TEXT_WORD_LEFT, down);
                    else
                        nk_input_key(ctx, NK_KEY_LEFT, down);
                    break;
                case SDLK_RIGHT:
                    if (state[SDL_SCANCODE_LCTRL])
                        nk_input_key(ctx, NK_KEY_TEXT_WORD_RIGHT, down);
                    else
                        nk_input_key(ctx, NK_KEY_RIGHT, down);
                    break;
            }
        }
            return true;

        case SDL_EVENT_MOUSE_BUTTON_UP: /* MOUSEBUTTONUP & MOUSEBUTTONDOWN share same routine */
        case SDL_EVENT_MOUSE_BUTTON_DOWN: {
            const bool down = evt->type == SDL_EVENT_MOUSE_BUTTON_DOWN;
            const int x = evt->button.x, y = evt->button.y;
            switch (evt->button.button) {
                case SDL_BUTTON_LEFT:
                    if (evt->button.clicks > 1)
                        nk_input_button(ctx, NK_BUTTON_DOUBLE, x, y, down);
                    nk_input_button(ctx, NK_BUTTON_LEFT, x, y, down);
                    break;
                case SDL_BUTTON_MIDDLE:
                    nk_input_button(ctx, NK_BUTTON_MIDDLE, x, y, down);
                    break;
                case SDL_BUTTON_RIGHT:
                    nk_input_button(ctx, NK_BUTTON_RIGHT, x, y, down);
                    break;
            }
        }
            return true;

        case SDL_EVENT_MOUSE_MOTION:
            if (ctx->input.mouse.grabbed) {
                int x = (int)ctx->input.mouse.prev.x, y = (int)ctx->input.mouse.prev.y;
                nk_input_motion(ctx, x + evt->motion.xrel, y + evt->motion.yrel);
            } else
                nk_input_motion(ctx, evt->motion.x, evt->motion.y);
            return true;

        case SDL_EVENT_TEXT_INPUT: {
            nk_glyph glyph;
            memcpy(glyph, evt->text.text, NK_UTF_SIZE);
            nk_input_glyph(ctx, glyph);
        }
            return true;

        case SDL_EVENT_MOUSE_WHEEL:
            nk_input_scroll(ctx, nk_vec2((float)evt->wheel.x, (float)evt->wheel.y));
            return true;

        default:
            return false;
    }
}

void
BackendUI::render(const DrawParams& dp)
{
    UIDeviceData* dev = &this->gl_state;

    const uint64_t now = SDL_GetTicks();
    ctx.delta_time_seconds = (float)(now - time_of_last_frame) / 1000;
    time_of_last_frame = now;
    const struct nk_vec2 scale
    {
        1.0f, 1.0f
    };

    /* setup global state */
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);

    BufferMapping::create(dev->buffers[2], 0, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT)
        .map([&dp](BufferMapping ubo) {
            const glm::mat4 projection = glm::ortho(
                0.0f, static_cast<float>(dp.surface_width), static_cast<float>(dp.surface_height), 0.0f, -1.0f, 1.0f);
            memcpy(ubo.mapped_addr, &projection, sizeof(projection));
        });

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, dev->buffers[2]);
    glBindProgramPipeline(dev->prog_pipeline);
    glBindVertexArray(dev->vao);
    glBindSampler(0, dev->sampler);

    {
        //
        // convert draw commands and fill vertex + index buffer
        {
            auto vertex_buffer =
                BufferMapping::create(dev->buffers[0], 0, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
            auto index_buffer =
                BufferMapping::create(dev->buffers[1], 0, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

            assert(vertex_buffer && index_buffer);

            static const struct nk_draw_vertex_layout_element vertex_layout[] = {
                { NK_VERTEX_POSITION, NK_FORMAT_FLOAT, NK_OFFSETOF(struct nk_sdl_vertex, position) },
                { NK_VERTEX_TEXCOORD, NK_FORMAT_FLOAT, NK_OFFSETOF(struct nk_sdl_vertex, uv) },
                { NK_VERTEX_COLOR, NK_FORMAT_R8G8B8A8, NK_OFFSETOF(struct nk_sdl_vertex, col) },
                { NK_VERTEX_LAYOUT_END }
            };

            nk_convert_config config{};
            config.vertex_layout = vertex_layout;
            config.vertex_size = sizeof(nk_sdl_vertex);
            config.vertex_alignment = NK_ALIGNOF(struct nk_sdl_vertex);
            config.tex_null = dev->tex_null;
            config.circle_segment_count = 22;
            config.curve_segment_count = 22;
            config.arc_segment_count = 22;
            config.global_alpha = 1.0f;
            config.shape_AA = NK_ANTI_ALIASING_ON;
            config.line_AA = NK_ANTI_ALIASING_ON;

            /* setup buffers to load vertices and elements */
            nk_buffer vbuf;
            nk_buffer_init_fixed(
                &vbuf, vertex_buffer->mapped_addr, (nk_size)BackendUI::MAX_VERTICES * sizeof(nk_sdl_vertex));

            nk_buffer ebuf;
            nk_buffer_init_fixed(&ebuf, index_buffer->mapped_addr, (nk_size)BackendUI::MAX_INDICES * sizeof(uint32_t));
            nk_convert(&ctx, &dev->cmds, &vbuf, &ebuf, &config);
        }

        /* iterate over and execute each draw command */
        const nk_draw_command* cmd;
        const nk_draw_index* offset = nullptr;
        nk_draw_foreach(cmd, &ctx, &dev->cmds)
        {
            if (!cmd->elem_count)
                continue;

            glBindTextureUnit(0, static_cast<GLuint>(cmd->texture.id));
            glScissor((GLint)(cmd->clip_rect.x * scale.x),
                      (GLint)((dp.surface_height - (GLint)(cmd->clip_rect.y + cmd->clip_rect.h)) * scale.y),
                      (GLint)(cmd->clip_rect.w * scale.x),
                      (GLint)(cmd->clip_rect.h * scale.y));
            glDrawElements(GL_TRIANGLES, (GLsizei)cmd->elem_count, GL_UNSIGNED_INT, offset);
            offset += cmd->elem_count;
        }

        nk_clear(&ctx);
        nk_buffer_clear(&dev->cmds);
    }

    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void
BackendUI::input_end()
{
    struct nk_context* pctx = &ctx;

    if (pctx->input.mouse.grab) {
        SDL_SetWindowRelativeMouseMode(window, true);
    } else if (pctx->input.mouse.ungrab) {
        /* better support for older SDL by setting mode first; causes an extra mouse motion event */
        SDL_SetWindowRelativeMouseMode(window, false);
        SDL_WarpMouseInWindow(window, (int)pctx->input.mouse.prev.x, (int)pctx->input.mouse.prev.y);
    } else if (pctx->input.mouse.grabbed) {
        pctx->input.mouse.pos.x = pctx->input.mouse.prev.x;
        pctx->input.mouse.pos.y = pctx->input.mouse.prev.y;
    }
    nk_input_end(&ctx);
}

struct UniformBuffer
{
    glm::mat4 wvp;
};

struct SimpleDemo
{
    GLuint buffers[3];
    GLuint vertex_array;
    GLuint gpu_programs[2];
    GLuint program_pipeline[1];
    GLuint sampler;
    Texture texture;
    uint32_t indexcount{};

    static tl::expected<SimpleDemo, GenericProgramError> create();

    SimpleDemo(const SimpleDemo&) = delete;
    SimpleDemo& operator=(const SimpleDemo&) = delete;

    SimpleDemo() = default;
    SimpleDemo(SimpleDemo&& rhs) noexcept
    {
        memcpy(this, &rhs, sizeof(*this));
        memset(&rhs, 0, sizeof(*this));
    }

    ~SimpleDemo();

    void ui(UIContext* ui);
    void render(const DrawParams& dp);
};

void
SimpleDemo::ui(UIContext* ui)
{
    struct nk_context* ctx = ui->ctx;
    static nk_colorf bg{ .r = 0.10f, .g = 0.18f, .b = 0.24f, .a = 1.0f };

    if (nk_begin(ctx,
                 "Demo",
                 nk_rect(50, 50, 230, 250),
                 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE | NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {
        enum
        {
            EASY,
            HARD
        };
        static int op = EASY;
        static int property = 20;

        nk_layout_row_static(ctx, 30, 80, 1);
        if (nk_button_label(ctx, "button"))
            printf("button pressed!\n");
        nk_layout_row_dynamic(ctx, 30, 2);
        if (nk_option_label(ctx, "easy", op == EASY))
            op = EASY;
        if (nk_option_label(ctx, "hard", op == HARD))
            op = HARD;
        nk_layout_row_dynamic(ctx, 22, 1);
        nk_property_int(ctx, "Compression:", 0, &property, 100, 10, 1);

        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "background:", NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, 25, 1);
        if (nk_combo_begin_color(ctx, nk_rgb_cf(bg), nk_vec2(nk_widget_width(ctx), 400))) {
            nk_layout_row_dynamic(ctx, 120, 1);
            bg = nk_color_picker(ctx, bg, NK_RGBA);
            nk_layout_row_dynamic(ctx, 25, 1);
            bg.r = nk_propertyf(ctx, "#R:", 0, bg.r, 1.0f, 0.01f, 0.005f);
            bg.g = nk_propertyf(ctx, "#G:", 0, bg.g, 1.0f, 0.01f, 0.005f);
            bg.b = nk_propertyf(ctx, "#B:", 0, bg.b, 1.0f, 0.01f, 0.005f);
            bg.a = nk_propertyf(ctx, "#A:", 0, bg.a, 1.0f, 0.01f, 0.005f);
            nk_combo_end(ctx);
        }

        // char temp_buffer[1024];
        // auto out = fmt::format_to_n(temp_buffer, size(temp_buffer), "Vertices {}, indices {}", vtx_idx.x, vtx_idx.y);
        //*out.out = 0;
        // nk_label_colored(ctx, temp_buffer, NK_TEXT_CENTERED, nk_color{ 255, 0, 0, 255 });
    }
    nk_end(ctx);
}

void
SimpleDemo::render(const DrawParams& dp)
{
    const GLuint uniform_buffer = buffers[2];

    BufferMapping::create(uniform_buffer, 0, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT).map([&](BufferMapping m) {
        const glm::mat4 mtx = glm::identity<glm::mat4>();
        const glm::mat4 final = glm::perspectiveFov(glm::radians(65.0f),
                                                    static_cast<float>(dp.surface_width),
                                                    static_cast<float>(dp.surface_height),
                                                    -1.0f,
                                                    +1.0f) *
                                dp.cam->view_transform * mtx;
        memcpy(m.mapped_addr, &final, sizeof(final));
    });

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glBindVertexArray(vertex_array);
    glBindProgramPipeline(this->program_pipeline[0]);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform_buffer, 0, 4096);
    glBindTextureUnit(0, texture.handle);
    glBindSampler(0, sampler);
    glDrawElements(GL_TRIANGLES, indexcount, GL_UNSIGNED_INT, nullptr);
}

SimpleDemo::~SimpleDemo()
{
    glDeleteBuffers(size(buffers), buffers);
    glDeleteVertexArrays(1, &vertex_array);

    for (const GLuint prog : gpu_programs)
        glDeleteProgram(prog);

    glDeleteProgramPipelines(size(program_pipeline), program_pipeline);
    glDeleteSamplers(1, &sampler);
    glDeleteTextures(1, &texture.handle);
}

tl::expected<SimpleDemo, GenericProgramError>
SimpleDemo::create()
{
    SimpleDemo obj{};

    glCreateBuffers(size(obj.buffers), obj.buffers);

    auto geometry = LoadedGeometry::from_file("data/models/ivanova_fury.glb");
    if (!geometry)
        return tl::make_unexpected(geometry.error());

    const glm::uvec2 counts = geometry->compute_vertex_index_count();
    const GLsizei buffer_sizes[] = { static_cast<GLsizei>(counts.x * sizeof(GeometryVertex)),
                                     static_cast<GLsizei>(counts.y * sizeof(uint32_t)),
                                     4096 };

    for (size_t i = 0; i < size(obj.buffers); ++i) {
        glNamedBufferStorage(obj.buffers[i], buffer_sizes[i], nullptr, GL_MAP_WRITE_BIT);
    }

    llvm::SmallVector<BufferMapping, 4> mapped_buffers;
    for (size_t i = 0; i < 2; ++i) {
        auto mapping = BufferMapping::create(obj.buffers[i], 0, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        if (!mapping)
            return tl::make_unexpected(mapping.error());

        mapped_buffers.push_back(std::move(*mapping));
    }

    const glm::vec2 extracted_count =
        geometry->extract_data(mapped_buffers[0].mapped_addr, mapped_buffers[1].mapped_addr, glm::vec2{});
    LOG_INFO(g_logger, "Extracted {} vertices, {} indices", extracted_count.x, extracted_count.y);

    glCreateVertexArrays(1, &obj.vertex_array);

    for (size_t i = 0; i < size(FormatDescriptors<GeometryVertex>::descriptors); ++i) {
        const VertexFormatDescriptor& fd = FormatDescriptors<GeometryVertex>::descriptors[i];
        glVertexArrayAttribFormat(obj.vertex_array, i, fd.size, fd.type, fd.normalized, fd.offset);
        glEnableVertexArrayAttrib(obj.vertex_array, i);
        glVertexArrayAttribBinding(obj.vertex_array, i, 0);
    }

    glVertexArrayVertexBuffer(obj.vertex_array, 0, obj.buffers[0], 0, sizeof(GeometryVertex));
    glVertexArrayElementBuffer(obj.vertex_array, obj.buffers[1]);

    glCreateProgramPipelines(size(obj.program_pipeline), obj.program_pipeline);
    constexpr const tuple<const char*, const char*, GLbitfield> shader_create_data[] = {
        { "data/shaders/triangle/tri.vert", "main", GL_VERTEX_SHADER_BIT },
        { "data/shaders/triangle/tri.frag", "main", GL_FRAGMENT_SHADER_BIT }
    };

    size_t idx{};
    for (auto [shader_path, entry_point, shader_stage] : shader_create_data) {
        auto shader_prog = create_gpu_program_from_file(shader_path, entry_point, {});
        if (!shader_prog)
            return tl::make_unexpected(shader_prog.error());

        obj.gpu_programs[idx++] = *shader_prog;
        glUseProgramStages(obj.program_pipeline[0], shader_stage, *shader_prog);
    }

    auto maybe_texture = Texture::from_file("data/textures/ash_uvgrid02.jpg");
    if (!maybe_texture) {
        return tl::make_unexpected(maybe_texture.error());
    }

    obj.texture = *maybe_texture;
    obj.indexcount = counts.y;

    glCreateSamplers(1, &obj.sampler);
    glSamplerParameteri(obj.sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(obj.sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return tl::expected<SimpleDemo, GenericProgramError>{ std::move(obj) };
}

int
main(int, char**)
{
    quill::Backend::start();
    auto console_sink = quill::Frontend::create_or_get_sink<quill::ConsoleSink>("console_color_sink");
    g_logger = quill::Frontend::create_or_get_logger("global_logger", std::move(console_sink));
    g_logger->set_log_level(quill::LogLevel::Debug);

    LOG_INFO(g_logger, "Starting up ...");

    if (!CHECKED_SDL(SDL_InitSubSystem, SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        return EXIT_FAILURE;
    }

    SDL_Window* window{ CHECKED_SDL(
        SDL_CreateWindow, "SDL Window", 1600, 1200, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE) };

    if (!window) {
        return EXIT_FAILURE;
    }

    LOG_DEBUG(g_logger, "Window created {:p}", reinterpret_cast<const void*>(window));

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG | SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);

    SDL_GLContext gl_context = CHECKED_SDL(SDL_GL_CreateContext, window);
    if (!gl_context) {
        return EXIT_FAILURE;
    }

    LOG_INFO(g_logger, "OpenGL context created {:p}", reinterpret_cast<const void*>(gl_context));

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
        return EXIT_FAILURE;
    }

    glDebugMessageCallback(gl_debug_callback, nullptr);

    GLint num_shader_binary_formats{};
    glGetIntegerv(GL_NUM_SHADER_BINARY_FORMATS, &num_shader_binary_formats);

    LOG_INFO(g_logger, "Supported binary formats {}", num_shader_binary_formats);

    if (num_shader_binary_formats > 0) {
        vector<GLint> binary_format_names{};
        binary_format_names.resize(num_shader_binary_formats, -1);
        glGetIntegerv(GL_SHADER_BINARY_FORMATS, binary_format_names.data());

        char tmp_buff[512];
        for (const GLint fmt : binary_format_names) {
            tmp_buff[0] = 0;

            switch (fmt) {
                case GL_SHADER_BINARY_FORMAT_SPIR_V_ARB: {
                    auto res =
                        fmt::format_to(tmp_buff, "{} {:#x}", PL_STRINGIZE(GL_SHADER_BINARY_FORMAT_SPIR_V_ARB), fmt);
                    *res.out = 0;
                } break;

                default: {
                    auto res = fmt::format_to(tmp_buff, "Unknown {:#x}", fmt);
                    *res.out = 0;
                } break;
            }

            LOG_INFO(g_logger, "{}", tmp_buff);
        }
    }

    auto tri_demo = SimpleDemo::create();
    if (!tri_demo)
        return EXIT_FAILURE;

    auto ui_backend = BackendUI::create(window);
    if (!ui_backend)
        return EXIT_FAILURE;

    glm::ivec2 screen_size{};
    SDL_GetWindowSizeInPixels(window, &screen_size.x, &screen_size.y);
    ArcballCamera cam{ glm::vec3{ 0.0f, 0.0f, 0.0f }, 1.0f, screen_size };

    for (bool quit = false; !quit;) {
        SDL_Event e;

        UIContext ui_ctx = ui_backend->new_frame();
        ui_backend->input_begin();

        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_EVENT_QUIT: {
                    quit = true;
                } break;

                case SDL_EVENT_KEY_DOWN: {
                    const SDL_KeyboardEvent* ke = &e.key;
                    if (ke->key == SDLK_ESCAPE)
                        quit = true;
                } break;

                default:
                    break;
            }

            ui_backend->handle_event(&e);
            cam.input_event(&e);
        }

        ui_backend->input_end();
        cam.update();

        SDL_GetWindowSizeInPixels(window, &screen_size.x, &screen_size.y);
        glViewportIndexedf(0, 0.0f, 0.0f, static_cast<float>(screen_size.x), static_cast<float>(screen_size.y));

        const float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        glClearNamedFramebufferfv(0, GL_COLOR, 0, clear_color);
        glClearNamedFramebufferfi(0, GL_DEPTH_STENCIL, 0, 1.0f, 0xff);

        const DrawParams draw_params{ screen_size.x, screen_size.y, screen_size.x, screen_size.y, &cam };

        tri_demo->ui(&ui_ctx);
        tri_demo->render(draw_params);
        ui_backend->render(draw_params);

        SDL_GL_SwapWindow(window);

        this_thread::sleep_for(chrono::milliseconds(25));
    }

    SDL_Quit();
    LOG_INFO(g_logger, "Shutting down ...");

    return EXIT_SUCCESS;
}
