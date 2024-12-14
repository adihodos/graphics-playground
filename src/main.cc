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
#include <optional>
#include <span>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <ranges>

#include "fn.hpp"
#include "small_vector.hpp"

#include <SDL3/SDL.h>
#include <glad/glad.h>

#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>
#include <quill/Logger.h>
#include <quill/sinks/ConsoleSink.h>

#include <shaderc/shaderc.hpp>

#define PL_STRINGIZE(x) #x

namespace fn = rangeless::fn;
using fn::operators::operator%;
using fn::operators::operator%=;

using namespace std;

template<typename T>
concept HasResizeAndClear = requires(T a) {
    a.resize(size_t{});
    a.clear();
};

// template<HasResizeAndClear MyContainer>
// void
// apply_to_cont(MyContainer& c, size_t new_size)
// {
//   c.clear();
//   c.resize(new_size);
// }
//

// template<typename Container>
//   requires HasResizeAndClear<Container>
// void
// apply_to_cont(Container& c, size_t new_size)
// {
//   c.clear();
//   c.resize(1024);
// }

// template<typename Container>
// void
// apply_to_cont(Container& c, size_t new_size)
//   requires HasResizeAndClear<Container>
// {
//   c.clear();
//   c.resize(new_size);
// }
//

template<typename Cont>
concept HasResizeWithClean = requires(Cont c) {
    c.clear();
    c.resize(size_t{});
};

template<typename... Containers>
void
resize_and_clear(size_t size, Containers&&... c)
    requires(HasResizeWithClean<Containers> && ...)
{
    // auto fn_apply = [size](auto x) {
    // 	cout << "\napplying";
    // 	x.resize(size);
    // 	x.clear();
    // };
    //
    // (fn_apply(c), ...);
    //(([size](auto x) { c.resize(size); c.clear(); }), ...);

    (
        [size](auto x) {
            cout << "\nApplying ...";
            x.resize(size);
            x.clear();
        }(std::forward<Containers>(c)),
        ...);
}

quill::Logger* g_logger{};

using glsl_preprocessor_define = pair<string_view, string_view>;

optional<pair<GLenum, shaderc_shader_kind>>
classify_shader_file(const filesystem::path& fspath)
{
    assert(fspath.has_extension());

    const filesystem::path ext = fspath.extension();
    if (ext == ".vert")
        return optional{ pair{ GL_VERTEX_SHADER, shaderc_vertex_shader } };
    if (ext == ".frag")
        return optional{ pair{ GL_FRAGMENT_SHADER, shaderc_fragment_shader } };

    return nullopt;
}

GLuint
compile_shader_from_memory(const GLenum shader_kind_gl,
                           const shaderc_shader_kind shader_kind_sc,
                           const string_view input_filename,
                           const string_view src_code,
                           const string_view entry_point,
                           const span<const glsl_preprocessor_define> preprocessor_defines,
                           const bool optimize = false)
{
    //
    // preprocess
    shaderc::Compiler compiler{};
    shaderc::CompileOptions compile_options{};

    for (const auto [macro_name, macro_val] : preprocessor_defines) {
        compile_options.AddMacroDefinition(
            macro_name.data(), macro_name.length(), macro_val.data(), macro_val.length());
    }

    // const shaderc_shader_kind shader_kind = [](const GLenum gl_shader_kind) {
    //   switch (gl_shader_kind) {
    //     case GL_VERTEX_SHADER:
    //       return shaderc_vertex_shader;
    //       break;
    //
    //     case GL_FRAGMENT_SHADER:
    //       return shaderc_fragment_shader;
    //       break;
    //
    //     default:
    //       assert(false && "Unsupported shader kind");
    //       return shaderc_vertex_shader;
    //       break;
    //   }
    // }(shader_type);

    shaderc::PreprocessedSourceCompilationResult preprocessing_result = compiler.PreprocessGlsl(
        src_code.data(), src_code.size(), shader_kind_sc, input_filename.data(), compile_options);

    if (preprocessing_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_ERROR(
            g_logger, "Shader {} preprocessing failure:\n{}", input_filename, preprocessing_result.GetErrorMessage());
        return 0;
    }

    compile_options.SetOptimizationLevel(optimize ? shaderc_optimization_level_performance
                                                  : shaderc_optimization_level_zero);
    compile_options.SetTargetEnvironment(shaderc_target_env_opengl, shaderc_env_version_opengl_4_5);
    compile_options.SetTargetSpirv(shaderc_spirv_version_1_0);

    const string_view preprocessed_source{ preprocessing_result.begin(), preprocessing_result.end() };

    LOG_INFO(g_logger, "Preprocessed shader:\n{}", preprocessed_source);

    shaderc::SpvCompilationResult compilation_result = compiler.CompileGlslToSpv(preprocessed_source.data(),
                                                                                 preprocessed_source.length(),
                                                                                 shader_kind_sc,
                                                                                 input_filename.data(),
                                                                                 compile_options);

    if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_ERROR(
            g_logger, "Shader {} compilation failure:\n{}", input_filename, preprocessing_result.GetErrorMessage());
        return 0;
    }

    const GLuint shader_handle{ glCreateShader(shader_kind_gl) };
    glShaderBinary(1,
                   &shader_handle,
                   GL_SHADER_BINARY_FORMAT_SPIR_V_ARB,
                   compilation_result.cbegin(),
                   static_cast<GLsizei>(distance(compilation_result.begin(), compilation_result.end())));
    glSpecializeShaderARB(shader_handle, entry_point.data(), 0, nullptr, nullptr);

    GLint compile_status{};
    glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &compile_status);

    char temp_buff[1024];
    if (compile_status != GL_TRUE) {
        GLsizei log_size{};
        glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &log_size);
        if (log_size > 0) {
            GLsizei buff_size = std::min<GLsizei>(size(temp_buff), log_size);
            glGetShaderInfoLog(shader_handle, buff_size, &log_size, temp_buff);
            temp_buff[log_size] = 0;
            LOG_ERROR(g_logger, "GL shader compile error:\n{}", temp_buff);
        }

        return 0;
    }

    const GLuint program_handle{ glCreateProgram() };
    glProgramParameteri(program_handle, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(program_handle, shader_handle);
    glLinkProgram(program_handle);

    GLint link_status{};
    glGetProgramiv(program_handle, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) {
        GLsizei log_size{};
        glGetProgramiv(program_handle, GL_INFO_LOG_LENGTH, &log_size);
        if (log_size > 0) {
            GLsizei buff_size = std::min<GLsizei>(size(temp_buff), log_size);
            glGetProgramInfoLog(program_handle, buff_size, &log_size, temp_buff);
            temp_buff[log_size] = 0;
            LOG_ERROR(g_logger, "GL program error:\n{}", temp_buff);
        }
    }

    glDetachShader(program_handle, shader_handle);
    glDeleteShader(shader_handle);

    return program_handle;
}

GLuint
compile_shader_from_file(const filesystem::path& source_file,
                         const string_view entry_point,
                         const span<const glsl_preprocessor_define> preprocessor_defines,
                         const bool optimize = false)
{
    error_code e{};
    const auto file_size = filesystem::file_size(source_file, e);
    if (e) {
        LOG_ERROR(g_logger, "FS error {}", e.message());
        return 0;
    }

    string shader_code{};
    shader_code.reserve(static_cast<size_t>(file_size) + 1);

    ifstream f{ source_file };
    if (!f) {
        LOG_ERROR(g_logger, "Can't open file {}", source_file.string());
        return 0;
    }

    shader_code.assign(istreambuf_iterator<char>{ f }, istreambuf_iterator<char>{});

    const auto [shader_kind_gl, shader_kind_shaderc] = *classify_shader_file(source_file);

    return compile_shader_from_memory(
        shader_kind_gl, shader_kind_shaderc, source_file.string(), shader_code, "main", preprocessor_defines);
}

void
gl_debug_callback(GLenum source,
                  GLenum type,
                  GLuint id,
                  GLenum severity,
                  GLsizei length,
                  const GLchar* message,
                  const void* userParam)
{
    LOG_DEBUG(g_logger, "GL: {}", message);
}

template<typename sdl_function, typename... sdl_func_args>
auto
sdl_func_call(const char* sdl_function_name,
              sdl_function function,
              sdl_func_args&&... func_args) -> std::invoke_result_t<sdl_function, sdl_func_args&&...>
{
    if constexpr (std::is_same_v<std::invoke_result_t<sdl_function, sdl_func_args&&...>, void>) {
        std::invoke(function, std::forward<sdl_func_args>(func_args)...);
    } else {
        const auto func_result = std::invoke(function, std::forward<sdl_func_args&&>(func_args)...);
        if (!func_result) {
            LOG_ERROR(g_logger, "{} error: {}", sdl_function_name, SDL_GetError());
        }
        return func_result;
    }
}

#define SDL_FUNC_CALL(sdl_func, ...) sdl_func_call(#sdl_func, sdl_func, ##__VA_ARGS__)

int
main(int, char**)
{
    // vector<int32_t> intvec{};
    // apply_to_cont(intvec, 1024);
    // assert(intvec.capacity() == 1024);
    //
    // unordered_map<int32_t, int32_t> intmap{};
    // apply_to_cont(intmap, 1024);

    // vector<int32_t> iv0{};
    // vector<int32_t> iv1{};
    //
    // resize_and_clear(1024, iv0, iv1);
    //

    quill::Backend::start();
    auto console_sink = quill::Frontend::create_or_get_sink<quill::ConsoleSink>("console_color_sink");
    g_logger = quill::Frontend::create_or_get_logger("global_logger", std::move(console_sink));
    g_logger->set_log_level(quill::LogLevel::Debug);

    LOG_INFO(g_logger, "Starting up ...");

    if (!SDL_FUNC_CALL(SDL_InitSubSystem, SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        return EXIT_FAILURE;
    }

    SDL_Window* window{ SDL_FUNC_CALL(
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

    SDL_GLContext gl_context = SDL_FUNC_CALL(SDL_GL_CreateContext, window);
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
                        fmt::format_to(tmp_buff, "{} {:0x}", PL_STRINGIZE(GL_SHADER_BINARY_FORMAT_SPIR_V_ARB), fmt);
                    *res.out = 0;
                } break;

                default: {
                    auto res = fmt::format_to(tmp_buff, "Unknown {:0x}", fmt);
                    *res.out = 0;
                } break;
            }

            LOG_INFO(g_logger, "{}", tmp_buff);
        }
    }

    const glsl_preprocessor_define shader_macros[] = { { "OUTPUT_COORDS", "vec4(1.0, 0.0, 1.0, 1.0)" } };
    compile_shader_from_file("shader.vert", "main", span{ shader_macros });

    array<char, 2048> temp_buffer{};

    for (bool quit = false; !quit;) {
        SDL_Event e;
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
        }

        int32_t width{};
        int32_t height{};
        SDL_GetWindowSizeInPixels(window, &width, &height);

        glViewportIndexedf(0, 0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));

        const float clear_color[] = { 0.0f, 1.0f, 0.0f, 1.0f };
        glClearNamedFramebufferfv(0, GL_COLOR, 0, clear_color);
        glClearNamedFramebufferfi(0, GL_DEPTH_STENCIL, 0, 1.0f, 0xff);

        SDL_GL_SwapWindow(window);

        this_thread::sleep_for(chrono::milliseconds(50));
    }

    SDL_Quit();
    LOG_INFO(g_logger, "Shutting down ...");

    return EXIT_SUCCESS;
}
