#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/ext/matrix_transform.hpp>

union SDL_Event;

struct ArcballCamera
{
    glm::mat4 translation{ glm::identity<glm::mat4>() };
    glm::mat4 center_translation{ glm::identity<glm::mat4>() };
    glm::quat rotation{ glm::identity<glm::quat>() };
    glm::mat4 view_transform{ glm::identity<glm::mat4>() };
    glm::mat4 inverse_view_transform{ glm::identity<glm::mat4>() };
    glm::vec2 inv_screen{ 1.0f, 1.0f };
    glm::vec2 prev_mouse{ 0.0f, 0.0f };
    float zoom_speed{ 1.0f };
    bool is_rotating{ false };
    bool is_first_rotation{ true };
    bool is_panning{ false };
    bool is_first_panning{ true };

    static constexpr const float INITIAL_FOV = 30.0f;
    static constexpr const float TRANSLATION_FACTOR = 1.0f;

    ArcballCamera(glm::vec3 center, const float zoom_speed, glm::ivec2 screen_size) noexcept;

    void update() noexcept
    {
        this->view_transform = translation * glm::toMat4(rotation) * center_translation;
        this->inverse_view_transform = glm::inverse(view_transform);
    }

    void update_screen(const int32_t width, const int32_t height) noexcept
    {
        inv_screen = glm::vec2{ 1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height) };
    }

    void rotate(const glm::vec2 mouse_pos) noexcept;
    void end_rotate() noexcept { is_rotating = false; }

    glm::quat screen_to_arcball(const glm::vec2 p) noexcept
    {
        const float distance = glm::dot(p, p);

        if (distance <= 1.0f) {
            return glm::normalize(glm::quat{ 0.0f, p.x, p.y, glm::sqrt(1.0f - distance) });
        } else {
            const glm::vec2 unit_p = glm::normalize(p);
            return glm::normalize(glm::quat{ 0.0f, unit_p.x, unit_p.y, 0.0f });
        }
    }

    void zoom(const float amount, const float) noexcept
    {
        const glm::vec3 motion{ 0.0f, 0.0f, amount };
        translation = glm::translate(translation, motion * zoom_speed);
        update();
    }

    void pan(const glm::vec2 mouse_cur) noexcept;

    void input_event(const SDL_Event* e) noexcept;
};
