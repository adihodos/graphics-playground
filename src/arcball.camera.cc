#include "arcball.camera.hpp"
#include <SDL3/SDL_events.h>

ArcballCamera::ArcballCamera(glm::vec3 center, const float zoom_speed_, glm::ivec2 screen_size) noexcept
    : translation{ glm::translate(glm::identity<glm::mat4>(), glm::vec3{ 0.0f, 0.0f, -1.0f }) }
    , center_translation{ glm::inverse(glm::translate(glm::identity<glm::mat4>(), center)) }
    , rotation{ glm::identity<glm::quat>() }
    , view_transform{ glm::identity<glm::mat4>() }
    , inverse_view_transform{ glm::identity<glm::mat4>() }
    , inv_screen{ 1.0f / static_cast<float>(screen_size.x), 1.0f / static_cast<float>(screen_size.y) }
    , zoom_speed{ zoom_speed_ }
{
}

void
ArcballCamera::rotate(const glm::vec2 mouse_pos) noexcept
{
    const glm::vec2 mouse_cur{ glm::clamp(mouse_pos.x * 2.0f * inv_screen.x - 1.0f, -1.0f, 1.0f),
                               glm::clamp(1.0f - 2.0f * mouse_pos.y * inv_screen.y, -1.0f, 1.0f) };

    const glm::vec2 mouse_prev{ glm::clamp(prev_mouse.x * 2.0f * inv_screen.x - 1.0f, -1.0f, 1.0f),
                                glm::clamp(1.0f - 2.0f * prev_mouse.y * inv_screen.y, -1.0f, 1.0f) };

    const glm::quat mouse_cur_ball = screen_to_arcball(mouse_cur);
    const glm::quat mouse_prev_ball = screen_to_arcball(mouse_prev);
    rotation = mouse_cur_ball * mouse_prev_ball * rotation;

    prev_mouse = mouse_pos;
    update();
}

void
ArcballCamera::pan(const glm::vec2 mouse_cur) noexcept
{
    const glm::vec2 mouse_delta = mouse_cur - prev_mouse;
    const float zoom_dist = glm::abs(translation[3][3]);
    const glm::vec4 delta{ glm::vec4{ mouse_delta.x * inv_screen.x, -mouse_delta.y * inv_screen.y, 0.0f, 0.0f } *
                           zoom_dist };

    const glm::vec4 motion = inverse_view_transform * delta;
    center_translation = glm::translate(center_translation, glm::vec3{ motion });
    prev_mouse = mouse_cur;
    update();
}

void
ArcballCamera::input_event(const SDL_Event* e) noexcept
{
    switch (e->type) {
        case SDL_EVENT_MOUSE_BUTTON_DOWN: {
            if (e->button.button == SDL_BUTTON_MIDDLE) {
                is_rotating = true;
                is_first_rotation = true;
            }

            if (e->button.button == SDL_BUTTON_RIGHT) {
                is_panning = true;
                is_first_panning = true;
            }
        } break;

        case SDL_EVENT_MOUSE_BUTTON_UP: {
            if (e->button.button == SDL_BUTTON_MIDDLE) {
                is_rotating = false;
                is_first_rotation = true;
            }

            if (e->button.button == SDL_BUTTON_RIGHT) {
                is_panning = false;
                is_first_panning = true;
            }
        } break;

        case SDL_EVENT_MOUSE_MOTION: {
            const SDL_MouseMotionEvent* mme = &e->motion;
            if (is_rotating) {
                if (is_first_rotation) {
                    prev_mouse = glm::vec2{ (float)mme->x, (float)mme->y };
                    is_first_rotation = false;
                } else {
                    rotate(glm::vec2{ (float)mme->x, (float)mme->y });
                }
            }

            if (is_panning) {
                if (is_first_panning) {
                    prev_mouse = glm::vec2{ (float)mme->x, (float)mme->y };
                    is_first_panning = false;
                } else {
                    pan(glm::vec2{ (float)mme->x, (float)mme->y });
                }
            }
        } break;

        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
            update_screen(e->window.data1, e->window.data2);
        } break;

        case SDL_EVENT_MOUSE_WHEEL: {
            zoom(e->wheel.y, 0.0f);
        } break;

        default:
            break;
    }
}