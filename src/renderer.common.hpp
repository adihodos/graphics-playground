#pragma once

#include <cstdint>
#include <glad/glad.h>

struct VertexFormatDescriptor
{
    int32_t size;
    uint32_t type;
    uint32_t offset;
    bool normalized;
};

class ArcballCamera;

struct DrawParams
{
    int32_t surface_width;
    int32_t surface_height;
    int32_t display_width;
    int32_t display_height;
    ArcballCamera* cam;
};
