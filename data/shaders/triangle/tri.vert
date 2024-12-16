#version 450 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec4 color;
layout (location = 2) in vec2 uv;

layout (binding = 0) uniform GlobalUniformStore {
	mat4 WorldViewProj;
};

layout (location = 0) out gl_PerVertex {
	vec4 gl_Position;
};

layout (location = 0) out VS_OUT_FS_IN {
	vec4 color;
	vec2 uv;
} vs_out;

void main() {
	gl_Position = WorldViewProj * vec4(pos, 1.0f);
	vs_out.color = color;
	vs_out.uv = uv;
}
