#version 450 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;
layout (location = 3) in vec4 tangent;
layout (location = 4) in vec2 uv;
layout (location = 5) in uint pbr_id;

layout (binding = 0) uniform GlobalUniformStore {
	mat4 WorldViewProj;
};

layout (location = 0) out gl_PerVertex {
	vec4 gl_Position;
};

layout (location = 0) out VS_OUT_FS_IN {
	vec3 pos;
	vec3 normal;
	vec4 color;
	vec4 tangent;
	vec2 uv;
	flat uint pbr_id;
} vs_out;

void main() {
	gl_Position = WorldViewProj * vec4(pos, 1.0f);
	vs_out.pos = pos;
	vs_out.normal = normal;
	vs_out.color = color;
	vs_out.tangent = tangent;
	vs_out.uv = uv;
	vs_out.pbr_id = pbr_id;
}
