#version 450 core

layout (location = 0) in VS_OUT_FS_IN {
	vec4 color;
	vec2 uv;
} fs_in;

layout (binding = 0) uniform sampler2D Material;
layout (location = 0) out vec4 FinalFragColor;

void main() {
	FinalFragColor = texture(Material, fs_in.uv);
}
