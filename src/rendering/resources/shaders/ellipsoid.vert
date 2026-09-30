#version 430 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_mvp;
uniform vec3 u_view_dir;

out float v_backface_factor;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);

    vec3 normal = normalize(a_position);
    float alignment = dot(normal, u_view_dir);
    v_backface_factor = clamp(-alignment, 0.0, 1.0);
}
