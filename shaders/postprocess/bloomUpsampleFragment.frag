#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D inputSampler;

vec3 upsample9Tap(vec2 uv) {
    vec2 texelSize = 1.0 / vec2(textureSize(inputSampler, 0));

    vec3 a = texture(inputSampler, uv + texelSize * vec2(-1.0, -1.0)).rgb;
    vec3 b = texture(inputSampler, uv + texelSize * vec2( 0.0, -1.0)).rgb;
    vec3 c = texture(inputSampler, uv + texelSize * vec2( 1.0, -1.0)).rgb;
    vec3 d = texture(inputSampler, uv + texelSize * vec2(-1.0,  0.0)).rgb;
    vec3 e = texture(inputSampler, uv).rgb;
    vec3 f = texture(inputSampler, uv + texelSize * vec2( 1.0,  0.0)).rgb;
    vec3 g = texture(inputSampler, uv + texelSize * vec2(-1.0,  1.0)).rgb;
    vec3 h = texture(inputSampler, uv + texelSize * vec2( 0.0,  1.0)).rgb;
    vec3 i = texture(inputSampler, uv + texelSize * vec2( 1.0,  1.0)).rgb;

    vec3 result = e * 4.0;
    result += (b + d + f + h) * 2.0;
    result += (a + c + g + i);
    result *= (1.0 / 16.0);

    return result;
}

void main() {
    outColor = vec4(upsample9Tap(fragTexCoord), 1.0);
}
