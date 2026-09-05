#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D inputSampler;

layout(push_constant) uniform PushConstants {
    float threshold;
    float knee;
    int isFirstPass;
} pc;

vec3 karisAverage(vec3 c) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return c / (1.0 + luma);
}

vec3 downsample13Tap(vec2 uv) {
    vec2 texelSize = 1.0 / vec2(textureSize(inputSampler, 0));

    vec3 a = texture(inputSampler, uv + texelSize * vec2(-1.0, -1.0)).rgb;
    vec3 b = texture(inputSampler, uv + texelSize * vec2( 0.0, -1.0)).rgb;
    vec3 c = texture(inputSampler, uv + texelSize * vec2( 1.0, -1.0)).rgb;
    vec3 d = texture(inputSampler, uv + texelSize * vec2(-0.5, -0.5)).rgb;
    vec3 e = texture(inputSampler, uv).rgb;
    vec3 f = texture(inputSampler, uv + texelSize * vec2( 0.5, -0.5)).rgb;
    vec3 g = texture(inputSampler, uv + texelSize * vec2(-1.0,  0.0)).rgb;
    vec3 h = texture(inputSampler, uv + texelSize * vec2( 1.0,  0.0)).rgb;
    vec3 i = texture(inputSampler, uv + texelSize * vec2(-0.5,  0.5)).rgb;
    vec3 j = texture(inputSampler, uv + texelSize * vec2( 0.5,  0.5)).rgb;
    vec3 k = texture(inputSampler, uv + texelSize * vec2(-1.0,  1.0)).rgb;
    vec3 l = texture(inputSampler, uv + texelSize * vec2( 0.0,  1.0)).rgb;
    vec3 m = texture(inputSampler, uv + texelSize * vec2( 1.0,  1.0)).rgb;

    if (pc.isFirstPass != 0) {
        vec3 g0 = (a + b + g + d) * 0.25;
        vec3 g1 = (b + c + d + f) * 0.25;
        vec3 g2 = (g + d + k + i) * 0.25;
        vec3 g3 = (d + f + i + j) * 0.25;
        vec3 g4 = e;

        g0 = karisAverage(g0);
        g1 = karisAverage(g1);
        g2 = karisAverage(g2);
        g3 = karisAverage(g3);
        g4 = karisAverage(g4);

        return g0 * 0.125 + g1 * 0.125 + g2 * 0.125 + g3 * 0.125 + g4 * 0.5;
    }

    vec3 result = e * 0.125;
    result += (b + g + h + l) * 0.0625;
    result += (d + f + i + j) * 0.125;
    result += (a + c + k + m) * 0.03125;

    return result;
}

vec3 applyThreshold(vec3 color) {
    float brightness = max(max(color.r, color.g), color.b);
    float soft = brightness - pc.threshold + pc.knee;
    soft = clamp(soft, 0.0, 2.0 * pc.knee);
    soft = soft * soft / (4.0 * pc.knee + 0.00001);
    float contribution = max(soft, brightness - pc.threshold);
    contribution /= max(brightness, 0.00001);
    return color * contribution;
}

void main() {
    vec3 color = downsample13Tap(fragTexCoord);

    if (pc.isFirstPass != 0) {
        color = applyThreshold(color);
    }

    outColor = vec4(color, 1.0);
}
