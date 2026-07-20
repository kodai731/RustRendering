#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/depth.glsl"

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(depth_any) out float gl_FragDepth;

layout(binding = 0) uniform sampler2D positionSampler;
layout(binding = 1) uniform sampler2D normalSampler;
layout(binding = 2) uniform sampler2D shadowMaskSampler;
layout(binding = 3) uniform sampler2D albedoSampler;

layout(binding = 4) uniform SceneData {
    vec4 lightPosition;
    vec4 lightColor;
    mat4 view;
    mat4 proj;
    int debugMode;
    float shadowStrength;
    int enableDistanceAttenuation;
    float exposureValue;
} sceneData;

layout(binding = 5) uniform usampler2D objectIDSampler;

layout(std140, binding = 6) uniform SelectionData {
    uvec4 selectedIDs[32];
    uint selectedCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} selection;

layout(push_constant) uniform PushConstants {
    int debugMode;
} pc;

const vec3 OUTLINE_COLOR = vec3(1.0, 0.5, 0.0);
const float OUTLINE_WIDTH = 1.5;

bool isSelected(uint id) {
    if (id == 0u) return false;
    for (uint i = 0u; i < selection.selectedCount; i++) {
        if (selection.selectedIDs[i].x == id) return true;
    }
    return false;
}

bool detectOutlineEdge() {
    vec2 texelSize = 1.0 / vec2(textureSize(objectIDSampler, 0));
    uint centerID = texture(objectIDSampler, fragTexCoord).r;
    bool centerSelected = isSelected(centerID);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            vec2 offset = vec2(float(dx), float(dy)) * texelSize * OUTLINE_WIDTH;
            uint neighborID = texture(objectIDSampler, fragTexCoord + offset).r;
            bool neighborSelected = isSelected(neighborID);

            if (centerSelected != neighborSelected) {
                return true;
            }
        }
    }
    return false;
}

void main() {
    // Sample G-Buffer
    vec3 worldPosition = texture(positionSampler, fragTexCoord).xyz;
    vec3 worldNormal = texture(normalSampler, fragTexCoord).xyz;
    float shadowMask = texture(shadowMaskSampler, fragTexCoord).r;
    vec4 albedo = texture(albedoSampler, fragTexCoord);

    // Check if this is a valid fragment (normal length check)
    bool isBackground = length(worldNormal) < 0.01;

    // Write scene depth for overlay depth testing
    if (isBackground) {
        gl_FragDepth = DEPTH_FAR;
    } else {
        gl_FragDepth = worldToClipDepth(worldPosition, sceneData.view, sceneData.proj);
    }

    // Alpha 0 marks "no opaque surface": volumetric passes (flame) blend
    // premultiplied coverage into it, which tonemap composites over the
    // background color.
    if (isBackground) {
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Debug view modes
    if (pc.debugMode == 1) {
        // Position view (world space) - visualize as color
        // Map world position to visible range
        vec3 posColor = worldPosition * 0.1 + 0.5;
        outColor = vec4(posColor, 1.0);
        return;
    }
    else if (pc.debugMode == 2) {
        // Normal view (world space) - map [-1,1] to [0,1]
        vec3 normalColor = normalize(worldNormal) * 0.5 + 0.5;
        outColor = vec4(normalColor, 1.0);
        return;
    }
    else if (pc.debugMode == 3) {
        // Shadow mask view - show shadow as grayscale
        outColor = vec4(vec3(shadowMask), 1.0);
        return;
    }
    else if (pc.debugMode == 4) {
        // N dot L view - visualize light direction relative to surface
        vec3 n = normalize(worldNormal);
        vec3 lightVector = sceneData.lightPosition.xyz - worldPosition;
        vec3 l = normalize(lightVector);
        float ndotl = dot(n, l);
        // Red = facing away from light, Green = facing toward light
        vec3 ndotlColor = ndotl > 0.0 ? vec3(0.0, ndotl, 0.0) : vec3(-ndotl, 0.0, 0.0);
        outColor = vec4(ndotlColor, 1.0);
        return;
    }
    else if (pc.debugMode == 5) {
        // Light direction view - show raw light direction as color
        vec3 lightVector = sceneData.lightPosition.xyz - worldPosition;
        vec3 l = normalize(lightVector);
        outColor = vec4(l * 0.5 + 0.5, 1.0);
        return;
    }
    else if (pc.debugMode == 6) {
        // View depth mode - visualize GBuffer depth in view space
        vec4 worldPos4 = texture(positionSampler, fragTexCoord);
        bool hasGeometry = worldPos4.w > 0.5;

        if (!hasGeometry) {
            outColor = vec4(0.0, 0.0, 0.2, 1.0);
            return;
        }

        vec4 viewPos = sceneData.view * vec4(worldPos4.xyz, 1.0);
        float viewDepth = -viewPos.z;

        float normalizedDepth = viewDepth * 0.005;
        normalizedDepth = clamp(normalizedDepth, 0.0, 1.0);

        outColor = vec4(0.0, normalizedDepth, 0.0, 1.0);
        return;
    }
    else if (pc.debugMode == 7) {
        uint objectID = texture(objectIDSampler, fragTexCoord).r;
        if (objectID == 0u) {
            outColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            float r = float((objectID * 37u) % 256u) / 255.0;
            float g = float((objectID * 97u) % 256u) / 255.0;
            float b = float((objectID * 151u) % 256u) / 255.0;
            outColor = vec4(r, g, b, 1.0);
        }
        return;
    }
    else if (pc.debugMode == 8) {
        uint objectID = texture(objectIDSampler, fragTexCoord).r;

        if (selection.selectedCount > 0u && objectID == selection.selectedIDs[0].x) {
            outColor = vec4(1.0, 0.5, 0.0, 1.0);
        } else if (objectID > 0u) {
            outColor = vec4(0.3, 0.3, 0.3, 1.0);
        } else {
            outColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
        return;
    }
    else if (pc.debugMode == 9) {
        float countVis = float(selection.selectedCount) / 5.0;
        outColor = vec4(countVis, countVis, countVis, 1.0);
        return;
    }
    else if (pc.debugMode == 10) {
        outColor = vec4(albedo.rgb, 1.0);
        return;
    }

    worldNormal = normalize(worldNormal);

    vec3 lightVector = sceneData.lightPosition.xyz - worldPosition;
    float lightDistance = length(lightVector);
    vec3 lightDir = lightVector / lightDistance;

    float diffuse = max(dot(worldNormal, lightDir), 0.0);

    float attenuation = 1.0;
    if (sceneData.enableDistanceAttenuation != 0) {
        attenuation = 1.0 / (1.0 + 0.01 * lightDistance + 0.001 * lightDistance * lightDistance);
    }

    float ambient = 0.3;

    float shadowFactor = mix(1.0, shadowMask, sceneData.shadowStrength);

    vec3 baseColor = albedo.rgb;

    vec3 ambientLight = ambient * vec3(1.0);
    vec3 diffuseLight = diffuse * attenuation * sceneData.lightColor.rgb;

    vec3 lighting = ambientLight + diffuseLight * shadowFactor;
    vec3 finalColor = baseColor * lighting;

    if (selection.selectedCount > 0u && detectOutlineEdge()) {
        outColor = vec4(OUTLINE_COLOR, 1.0);
    } else {
        outColor = vec4(finalColor, 1.0);
    }
}
