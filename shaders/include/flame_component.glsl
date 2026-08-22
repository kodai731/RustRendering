#ifndef FLAME_COMPONENT_GLSL
#define FLAME_COMPONENT_GLSL

// Every struct shared with the Rust side lives here; mirrored in
// thyllore-effect-core/src/flame/gpu/components/ (FlameUBO and its members) and
// flame/branch.rs (VortexElement).

struct FlameColorBase {
    vec3 rgb;
    float occlusionLumRef;
};

struct FlameColorMid {
    vec3 rgb;
    float pad0;
};

struct FlameColorTip {
    vec3 rgb;
    float pad0;
};

struct FlameTemporalParams {
    float accumWeight;
    float frameIndex;
    float noiseAnisoY;
    float warpYScale;
};

struct FlameLightParams {
    vec3 direction;
    float selfShadowStrength;
};

struct FlameWarpStyle {
    float warpAmp;
    float warpFreq;
    float riseSpeed;
    float taperPower;
};

struct FlameEdgeStyle {
    float radiusTipRatio;
    float edgeLow;
    float edgeHigh;
    float whiteBoost;
};

struct FlameWindBend {
    vec2 windDirection;
    float bendAmount;
    float bendPower;
};

struct FlameTrailMeta {
    float sampleCount;
    float maxAge;
    float pad0;
    float pad1;
};

struct FlameEmitterParams {
    float kind;
    float ringMajorRatio;
    float ringAngularSpeed;
    float sdfSlabDepth;
};

struct FlameContourParams {
    float wiggleAmp;
    float anisoAxisAdvect;
    float rteBands;
    float sigmaDispersion;
};

struct FlameErosionResponse {
    float center;
    float kappa;
    float weight1;
    float weight2;
};

struct FlameWaveCfParams {
    float enabled;
    float shearLayerCount;
    float skippedPowerPlain;
    float skippedPowerEnv;
};

struct FlameBoundaryParams {
    float amp;
    float freq;
    float speed;
    float radiusRatio;
};

struct FlameNearFadeParams {
    float radius;
    float carveResidual;
    float edgeLow;
    float edgeHigh;
};

struct FlameProfileParams {
    float radiusActive;
    float radiusMax;
    float colorActive;
    float pad0;
};

struct FlameWaveShaping {
    float trackedCount;
    float envCoeff;
    float inverseScale;
    float amplitude;
};

struct FlameTipCarveParams {
    float depth;
    float invReach;
    float primitiveTop;
    float invPrimitiveRange;
};

struct FlameWarpStrainParams {
    float strainBase;
    float strainTip;
    float invReach;
    float invStrainNorm;
};

struct FlameWarpFormParams {
    float displacementForm;
    float burnoutGain;
    float pad0;
    float pad1;
};

struct FlameUnifiedParams {
    float enabled;
    float sigmaFloor;
    float pad0;
    float pad1;
};

struct FlameMixParams {
    float lo;
    float hi;
    float invCarrierStd;
    float heightGain;
    float scale;
    float radialGain;
    float pad0;
    float pad1;
};

struct FlameThermalParams {
    float densityExp;
    float tempExp;
    float tempHotK;
    float tempColdK;
    float wienCK;
    float pad0;
    float pad1;
    float pad2;
};

struct FlameSegmentParams {
    float count;
    float invCount;
    float pad0;
    float pad1;
};

struct FlameSpreadParams {
    float gain;
    float edgeOuterSharpen;
    float twistGain;
    float erosionNoiseGain;
};

struct FlameSupportMotion {
    float supportMargin;
    float meanderAmp;
    float swirlSpeed;
    float twistSpeed;
};

struct FlameTwistMode {
    float kappa;
    float omega;
    float phase;
    float amp;
};

struct FlameTwistField {
    FlameTwistMode modes[2];
    float coreRadiusSq;
    float pad0;
    float pad1;
    float pad2;
};

struct FlameMeanderMode {
    vec2 direction;
    float kappa;
    float omega;
    float phase;
    float pad0;
    float pad1;
    float pad2;
};

const int FLAME_BRANCH_MAX_ELEMENTS = 32;

struct FlameBranchElement {
    float spawnTime;
    float side;
    float azimuth;
    float spawnHeight;
    float size;
    float tilt;
    float alongOffset;
    float hash01;
    float trunkRadius;
    float pad0;
    float pad1;
    float pad2;
};

struct FlameBranchAgeProfile {
    float windFraction;
    float burnoutStartFraction;
    float burnoutReleaseFraction;
    float burnoutMargin;
    float burnoutTrunkInner;
    float pad0;
    float pad1;
    float pad2;
};

struct FlameBranchField {
    float count;
    float period;
    float life;
    float gain;
    float riseRate;
    float driftRate;
    float aspect;
    float coreRadius;
    float reachStart;
    float reachEnd;
    float envelopeTime;
    float coreOffset;
    float boundingPad;
    float boundingPadY;
    float pad1;
    float pad2;
    FlameBranchAgeProfile ageProfile;
    FlameBranchElement elements[FLAME_BRANCH_MAX_ELEMENTS];
};

layout(set = 1, binding = 0) uniform FlameUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 heightPrimitiveCoefficients[3];
    vec4 radialCoefficients[2];
    vec4 heightCoefficients[2];
    float time;
    float sigmaT;
    float intensity;
    float heightAxisScale;
    float noiseAmplitude;
    float noiseFrequency;
    float noiseScrollSpeed;
    float radialSharpness;
    FlameColorBase colorBase;
    FlameColorMid colorMid;
    FlameColorTip colorTip;
    FlameTemporalParams temporalData;
    FlameLightParams lightData;
    FlameWarpStyle warpStyle;
    FlameEdgeStyle edgeStyle;
    FlameWindBend windBend;
    mat4 trailUnitInverse;
    FlameTrailMeta trailMeta;
    vec4 trail_coefficients[4];
    FlameEmitterParams emitterParams;
    FlameContourParams contourParams;
    FlameErosionResponse erosionResponse;
    FlameWaveCfParams waveCfParams;
    FlameBoundaryParams boundaryParams;
    FlameNearFadeParams nearFadeParams;
    vec4 radiusCoefficients[2];
    vec4 colorRamp[8];
    vec4 tempRamp[8];
    FlameProfileParams profileParams;
    FlameWaveShaping waveParams;
    FlameTipCarveParams tipCarveParams;
    FlameWarpStrainParams warpStrainParams;
    FlameWarpFormParams warpFormParams;
    FlameUnifiedParams unifiedParams;
    FlameMixParams mixParams;
    FlameSegmentParams segmentParams;
    FlameThermalParams thermalParams;
    FlameSpreadParams spreadParams;
    FlameSupportMotion supportMotion;
    FlameTwistField twistField;
    FlameMeanderMode meanderModes[2];
    FlameBranchField branchField;
    vec4 waveModes[428];
    vec4 waveJitter[96];
} flame;

struct FlameVortexElement {
    vec3 center;
    vec3 outward;
    vec3 line;
    vec3 up;
    float reach;
    float coreRadius;
    float circulation;
    float alongOffset;
};

#endif
