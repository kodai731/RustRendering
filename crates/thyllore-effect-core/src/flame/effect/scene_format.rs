use crate::flame::ownership::ParameterOwner;
use crate::flame::*;
use cgmath::{Quaternion, Vector2, Vector3};
use thyllore_scene_core::declare_scene_format;

declare_scene_format! {
    component: FlameEffect,
    record: FlameSceneRecord,
    tag: ParameterOwner,
    items {
        tags: PARAMETER_OWNERSHIP,
        snapshot: flame_parameter_snapshot,
        scalars: FLAME_SCALAR_PARAMS,
        ui: FLAME_UI_PARAMS,
        overwrite: overwrite_persisted_fields,
    },
    persisted {
        position: [f32; 3] = Frame {
            get: |e| [e.position.x, e.position.y, e.position.z],
            set: |e, v| e.position = Vector3::new(v[0], v[1], v[2]),
        },
        rotation: [f32; 4] = Frame {
            get: |e| [e.rotation.s, e.rotation.v.x, e.rotation.v.y, e.rotation.v.z],
            set: |e, v| e.rotation = Quaternion::new(v[0], v[1], v[2], v[3]),
        },
        height: f32 = Frame { get: |e| e.height, set: |e, v| e.height = v,
            ui {
                                min: 0.05,
                max: 10.0,
                format: "%.2f",
            },
        },
        radius: f32 = Frame { get: |e| e.radius, set: |e, v| e.radius = v,
            ui {
                                min: 0.05,
                max: 10.0,
                format: "%.2f",
            },
        },
        sigma_t: f32 = Style { get: |e| e.sigma_t, set: |e, v| e.sigma_t = v },
        intensity: f32 = Style { get: |e| e.intensity, set: |e, v| e.intensity = v,
            ui {
                                min: 0.0,
                max: 10.0,
            },
        },
        color_base: [f32; 3] = Style { get: |e| e.color.base, set: |e, v| e.color.base = v },
        color_tip: [f32; 3] = Style { get: |e| e.color.tip, set: |e, v| e.color.tip = v },
        temperature_base_k: f32 = Style {
            get: |e| e.color.temperature_base_k,
            set: |e, v| e.color.temperature_base_k = v,
        },
        temperature_tip_k: f32 = Style {
            get: |e| e.color.temperature_tip_k,
            set: |e, v| e.color.temperature_tip_k = v,
        },
        use_blackbody: bool = Style {
            get: |e| e.color.use_blackbody,
            set: |e, v| e.color.use_blackbody = v,
        },
        noise_amplitude: f32 = Style {
            get: |e| e.noise.amplitude,
            set: |e, v| e.noise.amplitude = v,
            ui {
                                min: 0.0,
                max: 3.0,
            },
        },
        noise_contrast: f32 = Style {
            get: |e| e.noise.contrast,
            set: |e, v| e.noise.contrast = v,
            ui {
                                min: 0.25,
                max: 4.0,
                format: "%.2f",
            },
        },
        noise_frequency: f32 = Style {
            get: |e| e.noise.frequency,
            set: |e, v| e.noise.frequency = v,
        },
        noise_scroll_speed: f32 = Style {
            get: |e| e.noise.scroll_speed,
            set: |e, v| e.noise.scroll_speed = v,
        },
        time_scale: f32 = Frame { get: |e| e.time_scale, set: |e, v| e.time_scale = v,
            ui {
                                min: 0.0,
                max: 4.0,
            },
        },
        time_offset: f32 = Frame { get: |e| e.time_offset, set: |e, v| e.time_offset = v },
        warp_amp: f32 = Style { get: |e| e.warp.amp, set: |e, v| e.warp.amp = v },
        warp_freq: f32 = Style { get: |e| e.warp.freq, set: |e, v| e.warp.freq = v },
        rise_speed: f32 = Style {
            get: |e| e.warp.rise_speed,
            set: |e, v| e.warp.rise_speed = v,
        },
        rise_accel: f32 = Style {
            get: |e| e.warp.rise_accel,
            set: |e, v| e.warp.rise_accel = v,
            ui {
                min: 0.0,
                max: 10.0,
                format: "%.2f",
                tooltip: "Height gain of the upward noise advection: speed = rise_speed * (1 + rise_accel * h); 0 = uniform",
            },
        },
        taper_power: f32 = Shape {
            get: |e| e.warp.taper_power,
            set: |e, v| e.warp.taper_power = v,
        },
        radius_tip_ratio: f32 = Shape {
            get: |e| e.edge.radius_tip_ratio,
            set: |e, v| e.edge.radius_tip_ratio = v,
        },
        base_spread: f32 = Style {
            get: |e| e.edge.base_spread,
            set: |e, v| e.edge.base_spread = v,
            ui {
                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Fire pool at the foot: extra radius ratio at h = 0 fading to the plain taper at base_spread_height; 0 = off",
            },
        },
        base_spread_height: f32 = Style {
            get: |e| e.edge.base_spread_height,
            set: |e, v| e.edge.base_spread_height = v,
            ui {
                min: 0.02,
                max: 1.0,
                format: "%.2f",
                tooltip: "Normalized height over which the base spread fades out",
            },
        },
        edge_low: f32 = Style { get: |e| e.edge.low, set: |e, v| e.edge.low = v },
        edge_high: f32 = Style { get: |e| e.edge.high, set: |e, v| e.edge.high = v },
        white_boost: f32 = Style {
            get: |e| e.edge.white_boost,
            set: |e, v| e.edge.white_boost = v,
        },
        wind_direction: [f32; 2] = Frame {
            get: |e| [e.wind.direction.x, e.wind.direction.y],
            set: |e, v| e.wind.direction = Vector2::new(v[0], v[1]),
            scalars {
                wind_x: {
                    get: |e| e.wind.direction.x,
                    set: |e, v| e.wind.direction.x = v,
                },
                wind_z: {
                    get: |e| e.wind.direction.y,
                    set: |e, v| e.wind.direction.y = v,
                },
            },
        },
        bend_amount: f32 = Frame {
            get: |e| e.wind.bend_amount,
            set: |e, v| e.wind.bend_amount = v,
        },
        bend_power: f32 = Frame {
            get: |e| e.wind.bend_power,
            set: |e, v| e.wind.bend_power = v,
        },
        self_shadow_strength: f32 = Style {
            get: |e| e.self_shadow_strength,
            set: |e, v| e.self_shadow_strength = v,
        },
        envelope_peak: f32 = Shape {
            get: |e| e.envelope.peak,
            set: |e, v| e.envelope.peak = v,
            default: 0.35,
        },
        envelope_base: f32 = Shape {
            get: |e| e.envelope.base,
            set: |e, v| e.envelope.base = v,
            default: 0.45,
        },
        envelope_tail: f32 = Shape {
            get: |e| e.envelope.tail,
            set: |e, v| e.envelope.tail = v,
            default: 1.6,
        },
        radial_sharpness: f32 = Shape {
            get: |e| e.radial_sharpness,
            set: |e, v| e.radial_sharpness = v,
        },
        occlusion_lum_ref: f32 = Style {
            get: |e| e.color.occlusion_lum_ref,
            set: |e, v| e.color.occlusion_lum_ref = v,
        },
        contour_wiggle_amp: f32 = Style {
            get: |e| e.contour.wiggle_amp,
            set: |e, v| e.contour.wiggle_amp = v,
        },
        aniso_axis_advect: f32 = Style {
            get: |e| e.contour.aniso_axis_advect,
            set: |e, v| e.contour.aniso_axis_advect = v,
        },
        rte_bands: f32 = Style {
            get: |e| e.contour.rte_bands,
            set: |e, v| e.contour.rte_bands = v,
        },
        sigma_dispersion: f32 = Style {
            get: |e| e.contour.sigma_dispersion,
            set: |e, v| e.contour.sigma_dispersion = v,
        },
        tip_carve_depth: f32 = Style {
            get: |e| e.carve.tip.depth,
            set: |e, v| e.carve.tip.depth = v,
        },
        tip_carve_reach: f32 = Style {
            get: |e| e.carve.tip.reach,
            set: |e, v| e.carve.tip.reach = v,
        },
        warp_reach: f32 = Style { get: |e| e.warp.reach, set: |e, v| e.warp.reach = v },
        swirl_gain: f32 = Style { get: |e| e.swirl.gain, set: |e, v| e.swirl.gain = v,
            ui {
                label: "Swirl",
                min: 0.0,
                max: 1.5,
                format: "%.2f",
                tooltip: "Medium swirl share: strain budget spent on azimuthal shear (0 = off; raising it thins the carve warp)",
            },
        },
        swirl_speed: f32 = Style { get: |e| e.swirl.speed, set: |e, v| e.swirl.speed = v,
            ui {
                                min: 0.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "How fast the swirl layers counter-rotate against the rise (time-only: costs no strain budget)",
            },
        },
        spread_gain: f32 = Style { get: |e| e.spread_gain, set: |e, v| e.spread_gain = v,
            ui {
                label: "Spread",
                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Medium spread toward the tip: noise features enlarge, drift outward and dissolve as they rise (0 = rigid scroll)",
            },
        },
        support_margin: f32 = Style {
            get: |e| e.support_margin,
            set: |e, v| e.support_margin = v,
            ui {
                label: "Support",
                min: 1.0,
                max: 2.5,
                format: "%.2f",
                tooltip: "Flame density support radius: multiplier for the biweight support radius (how much extra space is allowed for carving). 1.0 is default; higher values result in larger support and may leave chunks at the outer edges.",
            },
        },
        meander_amp: f32 = Style { get: |e| e.meander.amp, set: |e, v| e.meander.amp = v,
            ui {
                label: "Meander",
                min: 0.0,
                max: 2.0,
                format: "%.2f",
                tooltip: "Horizontal meandering motion of the flame (0 = off)",
            },
        },
        meander_frequency: f32 = Style {
            get: |e| e.meander.frequency,
            set: |e, v| e.meander.frequency = v,
            ui {
                                min: 0.2,
                max: 30.0,
                format: "%.1f",
                tooltip: "Wavenumber multiplier of the meander modes: 1 = two long bends over the height, ~12 folds the column into a snake with ~4 bends (pillar reference)",
            },
        },
        mix_lo: f32 = Style { get: |e| e.mix.lo, set: |e, v| e.mix.lo = v,
            ui {
                                min: -3.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Erosion carrier level (std units, carve-positive) where a parcel starts mixing with ambient air: lower mixes more of the body",
            },
        },
        mix_hi: f32 = Style { get: |e| e.mix.hi, set: |e, v| e.mix.hi = v,
            ui {
                                min: -3.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "Carrier level (std units) where a parcel counts as fully mixed (thin and cold)",
            },
        },
        mix_height_gain: f32 = Style {
            get: |e| e.mix.height_gain,
            set: |e, v| e.mix.height_gain = v,
            ui {
                                min: 0.0,
                max: 2.0,
                format: "%.2f",
                tooltip: "Height ramp added to the mixing degree, gain * h^2: the plume thins and cools toward the top",
            },
        },
        mix_scale: f32 = Style { get: |e| e.mix.scale, set: |e, v| e.mix.scale = v,
            ui {
                                min: 0.1,
                max: 2.0,
                format: "%.2f",
                tooltip: "Wavenumber of the mixing eddies relative to the low erosion octave: below 1 the mixed and unmixed regions grow larger than the carve detail",
            },
        },
        mix_radial_gain: f32 = Style {
            get: |e| e.mix.radial_gain,
            set: |e, v| e.mix.radial_gain = v,
            ui {
                                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Shear-layer ramp added to the mixing degree, gain * u^2 over the normalized radius: the axis stays an unmixed bright core while the rim thins and cools",
            },
        },
        mix_core_radius: f32 = Style {
            get: |e| e.mix.core_radius,
            set: |e, v| e.mix.core_radius = v,
            ui {
                                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Normalized radius below which the noise mixing fades out (from half this radius): keeps the core one connected bright mass while the rim still breaks up",
            },
        },
        density_exp: f32 = Style {
            get: |e| e.thermal.density_exp,
            set: |e, v| e.thermal.density_exp = v,
            ui {
                                min: 0.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "Mass curve of a mixing parcel, (1 - m)^a: larger thins the mixed regions faster",
            },
        },
        temp_exp: f32 = Style {
            get: |e| e.thermal.temp_exp,
            set: |e, v| e.thermal.temp_exp = v,
            ui {
                                min: 0.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "Temperature curve of a mixing parcel, T_cold + (T_hot - T_cold) (1 - m)^b: larger than Density Exp cools before thinning (dark red tufts remain), smaller thins before cooling",
            },
        },
        wien_c_k: f32 = Style {
            get: |e| e.thermal.wien_c_k,
            set: |e, v| e.thermal.wien_c_k = v,
            ui {
                label: "Wien C (K)",
                min: 0.0,
                max: 24000.0,
                format: "%.0f",
                tooltip: "Wien constant of the emissivity exp(-c/T): 24000 is physical at 0.6 um, smaller compresses the hot/cold brightness contrast like camera exposure",
            },
        },
        wave_segments: u32 = Frame {
            get: |e| e.wave_segments,
            set: |e, v| e.wave_segments = v,
        },
        noise_aniso_y: f32 = Style {
            get: |e| e.noise.aniso_y,
            set: |e, v| e.noise.aniso_y = v,
            ui {
                label: "Noise Aspect",
                min: 0.05,
                max: 1.5,
                format: "%.2f",
                tooltip: "Vertical scale of the noise cells: small = tall streaks, 1 = isotropic puffs (in the height-scaled mode)",
            },
        },
        noise_lobe_scale: f32 = Style {
            get: |e| e.noise.lobe_scale,
            set: |e, v| e.noise.lobe_scale = v,
            ui {
                label: "Lobe Scale",
                min: 0.1,
                max: 1.5,
                format: "%.2f",
                tooltip: "Knee of the silhouette-scale low octaves: smaller lets larger, rounder lobes through, larger keeps only the fine carving band",
            },
        },
        noise_lobe_aniso: f32 = Style {
            get: |e| e.noise.lobe_aniso,
            set: |e, v| e.noise.lobe_aniso = v,
            ui {
                label: "Lobe Aspect",
                min: 0.25,
                max: 2.0,
                format: "%.2f",
                tooltip: "Vertical wavenumber multiplier of the low octaves: below 1 stretches the lobes into tall streaks, above 1 flattens them into stacked puffs",
            },
        },
        edge_outer_sharpen: f32 = Style {
            get: |e| e.edge.outer_sharpen,
            set: |e, v| e.edge.outer_sharpen = v,
        },
        noise_scale_mode: f32 = Style {
            get: |e| e.noise.scale_mode,
            set: |e, v| e.noise.scale_mode = v,
        },
        erosion_noise_gain: f32 = Style {
            get: |e| e.noise.erosion_gain,
            set: |e, v| e.noise.erosion_gain = v,
        },
        twist_gain: f32 = Style { get: |e| e.twist.gain, set: |e, v| e.twist.gain = v,
            ui {
                label: "Twist",
                min: 0.0,
                max: 8.0,
                format: "%.2f",
                tooltip: "Azimuthal twist of the noise pattern around the axis (radians at the tip; a rotation never folds, so any amplitude is structurally safe; 0 = off)",
            },
        },
        twist_speed: f32 = Style { get: |e| e.twist.speed, set: |e, v| e.twist.speed = v,
            ui {
                                min: 0.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "Twist rotation rate scale (0 = follow Swirl Speed; > 0 gives the twist its own rate so depth and speed tune independently)",
            },
        },
        burnout_gain: f32 = Style {
            get: |e| e.carve.burnout_gain,
            set: |e, v| e.carve.burnout_gain = v,
            ui {
                label: "Burnout",
                min: 0.0,
                max: 32.0,
                format: "%.2f",
                tooltip: "Age-driven burnout of the rising material: deepens the erosion mean toward the flame top so noise troughs sever the column (base shedding) and detached tongues dissolve (0 = off; the range above ~8 is debug headroom for making the severing obvious, pair with Carve Residual 0)",
            },
        },
        noise_shaping_scale: f32 = Style {
            get: |e| e.noise.shaping_scale,
            set: |e, v| e.noise.shaping_scale = v,
        },
        optical_depth: f32 = Style {
            get: |e| e.optical_depth,
            set: |e, v| e.optical_depth = v,
            ui {
                                min: 0.0,
                max: 16.0,
                format: "%.2f",
                tooltip: "Line-of-sight optical thickness tau0 = sigma_t * radius: > 0 derives sigma_t as tau0 / radius so resizing the flame keeps its opacity (0 = use the raw sigma_t channel directly)",
            },
        },
        branch_period: f32 = Style {
            get: |e| e.branch.period,
            set: |e, v| e.branch.period = v,
            ui {
                                min: 0.0,
                max: 2.0,
                format: "%.2f",
                tooltip: "Spawn period [s] of the branch elements (vortex lines that roll the medium into side tongues); raised to life/31 when the table is full, so shorter periods never shrink the tongues; 0 = off",
            },
        },
        branch_life: f32 = Style { get: |e| e.branch.life, set: |e, v| e.branch.life = v,
            ui {
                                min: 0.1,
                max: 6.0,
                format: "%.2f",
                tooltip: "Lifetime [s] of one element (wind out fast, hold, burn out); the spawn period is raised to life/31 when the element table is full",
            },
        },
        branch_gain: f32 = Style { get: |e| e.branch.gain, set: |e, v| e.branch.gain = v,
            ui {
                                min: -8.0,
                max: 8.0,
                format: "%.2f",
                tooltip: "Rotation angle [rad] the core reaches at the end of winding; the medium flows along the arcs toward the tongue tip at a constant rate. Positive rolls trunk material down-out-up (cap curling inward), negative rolls it up-out-down (KH crest leaning upward). A compact rotation never folds; 0 = off",
            },
        },
        branch_core_radius: f32 = Style {
            get: |e| e.branch.core_radius,
            set: |e, v| e.branch.core_radius = v,
            ui {
                label: "Branch Core",
                min: 0.05,
                max: 3.0,
                format: "%.2f",
                tooltip: "Vortex core radius as a ratio of the local trunk radius: small values shear the medium into thin spirals, near 1 the whole disc turns together and tongues keep the trunk's thickness",
            },
        },
        branch_core_offset: f32 = Style {
            get: |e| e.branch.core_offset,
            set: |e, v| e.branch.core_offset = v,
            ui {
                                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Lateral position of the core at spawn as a ratio of the local trunk radius: 0 on the axis tilts the whole slab, 1 on the shear layer rolls trunk material outward as a billow",
            },
        },
        branch_reach: f32 = Style {
            get: |e| e.branch.reach,
            set: |e, v| e.branch.reach = v,
            ui {
                                min: 0.5,
                max: 8.0,
                format: "%.2f",
                tooltip: "Compact reach of one element at the end of its life as a ratio of the local trunk radius; nothing beyond it moves, so it bounds how far tongues can extend sideways",
            },
        },
        branch_spread: f32 = Style {
            get: |e| e.branch.spread,
            set: |e, v| e.branch.spread = v,
            ui {
                                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Scatter of azimuth, timing jitter, left/right alternation, element size, line tilt and window shift (0 = identical elements strictly alternating in one plane)",
            },
        },
        branch_spawn_height: f32 = Style {
            get: |e| e.branch.spawn_height,
            set: |e, v| e.branch.spawn_height = v,
            ui {
                label: "Branch Height",
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Center of the spawn height band (0 = base, 1 = top)",
            },
        },
        branch_spawn_range: f32 = Style {
            get: |e| e.branch.spawn_range,
            set: |e, v| e.branch.spawn_range = v,
            ui {
                label: "Branch Height Range",
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Full width of the spawn height band; 1.0 with center 0.5 spawns elements over the whole trunk",
            },
        },
        branch_seed: u32 = Frame { get: |e| e.branch.seed, set: |e, v| e.branch.seed = v },
        puff_gain: f32 = Style {
            get: |e| e.puff.gain,
            set: |e, v| e.puff.gain = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Thinning of the medium between the puffs in [0, 1]: the puff cores keep the full density while the gaps drop to 1 - gain, so the column reads as stacked lumps rising from the base; 0 = off",
            },
        },
        puff_period: f32 = Style {
            get: |e| e.puff.period,
            set: |e, v| e.puff.period = v,
            ui {
                min: 0.05,
                max: 3.0,
                format: "%.2f",
                tooltip: "Puffing period [s]: one density parcel leaves the base per period (puffing frequency 1 / period)",
            },
        },
        puff_rise: f32 = Style {
            get: |e| e.puff.rise,
            set: |e, v| e.puff.rise = v,
            ui {
                min: 0.01,
                max: 3.0,
                format: "%.2f",
                tooltip: "Rise velocity of the puffs in local height units per second; spacing between lumps = rise * period",
            },
        },
        puff_radius: f32 = Style {
            get: |e| e.puff.radius,
            set: |e, v| e.puff.radius = v,
            ui {
                min: 0.05,
                max: 2.0,
                format: "%.2f",
                tooltip: "Puff radius at spawn as a ratio of the base trunk radius",
            },
        },
        puff_spread: f32 = Style {
            get: |e| e.puff.spread,
            set: |e, v| e.puff.spread = v,
            ui {
                min: 0.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "Entrainment growth of the puff radius per unit height, in spawn radii",
            },
        },
        puff_decay: f32 = Style {
            get: |e| e.puff.decay,
            set: |e, v| e.puff.decay = v,
            ui {
                min: 0.0,
                max: 4.0,
                format: "%.2f",
                tooltip: "Height over which the puff density e-folds (burnout); 0 = no decay",
            },
        },
        puff_aspect: f32 = Style {
            get: |e| e.puff.aspect,
            set: |e, v| e.puff.aspect = v,
            ui {
                min: 0.1,
                max: 2.0,
                format: "%.2f",
                tooltip: "Vertical over lateral radius of a puff: below 1 flattens the lumps so a wide puff can still leave thin seams between neighbours",
            },
        },
        flow_gain: f32 = Style {
            get: |e| e.flow.gain,
            set: |e, v| e.flow.gain = v,
            ui {
                min: 0.0,
                max: 2.0,
                format: "%.2f",
                tooltip: "Scale of the fluid motion on the column centre and width (markers carried by the vortex-pair flow and the gust); 0 = off",
            },
        },
        flow_period: f32 = Style {
            get: |e| e.flow.period,
            set: |e, v| e.flow.period = v,
            ui {
                min: 0.1,
                max: 5.0,
                format: "%.2f",
                tooltip: "Vortex pair spawn period in seconds",
            },
        },
        flow_rise: f32 = Style {
            get: |e| e.flow.rise,
            set: |e, v| e.flow.rise = v,
            ui {
                min: 0.0,
                max: 2.0,
                format: "%.2f",
                tooltip: "Vortex pair rise speed in height units per second",
            },
        },
        flow_strength: f32 = Style {
            get: |e| e.flow.strength,
            set: |e, v| e.flow.strength = v,
            ui {
                min: 0.0,
                max: 5.0,
                format: "%.2f",
                tooltip: "Circulation of each vortex in base radii squared per second: how strongly a passing pair bulges and necks the column",
            },
        },
        flow_core: f32 = Style {
            get: |e| e.flow.core,
            set: |e, v| e.flow.core = v,
            ui {
                min: 0.1,
                max: 2.0,
                format: "%.2f",
                tooltip: "Gaussian core radius of a vortex in base radii: the lobe size",
            },
        },
        flow_gust: f32 = Style {
            get: |e| e.flow.gust,
            set: |e, v| e.flow.gust = v,
            ui {
                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Gust velocity amplitude at the tip in base radii per second: the whole-column sway",
            },
        },
        flow_gust_frequency: f32 = Style {
            get: |e| e.flow.gust_frequency,
            set: |e, v| e.flow.gust_frequency = v,
            ui {
                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Base gust frequency in Hz",
            },
        },
        flow_burst: f32 = Style {
            get: |e| e.flow.burst,
            set: |e, v| e.flow.burst = v,
            ui {
                min: 0.0,
                max: 5.0,
                format: "%.2f",
                tooltip: "Burst (whip) velocity amplitude in base radii per second, one burst every ten gust periods; 0 = none",
            },
        },
        flow_damping: f32 = Style {
            get: |e| e.flow.damping,
            set: |e, v| e.flow.damping = v,
            ui {
                min: 0.0,
                max: 5.0,
                format: "%.2f",
                tooltip: "Restoring rate of the markers toward the rest column per second: how long the column remembers the flow",
            },
        },
        flow_transport_speed: f32 = Style {
            get: |e| e.flow.transport_speed,
            set: |e, v| e.flow.transport_speed = v,
            ui {
                min: 0.0,
                max: 5.0,
                format: "%.2f",
                tooltip: "Upstream transport speed of the marker column in height units per second; 0 = off (bit-match)",
            },
        },
        flow_transport_accel: f32 = Style {
            get: |e| e.flow.transport_accel,
            set: |e, v| e.flow.transport_accel = v,
            ui {
                min: -5.0,
                max: 5.0,
                format: "%.2f",
                tooltip: "Transport speed increase with height (multiplied by y/aspect); 0 = uniform transport",
            },
        },
        lobe_gain: f32 = Style {
            get: |e| e.lobe.gain,
            set: |e, v| e.lobe.gain = v,
            ui {
                min: 0.0,
                max: 3.0,
                format: "%.2f",
                tooltip: "Lobe train: peak one-sided bulge of one lobe in base radii (needs flow_gain > 0); 0 = off",
            },
        },
        lobe_period: f32 = Style {
            get: |e| e.lobe.period,
            set: |e, v| e.lobe.period = v,
            ui {
                min: 0.01,
                max: 5.0,
                format: "%.3f",
                tooltip: "Lobe spawn period in seconds",
            },
        },
        lobe_life: f32 = Style {
            get: |e| e.lobe.life,
            set: |e, v| e.lobe.life = v,
            ui {
                min: 0.01,
                max: 10.0,
                format: "%.3f",
                tooltip: "Lobe lifetime in seconds: swells over the first half, fades over the second",
            },
        },
        lobe_rise: f32 = Style {
            get: |e| e.lobe.rise,
            set: |e, v| e.lobe.rise = v,
            ui {
                min: 0.0,
                max: 50.0,
                format: "%.2f",
                tooltip: "Lobe rise speed in height units per second",
            },
        },
        lobe_size: f32 = Style {
            get: |e| e.lobe.size,
            set: |e, v| e.lobe.size = v,
            ui {
                min: 0.01,
                max: 0.5,
                format: "%.3f",
                tooltip: "Vertical half-extent of one lobe in height units",
            },
        },
        lobe_spawn_height: f32 = Style {
            get: |e| e.lobe.spawn_height,
            set: |e, v| e.lobe.spawn_height = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Centre of the lobe spawn height band",
            },
        },
        lobe_spawn_range: f32 = Style {
            get: |e| e.lobe.spawn_range,
            set: |e, v| e.lobe.spawn_range = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Width of the uniform lobe spawn band above lobe_spawn_height; 0 = single band",
            },
        },
        lobe_accel: f32 = Style {
            get: |e| e.lobe.accel,
            set: |e, v| e.lobe.accel = v,
            ui {
                min: 0.0,
                max: 20.0,
                format: "%.2f",
                tooltip: "Exponential lobe rise rate in 1/s: higher lobes rise faster; 0 = constant rise",
            },
        },
        lobe_spread: f32 = Style {
            get: |e| e.lobe.spread,
            set: |e, v| e.lobe.spread = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Scatter of lobe spawn time, height and size",
            },
        },
        lobe_shift: f32 = Style {
            get: |e| e.lobe.shift,
            set: |e, v| e.lobe.shift = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Centre shift per unit bulge: 1 = one-sided tongue (far side still), 0 = symmetric puff",
            },
        },
    },
    runtime {
        time: f32 { get: |e| e.time, set: |e, v| e.time = v },
        warp_y_scale: f32 { get: |e| e.warp.y_scale, set: |e, v| e.warp.y_scale = v },
        emitter_kind: u32 { get: |e| e.emitter.kind, set: |e, v| e.emitter.kind = v },
        ring_major_radius: f32 {
            get: |e| e.emitter.ring_major_radius,
            set: |e, v| e.emitter.ring_major_radius = v,
        },
        ring_angular_speed: f32 {
            get: |e| e.emitter.ring_angular_speed,
            set: |e, v| e.emitter.ring_angular_speed = v,
        },
        boundary_amp: f32 { get: |e| e.boundary.amp, set: |e, v| e.boundary.amp = v },
        boundary_freq: f32 { get: |e| e.boundary.freq, set: |e, v| e.boundary.freq = v },
        boundary_speed: f32 { get: |e| e.boundary.speed, set: |e, v| e.boundary.speed = v },
        boundary_radius_ratio: f32 {
            get: |e| e.boundary.radius_ratio,
            set: |e, v| e.boundary.radius_ratio = v,
        },
        near_fade_radius: f32 {
            get: |e| e.carve.near_fade_radius,
            set: |e, v| e.carve.near_fade_radius = v,
        },
        carve_residual: f32 {
            get: |e| e.carve.residual,
            set: |e, v| e.carve.residual = v,
            ui {
                                min: 0.0,
                max: 0.5,
                format: "%.2f",
                tooltip: "Translucent floor left where the noise carves the medium away. 0 = fully carved spans become hard holes — the severing moment of Burnout is only visible near 0 (debug); the product look keeps ~0.12",
            },
        },
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_scene_core::find_scalar_param;

    #[test]
    fn test_missing_envelope_keys_take_legacy_scene_defaults() {
        let effect: FlameEffect = serde_json::from_str("{}").expect("all keys defaulted");
        assert_eq!(effect.envelope.peak, 0.35);
        assert_eq!(effect.envelope.base, 0.45);
        assert_eq!(effect.envelope.tail, 1.6);
        assert_eq!(effect.height, FlameEffect::default().height);
    }

    #[test]
    fn test_overwrite_persisted_fields_keeps_runtime_state() {
        let mut loaded = FlameEffect::default();
        loaded.height = 9.0;
        loaded.time = 5.0;
        loaded.emitter.kind = 1;

        let mut target = FlameEffect::default();
        target.time = 2.5;
        target.emitter.kind = 2;
        target.warp.y_scale = 0.9;

        overwrite_persisted_fields(&mut target, &loaded);
        assert_eq!(target.height, 9.0);
        assert_eq!(target.time, 2.5);
        assert_eq!(target.emitter.kind, 2);
        assert_eq!(target.warp.y_scale, 0.9);
    }

    #[test]
    fn test_ron_struct_syntax_roundtrip() {
        let mut effect = FlameEffect::default();
        effect.height = 3.25;
        effect.mix.scale = 0.5;

        let text = ron::ser::to_string_pretty(&effect, ron::ser::PrettyConfig::new())
            .expect("ron serialize");
        let restored: FlameEffect = ron::from_str(&text).expect("ron deserialize");
        assert_eq!(restored.height, 3.25);
        assert_eq!(restored.mix.scale, 0.5);
    }

    #[test]
    fn test_scalar_param_names_are_unique() {
        let mut names: Vec<&str> = FLAME_SCALAR_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    #[test]
    fn test_scalar_param_set_then_get_reaches_a_fixpoint() {
        for (i, param) in FLAME_SCALAR_PARAMS.iter().enumerate() {
            let mut effect = FlameEffect::default();
            (param.set)(&mut effect, 3.0 + i as f32);
            let first = (param.get)(&effect);
            (param.set)(&mut effect, first);
            assert_eq!((param.get)(&effect), first, "{}", param.name);
        }
    }

    #[test]
    fn test_bool_scalar_param_maps_zero_and_nonzero() {
        let param = find_scalar_param(FLAME_SCALAR_PARAMS, "use_blackbody").expect("registered");
        let mut effect = FlameEffect::default();
        (param.set)(&mut effect, 1.0);
        assert!(effect.color.use_blackbody);
        (param.set)(&mut effect, 0.0);
        assert!(!effect.color.use_blackbody);
    }

    #[test]
    fn test_wind_aliases_write_wind_direction_components() {
        let mut effect = FlameEffect::default();
        (find_scalar_param(FLAME_SCALAR_PARAMS, "wind_x")
            .expect("registered")
            .set)(&mut effect, 0.25);
        (find_scalar_param(FLAME_SCALAR_PARAMS, "wind_z")
            .expect("registered")
            .set)(&mut effect, -0.5);
        assert_eq!(effect.wind.direction.x, 0.25);
        assert_eq!(effect.wind.direction.y, -0.5);
    }

    #[test]
    fn test_every_ui_param_has_a_scalar_accessor_and_unique_name() {
        let mut names: Vec<&str> = FLAME_UI_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
        for param in FLAME_UI_PARAMS {
            assert!(
                find_scalar_param(FLAME_SCALAR_PARAMS, param.name).is_some(),
                "{}",
                param.name
            );
            assert!(param.min < param.max, "{}", param.name);
        }
    }

    #[test]
    fn test_runtime_params_are_not_serialized() {
        let value = serde_json::to_value(FlameEffect::default()).expect("serialize");
        let object = value.as_object().expect("flat object");
        for name in ["time", "warp_y_scale", "emitter_kind", "boundary_amp"] {
            assert!(!object.contains_key(name), "{name} must stay runtime-only");
        }
        assert_eq!(object.len(), PARAMETER_OWNERSHIP.len());
    }
}
