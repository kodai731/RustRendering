//! `flame_trace_replay` — rebuild the analytic flame CPU trace from a wall
//! probe dump (effect source) and a flame trace dump (view source), without
//! the GUI. Grid, segment count and estimator come from the environment:
//! THYLLORE_FLAME_TRACE_COLS / _ROWS / _SEGMENTS / _INTEGRATOR.
//!
//! Usage:
//!   flame_trace_replay --dump <wall_probe.json> --view <flame_trace.json> --out <out.json>

use std::fs;
use std::process;

use thyllore_effect_core::{build_flame_ubo, WallProbeView};

use thyllore_render_debug::dump_effect::flame_from_dump;
use thyllore_render_debug::flame_field_trace::trace_flame_field;

struct Args {
    dump: String,
    view: String,
    out: String,
}

fn parse_args() -> Args {
    let mut dump: Option<String> = None;
    let mut view: Option<String> = None;
    let mut out: Option<String> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dump" => {
                i += 1;
                dump = Some(args[i].clone());
            }
            "--view" => {
                i += 1;
                view = Some(args[i].clone());
            }
            "--out" => {
                i += 1;
                out = Some(args[i].clone());
            }
            other => {
                eprintln!("unknown argument: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    let missing = |name: &str| -> String {
        eprintln!("--{} is required", name);
        process::exit(1);
    };
    Args {
        dump: dump.unwrap_or_else(|| missing("dump")),
        view: view.unwrap_or_else(|| missing("view")),
        out: out.unwrap_or_else(|| missing("out")),
    }
}

fn read_json(path: &str) -> serde_json::Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {}", path, e);
        process::exit(1);
    });
    serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("failed to parse {}: {}", path, e);
        process::exit(1);
    })
}

fn vec3_from(value: &serde_json::Value) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (slot, component) in out.iter_mut().zip(0..3) {
        *slot = value[component].as_f64().unwrap_or_else(|| {
            eprintln!("view field is not a 3-vector: {}", value);
            process::exit(1);
        }) as f32;
    }
    out
}

fn main() {
    let args = parse_args();

    let dump = read_json(&args.dump);
    let (effect, baked, temporal) = flame_from_dump(&dump["flames"][0]);

    let view_source = read_json(&args.view);
    let v = &view_source["view"];
    let view = WallProbeView {
        position: vec3_from(&v["position"]),
        forward: vec3_from(&v["forward"]),
        right: vec3_from(&v["right"]),
        up: vec3_from(&v["up"]),
        fov_y_radians: v["fov_y_radians"].as_f64().unwrap_or(0.0) as f32,
        viewport_size_px: [
            v["viewport_size_px"][0].as_f64().unwrap_or(0.0) as f32,
            v["viewport_size_px"][1].as_f64().unwrap_or(0.0) as f32,
        ],
    };
    if view.fov_y_radians <= 0.0 {
        eprintln!("view file has no usable view.fov_y_radians");
        process::exit(1);
    }

    let ubo = build_flame_ubo(&effect, &baked, &temporal);
    let trace = trace_flame_field(&ubo, &view);

    let text = serde_json::to_string(&trace).unwrap();
    fs::write(&args.out, &text).unwrap_or_else(|e| {
        eprintln!("failed to write {}: {}", args.out, e);
        process::exit(1);
    });
    println!(
        "replayed {} rays ({} segments, {} integrator) to {}",
        trace["rays"].as_array().map_or(0, |r| r.len()),
        trace["segments"],
        trace["integrator"],
        args.out
    );
}
