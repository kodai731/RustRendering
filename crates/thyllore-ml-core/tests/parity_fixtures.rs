use std::fs;
use std::path::PathBuf;

use thyllore_ml_core::copilot::context::flatten_context;
use thyllore_ml_core::copilot::query::generate_query_times;
use thyllore_ml_core::copilot::sampling::sample_curve_linear;
use thyllore_ml_core::copilot::window::sample_window;
use thyllore_model_core::Skeleton;

#[test]
fn generate_parity_fixtures() {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/python_parity/fixtures");
    fs::create_dir_all(&out_dir).expect("create fixture dir");

    write_skeleton_fixture(&out_dir);
    write_context_fixture(&out_dir);
    write_window_fixture(&out_dir);
    write_query_fixture(&out_dir);
    write_sampling_fixture(&out_dir);
}

fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

fn write_skeleton_fixture(out_dir: &std::path::Path) {
    let mut skel = Skeleton::new("parity");
    skel.add_bone("root", None);
    skel.add_bone("hip", Some(0));
    skel.add_bone("spine", Some(1));
    skel.add_bone("chest", Some(2));
    skel.add_bone("neck", Some(3));
    skel.add_bone("head", Some(4));
    skel.add_bone("shoulder.L", Some(3));
    skel.add_bone("upper_arm.L", Some(6));

    let topo = thyllore_ml_core::compute_bone_topology(&skel);
    let tokens = thyllore_ml_core::compute_bone_name_tokens(&skel);

    let mut json = String::from("{\n");

    let names: Vec<String> = skel
        .bones
        .iter()
        .map(|b| format!("\"{}\"", b.name))
        .collect();
    json.push_str(&format!("  \"bone_names\": [{}],\n", names.join(", ")));

    let parents: Vec<String> = skel
        .bones
        .iter()
        .map(|b| match b.parent_id {
            Some(p) => format!("{}", p),
            None => "-1".to_string(),
        })
        .collect();
    json.push_str(&format!(
        "  \"parent_indices\": [{}],\n",
        parents.join(", ")
    ));

    let topo_rows: Vec<String> = skel
        .bones
        .iter()
        .map(|b| {
            let f = topo.get(&b.id).cloned().unwrap_or_default();
            let bits: Vec<String> = f
                .to_vec()
                .iter()
                .map(|v| format!("{}", f32_bits(*v)))
                .collect();
            format!("[{}]", bits.join(", "))
        })
        .collect();
    json.push_str(&format!("  \"topology\": [{}],\n", topo_rows.join(", ")));

    let token_rows: Vec<String> = skel
        .bones
        .iter()
        .map(|b| {
            let t = tokens.get(&b.id).cloned().unwrap_or([0_i64; 32]);
            let s: Vec<String> = t.iter().map(|v| format!("{}", v)).collect();
            format!("[{}]", s.join(", "))
        })
        .collect();
    json.push_str(&format!("  \"tokens\": [{}]\n", token_rows.join(", ")));

    json.push_str("}\n");

    fs::write(out_dir.join("skeleton_fixture.json"), json).expect("write skeleton fixture");
}

fn write_context_fixture(out_dir: &std::path::Path) {
    let times = vec![0.5_f32, 1.0, 1.5, 2.0, 2.5];
    let values = vec![0.0_f32, 0.84147096, 0.9092974, 0.14112, -0.7568025];
    let in_dt = vec![-0.1_f32; 5];
    let in_dv = vec![0.05_f32; 5];
    let out_dt = vec![0.1_f32; 5];
    let out_dv = vec![-0.05_f32; 5];

    let max_keyframes = 8;
    let clip_duration = 4.0_f32;

    let ctx = flatten_context(
        &times,
        &values,
        &in_dt,
        &in_dv,
        &out_dt,
        &out_dv,
        max_keyframes,
        clip_duration,
    );

    let mut json = String::from("{\n");

    let kf_rows: Vec<String> = (0..times.len())
        .map(|i| {
            format!(
                "[{}, {}, {}, {}, {}, {}]",
                f32_bits(times[i]),
                f32_bits(values[i]),
                f32_bits(in_dt[i]),
                f32_bits(in_dv[i]),
                f32_bits(out_dt[i]),
                f32_bits(out_dv[i])
            )
        })
        .collect();
    json.push_str(&format!("  \"keyframes\": [{}],\n", kf_rows.join(", ")));
    json.push_str(&format!("  \"max_keyframes\": {},\n", max_keyframes));
    json.push_str(&format!(
        "  \"clip_duration\": {},\n",
        f32_bits(clip_duration)
    ));

    let flat_bits: Vec<String> = ctx
        .flat
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();
    json.push_str(&format!("  \"flat\": [{}],\n", flat_bits.join(", ")));
    json.push_str(&format!("  \"mean\": {},\n", f32_bits(ctx.curve_mean)));
    json.push_str(&format!("  \"std\": {}\n", f32_bits(ctx.curve_std)));
    json.push_str("}\n");

    fs::write(out_dir.join("context_fixture.json"), json).expect("write context fixture");
}

fn write_window_fixture(out_dir: &std::path::Path) {
    let times = vec![0.0_f32, 0.5, 1.0, 1.5, 2.0];
    let values = vec![0.0_f32, 0.5, 0.0, -0.5, 0.0];
    let t_start = 0.25_f32;
    let t_end = 1.75_f32;
    let curve_mean = 0.0_f32;
    let curve_std = 0.5_f32;

    let window = sample_window(&times, &values, t_start, t_end, curve_mean, curve_std);

    let mut json = String::from("{\n");
    let times_bits: Vec<String> = times.iter().map(|v| format!("{}", f32_bits(*v))).collect();
    let values_bits: Vec<String> = values.iter().map(|v| format!("{}", f32_bits(*v))).collect();
    json.push_str(&format!("  \"times\": [{}],\n", times_bits.join(", ")));
    json.push_str(&format!("  \"values\": [{}],\n", values_bits.join(", ")));
    json.push_str(&format!("  \"t_start\": {},\n", f32_bits(t_start)));
    json.push_str(&format!("  \"t_end\": {},\n", f32_bits(t_end)));
    json.push_str(&format!("  \"curve_mean\": {},\n", f32_bits(curve_mean)));
    json.push_str(&format!("  \"curve_std\": {},\n", f32_bits(curve_std)));

    let win_bits: Vec<String> = window.iter().map(|v| format!("{}", f32_bits(*v))).collect();
    json.push_str(&format!("  \"window\": [{}]\n", win_bits.join(", ")));
    json.push_str("}\n");

    fs::write(out_dir.join("window_fixture.json"), json).expect("write window fixture");
}

fn write_query_fixture(out_dir: &std::path::Path) {
    let times_with_future = vec![0.0_f32, 1.0, 2.0, 3.0];
    let times_no_future = vec![0.0_f32, 0.5];
    let clip_duration = 4.0_f32;
    let max_steps = 4_usize;

    let qt_with_future = generate_query_times(&times_with_future, 0.5, clip_duration, max_steps);
    let qt_no_future = generate_query_times(&times_no_future, 1.0, clip_duration, max_steps);

    let mut json = String::from("{\n");
    let twf_bits: Vec<String> = times_with_future
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();
    let tnf_bits: Vec<String> = times_no_future
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();
    json.push_str(&format!(
        "  \"times_with_future\": [{}],\n",
        twf_bits.join(", ")
    ));
    json.push_str(&format!(
        "  \"times_no_future\": [{}],\n",
        tnf_bits.join(", ")
    ));
    json.push_str(&format!(
        "  \"clip_duration\": {},\n",
        f32_bits(clip_duration)
    ));
    json.push_str(&format!("  \"max_steps\": {},\n", max_steps));
    json.push_str(&format!(
        "  \"current_time_with_future\": {},\n",
        f32_bits(0.5)
    ));
    json.push_str(&format!(
        "  \"current_time_no_future\": {},\n",
        f32_bits(1.0)
    ));

    let qwf_bits: Vec<String> = qt_with_future
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();
    let qnf_bits: Vec<String> = qt_no_future
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();
    json.push_str(&format!(
        "  \"query_times_with_future\": [{}],\n",
        qwf_bits.join(", ")
    ));
    json.push_str(&format!(
        "  \"query_times_no_future\": [{}]\n",
        qnf_bits.join(", ")
    ));
    json.push_str("}\n");

    fs::write(out_dir.join("query_fixture.json"), json).expect("write query fixture");
}

fn write_sampling_fixture(out_dir: &std::path::Path) {
    let times = vec![0.0_f32, 1.0, 2.0, 3.0];
    let values = vec![10.0_f32, 20.0, 15.0, 30.0];
    let queries = vec![0.5_f32, 1.5, 2.5, -1.0, 5.0];

    let samples: Vec<f32> = queries
        .iter()
        .map(|&t| sample_curve_linear(&times, &values, t))
        .collect();

    let mut json = String::from("{\n");
    let times_bits: Vec<String> = times.iter().map(|v| format!("{}", f32_bits(*v))).collect();
    let values_bits: Vec<String> = values.iter().map(|v| format!("{}", f32_bits(*v))).collect();
    let queries_bits: Vec<String> = queries
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();
    let samples_bits: Vec<String> = samples
        .iter()
        .map(|v| format!("{}", f32_bits(*v)))
        .collect();

    json.push_str(&format!("  \"times\": [{}],\n", times_bits.join(", ")));
    json.push_str(&format!("  \"values\": [{}],\n", values_bits.join(", ")));
    json.push_str(&format!("  \"queries\": [{}],\n", queries_bits.join(", ")));
    json.push_str(&format!("  \"samples\": [{}]\n", samples_bits.join(", ")));
    json.push_str("}\n");

    fs::write(out_dir.join("sampling_fixture.json"), json).expect("write sampling fixture");
}
