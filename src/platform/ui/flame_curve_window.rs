use imgui::{Condition, DrawListMut};

use crate::animation::Interpolation;
use crate::ecs::component::{FlameParam, FlameTrack};
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::FlameCurveWindowState;

const PALETTE: [[f32; 4]; 16] = [
    [1.0, 0.3, 0.3, 1.0],
    [0.3, 1.0, 0.3, 1.0],
    [0.3, 0.3, 1.0, 1.0],
    [1.0, 1.0, 0.3, 1.0],
    [1.0, 0.3, 1.0, 1.0],
    [0.3, 1.0, 1.0, 1.0],
    [1.0, 0.6, 0.3, 1.0],
    [0.6, 1.0, 0.3, 1.0],
    [0.3, 0.6, 1.0, 1.0],
    [1.0, 0.6, 0.6, 1.0],
    [0.6, 1.0, 0.6, 1.0],
    [0.6, 0.6, 1.0, 1.0],
    [1.0, 0.8, 0.4, 1.0],
    [0.8, 1.0, 0.4, 1.0],
    [0.4, 0.8, 1.0, 1.0],
    [1.0, 0.4, 0.8, 1.0],
];

fn param_color(param: &FlameParam) -> [f32; 4] {
    let idx = match param {
        FlameParam::Height => 0,
        FlameParam::Radius => 1,
        FlameParam::Intensity => 2,
        FlameParam::SigmaT => 3,
        FlameParam::TemperatureBaseK => 4,
        FlameParam::TemperatureTipK => 5,
        FlameParam::WarpAmp => 6,
        FlameParam::WarpFreq => 7,
        FlameParam::RiseSpeed => 8,
        FlameParam::NoiseAmplitude => 9,
        FlameParam::WhiteBoost => 10,
        FlameParam::BendAmount => 11,
        FlameParam::WindX => 12,
        FlameParam::WindZ => 13,
        FlameParam::EdgeLow => 14,
        FlameParam::EdgeHigh => 15,
    };
    PALETTE[idx]
}

fn channel_value_range(keys: &[crate::animation::Keyframe<f32>]) -> (f32, f32) {
    let min = keys.iter().map(|k| k.value).fold(f32::INFINITY, f32::min);
    let max = keys.iter().map(|k| k.value).fold(f32::NEG_INFINITY, f32::max);
    let pad = (max - min) * 0.1;
    if (max - min).abs() < 1e-6 {
        (min - 0.5, min + 0.5)
    } else {
        (min - pad, max + pad)
    }
}

pub fn build_flame_curve_window(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &FlameCurveWindowState,
    track: Option<&FlameTrack>,
    current_time: f32,
) {
    if !state.open {
        return;
    }

    let track = match track {
        Some(t) => t,
        None => return,
    };

    ui.window("Flame Curves")
        .size([520.0, 320.0], Condition::FirstUseEver)
        .build(|| {
            // (1) Left column: checkboxes
            ui.columns(2, "curve_cols", true);
            for channel in &track.channels {
                let param = channel.param;
                let is_hidden = state.hidden_params.contains(&param);
                let mut checked = !is_hidden;
                ui.checkbox(format!("{:?}", param), &mut checked);
                if ui.is_item_clicked() {
                    ui_events.send(UIEvent::ToggleFlameCurveParam { param });
                }
            }
            ui.next_column();

            // (2) Plot area
            let t_max = track
                .channels
                .iter()
                .flat_map(|c| c.keys.iter())
                .map(|k| k.time)
                .fold(0.0f32, f32::max)
                + 1.0;
            let t_max = t_max.max(2.0);

            let draw_list = ui.get_window_draw_list();
            let cursor_pos = ui.cursor_screen_pos();
            let plot_left = cursor_pos[0];
            let plot_top = cursor_pos[1];
            let avail = ui.content_region_avail();
            let checkbox_col_width = avail[0] * 0.3;
            let plot_width = (avail[0] - checkbox_col_width).max(100.0);
            let plot_height = avail[1].max(100.0);

            // Draw background
            draw_list.add_rect(
                [plot_left, plot_top],
                [plot_left + plot_width, plot_top + plot_height],
                [0.118, 0.118, 0.118, 1.0],
            )
            .filled(true)
            .build();

            // Draw curves for each visible channel
            for channel in &track.channels {
                if state.hidden_params.contains(&channel.param) {
                    continue;
                }
                let color = param_color(&channel.param);

                let (v_min, v_max) = channel_value_range(&channel.keys);

                fn time_to_x(time: f32, t_max: f32, plot_left: f32, plot_width: f32) -> f32 {
                    plot_left + time / t_max * plot_width
                }

                fn value_to_y(value: f32, v_min: f32, v_max: f32, plot_top: f32, plot_height: f32) -> f32 {
                    let ratio = (value - v_min) / (v_max - v_min);
                    plot_top + plot_height * (1.0 - ratio)
                }

                fn x_to_time(x: f32, t_max: f32, plot_left: f32, plot_width: f32) -> f32 {
                    (x - plot_left) / plot_width * t_max
                }

                fn y_to_value(y: f32, v_min: f32, v_max: f32, plot_top: f32, plot_height: f32) -> f32 {
                    let ratio = 1.0 - (y - plot_top) / plot_height;
                    v_min + ratio * (v_max - v_min)
                }

                // Draw polyline segments between consecutive keys
                let mut prev_x: Option<f32> = None;
                let mut prev_y: Option<f32> = None;

                for (i, key) in channel.keys.iter().enumerate() {
                    let kx = time_to_x(key.time, t_max, plot_left, plot_width);
                    let ky = value_to_y(key.value, v_min, v_max, plot_top, plot_height);

                    if let (Some(px), Some(py)) = (prev_x, prev_y) {
                        // Check interpolation of the previous key
                        let prev_key = &channel.keys[i - 1];
                        match prev_key.interpolation {
                            Interpolation::Step => {
                                // Horizontal then vertical: draw L-shape
                                draw_list.add_line([px, py], [kx, py], color).thickness(1.0).build();
                                draw_list.add_line([kx, py], [kx, ky], color).thickness(1.0).build();
                            }
                            Interpolation::Linear | Interpolation::CubicSpline => {
                                draw_list.add_line([px, py], [kx, ky], color).thickness(1.0).build();
                            }
                        }
                    }

                    prev_x = Some(kx);
                    prev_y = Some(ky);

                    // Draw filled circle for key
                    draw_list.add_circle([kx, ky], 4.0, color).filled(true).build();

                    // (3) Interaction per key: invisible button
                    let btn_size = [12.0, 12.0];
                    let btn_x = kx - btn_size[0] / 2.0;
                    let btn_y = ky - btn_size[1] / 2.0;

                    ui.set_cursor_screen_pos([btn_x, btn_y]);
                    let id = format!("fc_key##{}##{:?}", i, channel.param);
                    ui.invisible_button(&id, btn_size);

                    // Tooltip on hover
                    if ui.is_item_hovered() {
                        ui.get_window_draw_list().add_rect(
                            [btn_x, btn_y],
                            [btn_x + btn_size[0], btn_y + btn_size[1]],
                            [1.0, 1.0, 1.0, 1.0],
                        ).build();
                        ui.tooltip(|| {
                            ui.text(format!("t={:.2} v={:.3}", key.time, key.value));
                        });
                    }

                    // Right-click to delete
                    if ui.is_item_hovered() && ui.is_mouse_clicked(imgui::MouseButton::Right) {
                        ui_events.send(UIEvent::DeleteFlameKeyExact {
                            param: channel.param,
                            time: key.time,
                        });
                    }

                    // Drag to move: send MoveFlameKey continuously while dragging
                    if ui.is_item_active() {
                        let mouse_pos = ui.io().mouse_pos;
                        let new_time = x_to_time(mouse_pos[0], t_max, plot_left, plot_width);
                        let new_value = y_to_value(mouse_pos[1], v_min, v_max, plot_top, plot_height);
                        ui_events.send(UIEvent::MoveFlameKey {
                            param: channel.param,
                            old_time: key.time,
                            new_time,
                            new_value,
                        });
                    }
                }
            }

            // Draw playhead line
            let playhead_x = current_time / t_max * plot_width + plot_left;
            draw_list.add_line(
                [playhead_x, plot_top],
                [playhead_x, plot_top + plot_height],
                [1.0, 1.0, 1.0, 1.0],
            )
            .thickness(1.0)
            .build();

            // Reserve the invisible area for mouse interaction on the plot
            ui.set_cursor_screen_pos([plot_left, plot_top]);
            ui.invisible_button("flame_curve_plot", [plot_width, plot_height]);
        });
}
