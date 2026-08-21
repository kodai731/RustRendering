use thyllore_scene_core::{find_scalar_param, find_ui_param, ScalarParam, UiParam};

pub fn draw_scalar_params<C>(
    ui: &imgui::Ui,
    names: &[&str],
    ui_params: &[UiParam],
    scalars: &[ScalarParam<C>],
    component: &mut C,
    mut after_item: impl FnMut(&imgui::Ui, &'static str, f32),
) {
    for name in names {
        let Some(meta) = find_ui_param(ui_params, name) else {
            debug_assert!(false, "no ui metadata for {name}");
            continue;
        };
        let Some(scalar) = find_scalar_param(scalars, name) else {
            debug_assert!(false, "no scalar accessor for {name}");
            continue;
        };

        let mut value = (scalar.get)(component);
        if ui
            .slider_config(meta.label, meta.min, meta.max)
            .display_format(meta.format)
            .build(&mut value)
        {
            (scalar.set)(component, value);
        }
        if !meta.tooltip.is_empty() && ui.is_item_hovered() {
            ui.tooltip_text(meta.tooltip);
        }
        after_item(ui, meta.name, value);
    }
}
