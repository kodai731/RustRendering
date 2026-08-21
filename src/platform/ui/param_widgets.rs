use thyllore_scene_core::{find_scalar_param, find_ui_param, ScalarParam, UiParam};

/// Names are validated against both tables by unit tests; an unresolved name is skipped.
pub fn draw_scalar_params<C>(
    ui: &imgui::Ui,
    names: &[&str],
    ui_params: &[UiParam],
    scalars: &[ScalarParam<C>],
    component: &mut C,
    mut after_item: impl FnMut(&imgui::Ui, &'static str, f32),
) {
    for name in names {
        let (Some(meta), Some(scalar)) = (
            find_ui_param(ui_params, name),
            find_scalar_param(scalars, name),
        ) else {
            continue;
        };

        let mut value = (scalar.get)(component);
        if ui
            .slider_config(meta.display_label(), meta.min, meta.max)
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
