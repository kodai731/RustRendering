use imgui::ColorEditFlags;
use thyllore_effect_core::{
    absorption_to_transmitted_color, transmitted_color_to_absorption, ABSORPTION_REFERENCE_DISTANCE,
};
use thyllore_scene_core::{find_scalar_param, find_ui_param, ScalarParam, UiKind, UiParam};

/// Scalar keys touched by one widget, `(alias name, value)`; a colour yields its r, g, b aliases.
pub type EditedScalars<'a> = &'a [(&'static str, f32)];

/// Names are validated against both tables by unit tests; an unresolved name is skipped.
pub fn draw_params<C>(
    ui: &imgui::Ui,
    names: &[&str],
    ui_params: &[UiParam],
    scalars: &[ScalarParam<C>],
    component: &mut C,
    mut after_item: impl FnMut(&imgui::Ui, EditedScalars),
) {
    for name in names {
        let Some(meta) = find_ui_param(ui_params, name) else {
            continue;
        };
        match meta.kind {
            UiKind::Scalar => draw_scalar(ui, meta, scalars, component, &mut after_item),
            UiKind::Color => draw_color(
                ui,
                meta,
                scalars,
                component,
                ColorMapping::Identity,
                &mut after_item,
            ),
            UiKind::Absorption => draw_color(
                ui,
                meta,
                scalars,
                component,
                ColorMapping::Transmitted,
                &mut after_item,
            ),
        }
    }
}

fn draw_scalar<C>(
    ui: &imgui::Ui,
    meta: &UiParam,
    scalars: &[ScalarParam<C>],
    component: &mut C,
    after_item: &mut impl FnMut(&imgui::Ui, EditedScalars),
) {
    let Some(scalar) = find_scalar_param(scalars, meta.name) else {
        return;
    };

    let mut value = (scalar.get)(component);
    if ui
        .slider_config(meta.display_label(), meta.min, meta.max)
        .display_format(meta.format)
        .build(&mut value)
    {
        (scalar.set)(component, value);
    }
    show_tooltip(ui, meta.tooltip);
    after_item(ui, &[(scalar.name, value)]);
}

#[derive(Clone, Copy)]
enum ColorMapping {
    Identity,
    Transmitted,
}

impl ColorMapping {
    fn to_picker(self, stored: [f32; 3]) -> [f32; 3] {
        match self {
            ColorMapping::Identity => stored,
            ColorMapping::Transmitted => {
                absorption_to_transmitted_color(stored, ABSORPTION_REFERENCE_DISTANCE)
            }
        }
    }

    fn from_picker(self, picked: [f32; 3]) -> [f32; 3] {
        match self {
            ColorMapping::Identity => picked,
            ColorMapping::Transmitted => {
                transmitted_color_to_absorption(picked, ABSORPTION_REFERENCE_DISTANCE)
            }
        }
    }

    fn tooltip(self, base: &str) -> String {
        match self {
            ColorMapping::Identity => base.to_string(),
            ColorMapping::Transmitted => {
                format!("{base} (transmitted colour over {ABSORPTION_REFERENCE_DISTANCE} m)")
            }
        }
    }
}

fn draw_color<C>(
    ui: &imgui::Ui,
    meta: &UiParam,
    scalars: &[ScalarParam<C>],
    component: &mut C,
    mapping: ColorMapping,
    after_item: &mut impl FnMut(&imgui::Ui, EditedScalars),
) {
    let component_names = meta.color_component_names();
    let Some(channels) = resolve_channels(scalars, &component_names) else {
        return;
    };

    let stored = channels.map(|channel| (channel.get)(component));
    let mut picked = mapping.to_picker(stored);
    let changed = ui
        .color_edit3_config(meta.display_label(), &mut picked)
        .flags(ColorEditFlags::FLOAT | ColorEditFlags::NO_ALPHA)
        .build();
    show_tooltip(ui, &mapping.tooltip(meta.tooltip));

    let written = if changed {
        let clamped = mapping
            .from_picker(picked)
            .map(|value| value.clamp(meta.min, meta.max));
        for (channel, value) in channels.iter().zip(clamped) {
            (channel.set)(component, value);
        }
        clamped
    } else {
        stored
    };

    let edited: [(&'static str, f32); 3] = [
        (channels[0].name, written[0]),
        (channels[1].name, written[1]),
        (channels[2].name, written[2]),
    ];
    after_item(ui, &edited);
}

fn resolve_channels<'a, C>(
    scalars: &'a [ScalarParam<C>],
    names: &[String; 3],
) -> Option<[&'a ScalarParam<C>; 3]> {
    Some([
        find_scalar_param(scalars, &names[0])?,
        find_scalar_param(scalars, &names[1])?,
        find_scalar_param(scalars, &names[2])?,
    ])
}

fn show_tooltip(ui: &imgui::Ui, tooltip: &str) {
    if !tooltip.is_empty() && ui.is_item_hovered() {
        ui.tooltip_text(tooltip);
    }
}
