use std::borrow::Cow;

/// Flat-name f32 accessor for one scalar parameter; one static table per component type.
pub struct ScalarParam<C: 'static> {
    pub name: &'static str,
    pub get: fn(&C) -> f32,
    pub set: fn(&mut C, f32),
}

pub fn find_scalar_param<'a, C>(
    params: &'a [ScalarParam<C>],
    name: &str,
) -> Option<&'a ScalarParam<C>> {
    params.iter().find(|param| param.name == name)
}

/// UI-toolkit-free display metadata of one scalar parameter, joined to the accessor table by `name`.
pub struct UiParam {
    pub name: &'static str,
    pub label: Option<&'static str>,
    pub min: f32,
    pub max: f32,
    pub format: &'static str,
    pub tooltip: &'static str,
}

impl UiParam {
    /// Explicit label, or the parameter name title-cased (`noise_amplitude` -> `Noise Amplitude`).
    pub fn display_label(&self) -> Cow<'static, str> {
        match self.label {
            Some(label) => Cow::Borrowed(label),
            None => Cow::Owned(title_case_snake(self.name)),
        }
    }
}

pub fn title_case_snake(name: &str) -> String {
    name.split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn find_ui_param<'a>(params: &'a [UiParam], name: &str) -> Option<&'a UiParam> {
    params.iter().find(|param| param.name == name)
}

pub trait SnapshotValues {
    fn snapshot_values(&self) -> Vec<f32>;
}

impl SnapshotValues for f32 {
    fn snapshot_values(&self) -> Vec<f32> {
        vec![*self]
    }
}

impl SnapshotValues for u32 {
    fn snapshot_values(&self) -> Vec<f32> {
        vec![*self as f32]
    }
}

impl SnapshotValues for bool {
    fn snapshot_values(&self) -> Vec<f32> {
        vec![u8::from(*self) as f32]
    }
}

impl<const N: usize> SnapshotValues for [f32; N] {
    fn snapshot_values(&self) -> Vec<f32> {
        self.to_vec()
    }
}

/// Generates a component's scene serde impls, tag table, snapshot, scalar/UI registries and
/// overwrite fn from one declaration table (RON rejects serde(flatten); invoke in the component's crate).
#[macro_export]
macro_rules! declare_scene_format {
    (
        component: $component:ty,
        record: $record:ident,
        tag: $tag_ty:ty,
        items {
            tags: $tags_name:ident,
            snapshot: $snapshot_name:ident,
            scalars: $scalars_name:ident,
            ui: $ui_name:ident,
            overwrite: $overwrite_name:ident $(,)?
        },
        persisted {
            $( $name:ident : $ty:tt = $tag:ident {
                get: $get:expr,
                set: $set:expr
                $(, default: $default:expr)?
                $(, scalars { $( $alias:ident : {
                    get: $alias_get:expr,
                    set: $alias_set:expr $(,)?
                } ),+ $(,)? })?
                $(, ui {
                    $( label: $ui_label:expr, )?
                    min: $ui_min:expr,
                    max: $ui_max:expr
                    $(, format: $ui_format:expr)?
                    $(, tooltip: $ui_tooltip:expr)? $(,)?
                })?
                $(,)?
            } ),+ $(,)?
        },
        runtime {
            $( $runtime_name:ident : $runtime_ty:tt {
                get: $runtime_get:expr,
                set: $runtime_set:expr
                $(, ui {
                    $( label: $rt_ui_label:expr, )?
                    min: $rt_ui_min:expr,
                    max: $rt_ui_max:expr
                    $(, format: $rt_ui_format:expr)?
                    $(, tooltip: $rt_ui_tooltip:expr)? $(,)?
                })?
                $(,)?
            } ),* $(,)?
        } $(,)?
    ) => {
        /// Persisted parameters (scene serde field names) mapped to their tag.
        pub const $tags_name: &[(&str, $tag_ty)] = &[
            $( (stringify!($name), <$tag_ty>::$tag) ),+
        ];

        $crate::declare_scene_format! {
            component: $component,
            record: $record,
            items {
                snapshot: $snapshot_name,
                scalars: $scalars_name,
                ui: $ui_name,
                overwrite: $overwrite_name,
            },
            persisted {
                $( $name : $ty {
                    get: $get,
                    set: $set
                    $(, default: $default)?
                    $(, scalars { $( $alias : {
                        get: $alias_get,
                        set: $alias_set,
                    } ),+ })?
                    $(, ui {
                        $( label: $ui_label, )?
                        min: $ui_min,
                        max: $ui_max
                        $(, format: $ui_format)?
                        $(, tooltip: $ui_tooltip)?
                    })?
                } ),+
            },
            runtime {
                $( $runtime_name : $runtime_ty {
                    get: $runtime_get,
                    set: $runtime_set
                    $(, ui {
                        $( label: $rt_ui_label, )?
                        min: $rt_ui_min,
                        max: $rt_ui_max
                        $(, format: $rt_ui_format)?
                        $(, tooltip: $rt_ui_tooltip)?
                    })?
                } ),*
            },
        }
    };
    (
        component: $component:ty,
        record: $record:ident,
        items {
            snapshot: $snapshot_name:ident,
            scalars: $scalars_name:ident,
            ui: $ui_name:ident,
            overwrite: $overwrite_name:ident $(,)?
        },
        persisted {
            $( $name:ident : $ty:tt {
                get: $get:expr,
                set: $set:expr
                $(, default: $default:expr)?
                $(, scalars { $( $alias:ident : {
                    get: $alias_get:expr,
                    set: $alias_set:expr $(,)?
                } ),+ $(,)? })?
                $(, ui {
                    $( label: $ui_label:expr, )?
                    min: $ui_min:expr,
                    max: $ui_max:expr
                    $(, format: $ui_format:expr)?
                    $(, tooltip: $ui_tooltip:expr)? $(,)?
                })?
                $(,)?
            } ),+ $(,)?
        },
        runtime {
            $( $runtime_name:ident : $runtime_ty:tt {
                get: $runtime_get:expr,
                set: $runtime_set:expr
                $(, ui {
                    $( label: $rt_ui_label:expr, )?
                    min: $rt_ui_min:expr,
                    max: $rt_ui_max:expr
                    $(, format: $rt_ui_format:expr)?
                    $(, tooltip: $rt_ui_tooltip:expr)? $(,)?
                })?
                $(,)?
            } ),* $(,)?
        } $(,)?
    ) => {
        #[derive(::serde::Serialize, ::serde::Deserialize)]
        #[serde(default)]
        struct $record {
            $( $name: $ty, )+
        }

        impl Default for $record {
            fn default() -> Self {
                let component = <$component as Default>::default();
                Self {
                    $( $name: $crate::declare_scene_format!(
                        @default component, $component, $ty, $get $(, $default)?
                    ), )+
                }
            }
        }

        impl $record {
            fn capture(component: &$component) -> Self {
                Self {
                    $( $name: {
                        let get: fn(&$component) -> $ty = $get;
                        get(component)
                    }, )+
                }
            }

            fn apply(self, component: &mut $component) {
                $( {
                    let set: fn(&mut $component, $ty) = $set;
                    set(component, self.$name);
                } )+
            }
        }

        impl ::serde::Serialize for $component {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                $record::capture(self).serialize(serializer)
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $component {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                let record = $record::deserialize(deserializer)?;
                let mut component = <$component as Default>::default();
                record.apply(&mut component);
                Ok(component)
            }
        }

        /// Bit-exact snapshot of every persisted parameter; diffing two yields what a writer touched.
        pub fn $snapshot_name(component: &$component) -> Vec<(&'static str, Vec<f32>)> {
            vec![ $( (stringify!($name), {
                let get: fn(&$component) -> $ty = $get;
                $crate::SnapshotValues::snapshot_values(&get(component))
            }) ),+ ]
        }

        /// Scalar accessors: persisted f32/u32/bool, vector-component aliases, runtime-only keys.
        pub const $scalars_name: &[$crate::ScalarParam<$component>] =
            $crate::declare_scene_format!(@scalars $component, [
                $(
                    ($name, $ty, $get, $set)
                    $( $( ($alias, f32, $alias_get, $alias_set) )+ )?
                )+
                $( ($runtime_name, $runtime_ty, $runtime_get, $runtime_set) )*
            ], []);

        /// Display metadata of the parameters that declared a `ui` node, in declaration order.
        pub const $ui_name: &[$crate::UiParam] = &[
            $( $(
                $crate::UiParam {
                    name: stringify!($name),
                    label: $crate::declare_scene_format!(@ui_label $(, $ui_label)?),
                    min: $ui_min,
                    max: $ui_max,
                    format: $crate::declare_scene_format!(@ui_or_default "%.3f" $(, $ui_format)?),
                    tooltip: $crate::declare_scene_format!(@ui_or_default "" $(, $ui_tooltip)?),
                },
            )? )+
            $( $(
                $crate::UiParam {
                    name: stringify!($runtime_name),
                    label: $crate::declare_scene_format!(@ui_label $(, $rt_ui_label)?),
                    min: $rt_ui_min,
                    max: $rt_ui_max,
                    format: $crate::declare_scene_format!(@ui_or_default "%.3f" $(, $rt_ui_format)?),
                    tooltip: $crate::declare_scene_format!(@ui_or_default "" $(, $rt_ui_tooltip)?),
                },
            )? )*
        ];

        /// Writes every persisted parameter of `loaded` onto `target`, keeping runtime state.
        pub fn $overwrite_name(target: &mut $component, loaded: &$component) {
            $record::capture(loaded).apply(target);
        }
    };
    (@ui_label) => {
        None
    };
    (@ui_label, $label:expr) => {
        Some($label)
    };
    (@ui_or_default $default:expr) => {
        $default
    };
    (@ui_or_default $default:expr, $value:expr) => {
        $value
    };
    (@default $component_value:ident, $component:ty, $ty:tt, $get:expr) => {{
        let get: fn(&$component) -> $ty = $get;
        get(&$component_value)
    }};
    (@default $component_value:ident, $component:ty, $ty:tt, $get:expr, $default:expr) => {
        $default
    };
    (@scalars $component:ty, [], [ $($acc:tt)* ]) => {
        &[ $($acc)* ]
    };
    (@scalars $component:ty,
        [ ($name:ident, f32, $get:expr, $set:expr) $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        $crate::declare_scene_format!(@scalars $component, [ $($rest)* ], [ $($acc)*
            $crate::ScalarParam {
                name: stringify!($name),
                get: {
                    fn get_scalar(component: &$component) -> f32 {
                        let get: fn(&$component) -> f32 = $get;
                        get(component)
                    }
                    get_scalar
                },
                set: {
                    fn set_scalar(component: &mut $component, value: f32) {
                        let set: fn(&mut $component, f32) = $set;
                        set(component, value);
                    }
                    set_scalar
                },
            },
        ])
    };
    (@scalars $component:ty,
        [ ($name:ident, u32, $get:expr, $set:expr) $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        $crate::declare_scene_format!(@scalars $component, [ $($rest)* ], [ $($acc)*
            $crate::ScalarParam {
                name: stringify!($name),
                get: {
                    fn get_scalar(component: &$component) -> f32 {
                        let get: fn(&$component) -> u32 = $get;
                        get(component) as f32
                    }
                    get_scalar
                },
                set: {
                    fn set_scalar(component: &mut $component, value: f32) {
                        let set: fn(&mut $component, u32) = $set;
                        set(component, value as u32);
                    }
                    set_scalar
                },
            },
        ])
    };
    (@scalars $component:ty,
        [ ($name:ident, bool, $get:expr, $set:expr) $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        $crate::declare_scene_format!(@scalars $component, [ $($rest)* ], [ $($acc)*
            $crate::ScalarParam {
                name: stringify!($name),
                get: {
                    fn get_scalar(component: &$component) -> f32 {
                        let get: fn(&$component) -> bool = $get;
                        u8::from(get(component)) as f32
                    }
                    get_scalar
                },
                set: {
                    fn set_scalar(component: &mut $component, value: f32) {
                        let set: fn(&mut $component, bool) = $set;
                        set(component, value != 0.0);
                    }
                    set_scalar
                },
            },
        ])
    };
    (@scalars $component:ty,
        [ ($name:ident, $other:tt, $get:expr, $set:expr) $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        $crate::declare_scene_format!(@scalars $component, [ $($rest)* ], [ $($acc)* ])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_title_case_snake_capitalizes_each_word() {
        assert_eq!(title_case_snake("noise_amplitude"), "Noise Amplitude");
        assert_eq!(title_case_snake("height"), "Height");
        assert_eq!(title_case_snake("wien_c_k"), "Wien C K");
    }

    #[test]
    fn test_display_label_prefers_explicit_label() {
        let explicit = UiParam {
            name: "swirl_gain",
            label: Some("Swirl"),
            min: 0.0,
            max: 1.0,
            format: "",
            tooltip: "",
        };
        let derived = UiParam {
            label: None,
            ..explicit
        };
        assert_eq!(explicit.display_label(), "Swirl");
        assert_eq!(derived.display_label(), "Swirl Gain");
    }
}
