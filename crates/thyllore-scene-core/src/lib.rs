/// Flat-name accessor for one scalar parameter of a component: the single
/// entry format shared by scene persistence, batch CLI overrides and
/// animatable channel reads. Stateless and per component type, so every
/// entity holding the component shares one static table.
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

/// One declaration per parameter of a component, generating everything the
/// engine derives from it: the private scene serde record (plain struct shape,
/// so both RON and JSON stay readable — `#[serde(flatten)]` cannot, RON
/// rejects flattened maps), the `Serialize`/`Deserialize` impls of the
/// component, the tag table, the bit-exact snapshot fn, the overwrite fn that
/// keeps runtime state on load, and the flat-name scalar accessor registry.
///
/// `persisted` entries carry the flat scene key, type, tag, accessor pair and
/// (for keys that may be absent in old scenes with a value other than the
/// component default) the legacy on-disk default; f32 / u32 / bool entries are
/// auto-registered as scalars, vector entries may declare per-component
/// `scalars` aliases. `runtime` entries are never serialized and only join the
/// scalar registry. Invoke next to the component definition (orphan rule).
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
                $(,)?
            } ),+ $(,)?
        },
        runtime {
            $( $runtime_name:ident : $runtime_ty:tt {
                get: $runtime_get:expr,
                set: $runtime_set:expr $(,)?
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

        /// Persisted parameters (scene serde field names) mapped to their tag.
        pub const $tags_name: &[(&str, $tag_ty)] = &[
            $( (stringify!($name), <$tag_ty>::$tag) ),+
        ];

        /// Bit-exact value snapshot of every persisted parameter, keyed by the
        /// same field names as the tag table. Diffing two snapshots yields the
        /// exact set of parameters a writer touched.
        pub fn $snapshot_name(component: &$component) -> Vec<(&'static str, Vec<f32>)> {
            vec![ $( (stringify!($name), {
                let get: fn(&$component) -> $ty = $get;
                $crate::SnapshotValues::snapshot_values(&get(component))
            }) ),+ ]
        }

        /// Flat-name scalar accessors: persisted f32 / u32 / bool parameters,
        /// declared vector-component aliases, and runtime-only parameters.
        pub const $scalars_name: &[$crate::ScalarParam<$component>] =
            $crate::declare_scene_format!(@scalars $component, [
                $(
                    ($name, $ty, $get, $set)
                    $( $( ($alias, f32, $alias_get, $alias_set) )+ )?
                )+
                $( ($runtime_name, $runtime_ty, $runtime_get, $runtime_set) )*
            ], []);

        /// Write every persisted parameter of `loaded` onto `target`, keeping
        /// runtime state as it was; the persisted set is exactly the
        /// declaration table above.
        pub fn $overwrite_name(target: &mut $component, loaded: &$component) {
            $record::capture(loaded).apply(target);
        }
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
