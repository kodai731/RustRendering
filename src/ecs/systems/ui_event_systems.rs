use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum DeferredAction {
    LoadModel {
        path: String,
    },
    TakeScreenshot,
    #[cfg(debug_assertions)]
    DebugShadowInfo,
    #[cfg(debug_assertions)]
    DebugBillboardDepth,
    DumpDebugInfo,
    DumpAnimationDebug,
    LoadClipFromFile {
        path: PathBuf,
    },
    SaveClipToFile {
        source_id: u64,
        path: PathBuf,
    },
    SaveSpringBoneBake {
        baked_id: u64,
        path: PathBuf,
    },
    #[cfg(feature = "auto-rig")]
    LoadModelFromMemory {
        glb_data: Vec<u8>,
    },
    LoadModelAdditive {
        path: String,
    },
    DeleteEntities {
        entities: Vec<u64>,
    },
}
