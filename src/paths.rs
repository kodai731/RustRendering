// Forward-slash UNC: `\\wsl.localhost\...` is not resolved by std::fs.
pub const SHARED_DATA_DIR: &str = "//wsl.localhost/Ubuntu/home/kodai/Projects/SharedData";
pub const SHARED_EXPORTS_DIR: &str =
    "//wsl.localhost/Ubuntu/home/kodai/Projects/SharedData/exports";

pub const CURVE_COPILOT_LOCAL_MODEL: &str = "ml/model/curve_copilot.onnx";
pub const CURVE_COPILOT_SHARED_MODEL: &str =
    "//wsl.localhost/Ubuntu/home/kodai/Projects/SharedData/exports/curve_copilot.onnx";
pub const CURVE_COPILOT_DUMMY_MODEL: &str = "assets/ml/curve_copilot_dummy.onnx";
