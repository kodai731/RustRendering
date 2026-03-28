use std::process::{Child, Command};

const WSL_SERVER_DIR: &str = "/home/kodai/Projects/AnimationModelTraining";
const CONFIG_PATH: &str = "configs/server.yaml";

pub struct MeshServerProcess {
    child: Child,
}

impl MeshServerProcess {
    pub fn launch() -> Result<Self, String> {
        let is_debug = cfg!(debug_assertions);
        update_server_config_debug(is_debug)?;

        let child = Command::new("wsl")
            .args([
                "-d",
                "Ubuntu",
                "--cd",
                WSL_SERVER_DIR,
                "bash",
                "-lc",
                "python -m anim_ml.server.service --config configs/server.yaml",
            ])
            .spawn()
            .map_err(|e| format!("Failed to launch server via WSL: {}", e))?;

        Ok(Self { child })
    }

    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for MeshServerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

fn update_server_config_debug(debug: bool) -> Result<(), String> {
    let config_path = format!("//wsl.localhost/Ubuntu{}/{}", WSL_SERVER_DIR, CONFIG_PATH);

    let content = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read {}: {}", config_path, e))?;

    let updated = set_yaml_debug_field(&content, debug);

    std::fs::write(&config_path, &updated)
        .map_err(|e| format!("Failed to write {}: {}", config_path, e))?;

    Ok(())
}

fn set_yaml_debug_field(content: &str, debug: bool) -> String {
    let debug_value = if debug { "true" } else { "false" };
    let mut lines: Vec<String> = content.lines().map(String::from).collect();
    let mut found = false;

    for line in &mut lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with("debug:") {
            let indent = &line[..line.len() - trimmed.len()];
            *line = format!("{}debug: {}", indent, debug_value);
            found = true;
            break;
        }
    }

    if !found {
        let mesh_idx = lines
            .iter()
            .position(|l| l.trim_start().starts_with("mesh:"));
        if let Some(idx) = mesh_idx {
            let indent = find_mesh_section_indent(&lines, idx);
            lines.insert(idx + 1, format!("{}debug: {}", indent, debug_value));
        }
    }

    let mut result = lines.join("\n");
    if content.ends_with('\n') && !result.ends_with('\n') {
        result.push('\n');
    }
    result
}

fn find_mesh_section_indent(lines: &[String], mesh_idx: usize) -> String {
    for line in &lines[mesh_idx + 1..] {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.starts_with('#') {
            let indent_len = line.len() - trimmed.len();
            return line[..indent_len].to_string();
        }
    }
    "  ".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_debug_true_when_exists() {
        let input = "mesh:\n  model_name: trellis\n  debug: false\n";
        let result = set_yaml_debug_field(input, true);
        assert_eq!(result, "mesh:\n  model_name: trellis\n  debug: true\n");
    }

    #[test]
    fn test_set_debug_false_when_exists() {
        let input = "mesh:\n  debug: true\n  device: cuda\n";
        let result = set_yaml_debug_field(input, false);
        assert_eq!(result, "mesh:\n  debug: false\n  device: cuda\n");
    }

    #[test]
    fn test_insert_debug_when_missing() {
        let input = "mesh:\n  model_name: trellis\n  device: cuda\n";
        let result = set_yaml_debug_field(input, true);
        assert_eq!(
            result,
            "mesh:\n  debug: true\n  model_name: trellis\n  device: cuda\n"
        );
    }

    #[test]
    fn test_preserves_trailing_newline() {
        let input = "mesh:\n  debug: false\n";
        let result = set_yaml_debug_field(input, true);
        assert!(result.ends_with('\n'));
    }
}
