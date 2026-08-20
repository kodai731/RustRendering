pub fn binding_const_name(glsl_identifier: &str) -> String {
    let characters: Vec<char> = glsl_identifier.chars().collect();
    let mut name = String::new();
    for (index, &current) in characters.iter().enumerate() {
        if current.is_ascii_uppercase() && starts_word(&characters, index) {
            name.push('_');
        }
        name.push(current.to_ascii_uppercase());
    }

    if name
        .chars()
        .next()
        .is_some_and(|first| first.is_ascii_digit())
    {
        name.insert(0, '_');
    }
    name
}

fn starts_word(characters: &[char], index: usize) -> bool {
    let Some(previous) = index.checked_sub(1).map(|prev| characters[prev]) else {
        return false;
    };
    let follows_lowercase = previous.is_ascii_lowercase();
    let ends_acronym = previous.is_ascii_uppercase()
        && characters
            .get(index + 1)
            .is_some_and(|next| next.is_ascii_lowercase());
    follows_lowercase || ends_acronym
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_camel_case_to_screaming_snake_case() {
        assert_eq!(binding_const_name("historySampler"), "HISTORY_SAMPLER");
        assert_eq!(binding_const_name("ubo"), "UBO");
        assert_eq!(binding_const_name("FlameUBO"), "FLAME_UBO");
        assert_eq!(binding_const_name("sceneDepth2D"), "SCENE_DEPTH2D");
        assert_eq!(binding_const_name("already_snake"), "ALREADY_SNAKE");
        assert_eq!(binding_const_name("objectIDSampler"), "OBJECT_ID_SAMPLER");
        assert_eq!(binding_const_name("topLevelAS"), "TOP_LEVEL_AS");
    }

    #[test]
    fn prefixes_leading_digit() {
        assert_eq!(binding_const_name("2dSampler"), "_2D_SAMPLER");
    }
}
