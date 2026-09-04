use anyhow::{bail, Result};

use super::{
    BATCH_WATER_CAUSTIC_DEBUG_FLAG, BATCH_WATER_DEBUG_VIEW_FLAG, BATCH_WATER_HISTORY_FLAG,
    BATCH_WATER_SECONDARY_FLAG, BATCH_WATER_TIME_FLAG,
};

pub fn water_debug_view_resolve_from_args(args: &[String]) -> Result<Option<i32>> {
    let Some(position) = args
        .iter()
        .position(|arg| arg == BATCH_WATER_DEBUG_VIEW_FLAG)
    else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_WATER_DEBUG_VIEW_FLAG} requires a value (integer debug view index)");
    };
    let view: i32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid water debug view '{value}': expected integer"))?;
    Ok(Some(view))
}

pub fn water_caustic_debug_resolve_from_args(args: &[String]) -> Result<Option<i32>> {
    let Some(position) = args
        .iter()
        .position(|arg| arg == BATCH_WATER_CAUSTIC_DEBUG_FLAG)
    else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_WATER_CAUSTIC_DEBUG_FLAG} requires a value (integer caustic debug mode)");
    };
    let mode: i32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid water caustic debug '{value}': expected integer"))?;
    Ok(Some(mode))
}

pub fn water_secondary_resolve_from_args(
    args: &[String],
) -> Result<Option<thyllore_effect_core::WaterSecondaryRays>> {
    let Some(position) = args
        .iter()
        .position(|arg| arg == BATCH_WATER_SECONDARY_FLAG)
    else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_WATER_SECONDARY_FLAG} requires a value (rayquery|screenspace|raytracing)");
    };
    let secondary = thyllore_effect_core::WaterSecondaryRays::parse(value).ok_or_else(|| {
        anyhow::anyhow!(
            "{BATCH_WATER_SECONDARY_FLAG} requires a value (rayquery|screenspace|raytracing)"
        )
    })?;
    Ok(Some(secondary))
}

pub fn water_history_weight_resolve_from_args(args: &[String]) -> Result<Option<f32>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_WATER_HISTORY_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_WATER_HISTORY_FLAG} requires a value (history blend weight)");
    };
    let weight: f32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid water history weight '{value}': expected float"))?;
    Ok(Some(weight))
}

pub fn water_fixed_time_resolve_from_args(args: &[String]) -> Result<Option<f32>> {
    let Some(position) = args.iter().position(|arg| arg == BATCH_WATER_TIME_FLAG) else {
        return Ok(None);
    };
    let Some(value) = args.get(position + 1) else {
        bail!("{BATCH_WATER_TIME_FLAG} requires a value (seconds)");
    };
    let seconds: f32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid water time '{value}': expected float seconds"))?;
    Ok(Some(seconds))
}
