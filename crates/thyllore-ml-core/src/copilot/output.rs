use anyhow::Result;
use ort::session::SessionOutputs;

use super::CopilotStepPrediction;

pub const COPILOT_VALUES_PER_STEP: usize = 5;

pub fn parse_curve_predict_output(outputs: &SessionOutputs) -> Result<Vec<f32>> {
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

pub fn parse_curve_copilot_output(
    outputs: &SessionOutputs,
    num_steps: usize,
) -> Result<Vec<CopilotStepPrediction>> {
    let (_pred_shape, prediction_data) = outputs[0].try_extract_tensor::<f32>()?;
    let pred: Vec<f32> = prediction_data.to_vec();

    let (_conf_shape, confidence_data) = outputs[1].try_extract_tensor::<f32>()?;
    let conf: Vec<f32> = confidence_data.to_vec();

    let mut steps = Vec::with_capacity(num_steps);

    for i in 0..num_steps {
        let offset = i * COPILOT_VALUES_PER_STEP;
        if offset + COPILOT_VALUES_PER_STEP > pred.len() {
            break;
        }
        let value = pred[offset];
        let tangent_in = (pred[offset + 1], pred[offset + 2]);
        let tangent_out = (pred[offset + 3], pred[offset + 4]);
        let confidence = conf.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);

        steps.push(CopilotStepPrediction {
            value,
            tangent_in,
            tangent_out,
            confidence,
        });
    }

    Ok(steps)
}
