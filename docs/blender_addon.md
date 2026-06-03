# Thyllore Animation — Blender Addon

The Blender addon shares Thyllore Animation's core ML logic (the Rust `thyllore_ml_core`
wheel) so the addon and the desktop engine run the *identical* ONNX model and numeric
pipeline. This page documents the **Curve Copilot** feature.

## Curve Copilot

Curve Copilot is an AI-driven *forecast* for animation FCurves. While you animate, it
predicts how the selected channels will continue and draws the prediction as a
non-destructive **ghost curve** in the Graph Editor. It is a preview only — it never
inserts keyframes or edits the real FCurve.

![Curve Copilot](images/curve_copilot.png)

In the screenshot the dashed coloured polylines in the Graph Editor (bottom) are the
forecasted ghost curves, one per selected channel, extending forward from the playhead.
The real keyframes remain untouched.

### How it works

1. Select one or more curves in the Graph Editor (each needs at least 2 keyframes).
2. Place the playhead **on or after** a keyframe of those curves.
3. Run **Preview Forecast** (`Shift+C`).

The addon extracts samples from the selected FCurve at the model's deploy rate
(`scene_fps / deploy_fps`, so the input matches the engine and the 60 fps training set
regardless of scene fps), hands them to the `thyllore_ml_core` wheel for inference, and
renders the returned polyline as a GPU ghost overlay. All numeric work — window offsets,
origin resolution, ONNX inference, continuity, and the ghost polyline — lives in the Rust
wheel as the single source of truth shared with the engine.

`rotation_quaternion` channels are forecast in Euler space (the trained representation):
the bone's full quaternion is converted to a continuous Euler curve, each axis is
forecast, then the result is converted back to a quaternion. Location, scale, and Euler
rotation channels are forecast directly.

### Toggle behaviour

`Shift+C` toggles the preview: press once to draw the forecast, press again to clear it.
This lets you quickly drop an unwanted prediction and redraw it from a new channel
selection or playhead position. **Clear Preview** removes the ghost explicitly.

### UI

The operators are exposed in the **Curve Copilot** panel (3D Viewport → N-panel →
Thyllore category), available when an armature is the active object:

| Control | Action |
|---|---|
| Preview Forecast (`Shift+C`) | Draw / clear the ghost forecast for the selected channels |
| Clear Preview | Remove the ghost curve |

### Preferences

Configure Curve Copilot under **Edit → Preferences → Add-ons → Thyllore Animation**:

| Setting | Description |
|---|---|
| Enable Curve Copilot | Turns the feature on/off |
| Curve Copilot Model Path | Path to `curve_copilot.onnx`. Leave empty to use the model the engine resolves from SharedData, falling back to the model bundled in the addon ZIP |

### Requirements

- The `thyllore_ml_core` wheel must be loaded (it provides the `curve_forecast`
  capability). If the wheel is absent, the operator is unavailable.
- An armature with an action containing FCurves.
