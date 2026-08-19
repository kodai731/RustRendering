---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'query: スピードを上げて'
- text: 'query: what frame are we on'
- text: 'query: セーブして'
- text: 'query: 標準のカメラ位置に戻す'
- text: 'query: zoom out to see the whole character'
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 35 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                              | Examples                                                                                                                                                                    |
|:-----------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| list_objects                       | <ul><li>"query: what's in this scene"</li><li>'query: give me a list of all objects'</li><li>'query: enumerate the scene contents'</li></ul>                                |
| describe_selection                 | <ul><li>'query: what did I select'</li><li>'query: give me details on the selection'</li><li>'query: explain the selected object'</li></ul>                                 |
| get_playback_state                 | <ul><li>'query: how fast is it playing'</li><li>'query: tell me the playback status'</li><li>'query: are we paused'</li></ul>                                               |
| take_screenshot                    | <ul><li>'query: screenshot please'</li><li>'query: save a picture of the viewport'</li><li>'query: snap the current frame'</li></ul>                                        |
| play_animation                     | <ul><li>'query: begin the animation'</li><li>'query: hit play'</li><li>'query: let the animation roll'</li></ul>                                                            |
| pause_animation                    | <ul><li>'query: pause it'</li><li>'query: freeze the playback here'</li><li>'query: hold the current frame'</li></ul>                                                       |
| stop_animation                     | <ul><li>'query: stop and rewind'</li><li>'query: end playback and reset to frame zero'</li><li>'query: abort the animation and return to start'</li></ul>                   |
| set_playback_speed:slow            | <ul><li>'query: slow the playback down'</li><li>'query: reduce the play speed'</li><li>'query: playback at slow speed'</li></ul>                                            |
| set_playback_speed:normal          | <ul><li>'query: set the speed back to normal'</li><li>'query: default playback speed please'</li><li>'query: restore standard play speed'</li></ul>                         |
| set_playback_speed:fast            | <ul><li>'query: speed up the playback'</li><li>'query: play it faster'</li><li>'query: increase the play speed'</li></ul>                                                   |
| seek_time:start                    | <ul><li>'query: go to frame zero'</li><li>'query: move the playhead to the beginning'</li><li>'query: rewind to the first frame'</li></ul>                                  |
| seek_time:end                      | <ul><li>'query: seek to the last frame'</li><li>'query: move the playhead to the end'</li><li>'query: jump to the finish of the timeline'</li></ul>                         |
| seek_time:next_key                 | <ul><li>'query: advance to the following keyframe'</li><li>'query: step forward to the next key'</li><li>'query: jump ahead one keyframe'</li></ul>                         |
| seek_time:prev_key                 | <ul><li>'query: step back to the previous key'</li><li>'query: jump to the earlier keyframe'</li><li>'query: seek to the prior key'</li></ul>                               |
| toggle_loop                        | <ul><li>'query: enable loop playback'</li><li>'query: switch looping off'</li><li>'query: flip the loop setting'</li></ul>                                                  |
| select_object                      | <ul><li>'query: select the sphere'</li><li>'query: choose the object called Robot'</li><li>'query: pick Light01'</li></ul>                                                  |
| set_object_visibility:show         | <ul><li>'query: make the sphere visible again'</li><li>'query: unhide the wall'</li><li>'query: show Robot'</li></ul>                                                       |
| set_object_visibility:hide         | <ul><li>'query: hide the wall'</li><li>'query: make the sphere invisible'</li><li>'query: turn off visibility for Light01'</li></ul>                                        |
| focus_camera:selection             | <ul><li>'query: zoom to what I selected'</li><li>'query: center the view on the selected object'</li><li>'query: look at the selection'</li></ul>                           |
| focus_camera:model                 | <ul><li>'query: fit the entire model on screen'</li><li>'query: zoom out to see the whole character'</li><li>'query: frame everything'</li></ul>                            |
| focus_camera:reset                 | <ul><li>'query: put the camera back to default'</li><li>'query: restore the original view'</li><li>'query: reset the viewport camera'</li></ul>                             |
| undo                               | <ul><li>'query: take that back'</li><li>'query: cancel the last operation'</li><li>'query: roll back what I just did'</li></ul>                                             |
| redo                               | <ul><li>'query: reapply what I undid'</li><li>'query: bring back the change I reverted'</li><li>'query: restore the undone change'</li></ul>                                |
| save_scene                         | <ul><li>'query: write the scene to disk'</li><li>'query: store my progress'</li><li>'query: please save'</li></ul>                                                          |
| generate_motion:walk               | <ul><li>'query: create a walking animation'</li><li>'query: make the character walk'</li><li>'query: generate a walk cycle'</li></ul>                                       |
| generate_motion:run                | <ul><li>'query: generate a running animation'</li><li>'query: make the character run'</li><li>'query: create a run cycle'</li></ul>                                         |
| generate_motion:idle               | <ul><li>'query: make an idle pose animation'</li><li>'query: generate a standing idle'</li><li>'query: create an idle loop'</li></ul>                                       |
| generate_motion:jump               | <ul><li>'query: create a jump animation'</li><li>'query: make the character jump'</li><li>'query: generate a jumping motion'</li></ul>                                      |
| generate_motion:turn               | <ul><li>'query: generate a turning animation'</li><li>'query: make the character turn around'</li><li>'query: create a turn motion'</li></ul>                               |
| camera_shot:look_at_selection      | <ul><li>'query: Keep the camera tracking the selection'</li><li>'query: Follow the selected object with the camera'</li><li>'query: Make the camera follow along'</li></ul> |
| camera_shot:orbit_around_selection | <ul><li>'query: Orbit the camera around the selection'</li><li>"query: Circle around what's selected"</li><li>'query: Rotate the view around it'</li></ul>                  |
| camera_shot:dolly_in               | <ul><li>'query: Push the camera in closer'</li><li>'query: Dolly in'</li><li>'query: Move the camera closer'</li></ul>                                                      |
| camera_shot:dolly_out              | <ul><li>'query: Pull the camera back'</li><li>'query: Dolly out'</li><li>'query: Move the camera away'</li></ul>                                                            |
| camera_shot:crane_up               | <ul><li>'query: Raise the camera up'</li><li>'query: Crane up'</li><li>'query: Lift the camera higher'</li></ul>                                                            |
| camera_shot:crane_down             | <ul><li>'query: Lower the camera'</li><li>'query: Crane down'</li><li>'query: Bring the camera down'</li></ul>                                                              |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("query: セーブして")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 2   | 3.1545 | 10  |

| Label                              | Training Sample Count |
|:-----------------------------------|:----------------------|
| camera_shot:crane_down             | 12                    |
| camera_shot:crane_up               | 12                    |
| camera_shot:dolly_in               | 12                    |
| camera_shot:dolly_out              | 12                    |
| camera_shot:look_at_selection      | 12                    |
| camera_shot:orbit_around_selection | 12                    |
| describe_selection                 | 40                    |
| focus_camera:model                 | 40                    |
| focus_camera:reset                 | 35                    |
| focus_camera:selection             | 40                    |
| generate_motion:idle               | 38                    |
| generate_motion:jump               | 40                    |
| generate_motion:run                | 34                    |
| generate_motion:turn               | 38                    |
| generate_motion:walk               | 40                    |
| get_playback_state                 | 40                    |
| list_objects                       | 38                    |
| pause_animation                    | 40                    |
| play_animation                     | 40                    |
| redo                               | 24                    |
| save_scene                         | 40                    |
| seek_time:end                      | 40                    |
| seek_time:next_key                 | 40                    |
| seek_time:prev_key                 | 38                    |
| seek_time:start                    | 40                    |
| select_object                      | 37                    |
| set_object_visibility:hide         | 32                    |
| set_object_visibility:show         | 37                    |
| set_playback_speed:fast            | 40                    |
| set_playback_speed:normal          | 39                    |
| set_playback_speed:slow            | 40                    |
| stop_animation                     | 39                    |
| take_screenshot                    | 40                    |
| toggle_loop                        | 40                    |
| undo                               | 24                    |

### Training Hyperparameters
- batch_size: (32, 32)
- num_epochs: (3, 3)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 20260726
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0007 | 1    | 0.353         | -               |
| 0.0343 | 50   | 0.3326        | -               |
| 0.0686 | 100  | 0.2578        | -               |
| 0.1030 | 150  | 0.1768        | -               |
| 0.1373 | 200  | 0.1456        | -               |
| 0.1716 | 250  | 0.1207        | -               |
| 0.2059 | 300  | 0.0939        | -               |
| 0.2402 | 350  | 0.0882        | -               |
| 0.2745 | 400  | 0.0712        | -               |
| 0.3089 | 450  | 0.0711        | -               |
| 0.3432 | 500  | 0.0552        | -               |
| 0.3775 | 550  | 0.049         | -               |
| 0.4118 | 600  | 0.0499        | -               |
| 0.4461 | 650  | 0.042         | -               |
| 0.4804 | 700  | 0.0425        | -               |
| 0.5148 | 750  | 0.0426        | -               |
| 0.5491 | 800  | 0.0369        | -               |
| 0.5834 | 850  | 0.0356        | -               |
| 0.6177 | 900  | 0.0349        | -               |
| 0.6520 | 950  | 0.0364        | -               |
| 0.6863 | 1000 | 0.0354        | -               |
| 0.7207 | 1050 | 0.0254        | -               |
| 0.7550 | 1100 | 0.0279        | -               |
| 0.7893 | 1150 | 0.0273        | -               |
| 0.8236 | 1200 | 0.0222        | -               |
| 0.8579 | 1250 | 0.0243        | -               |
| 0.8922 | 1300 | 0.0257        | -               |
| 0.9266 | 1350 | 0.0228        | -               |
| 0.9609 | 1400 | 0.0186        | -               |
| 0.9952 | 1450 | 0.0164        | -               |
| 1.0295 | 1500 | 0.02          | -               |
| 1.0638 | 1550 | 0.0173        | -               |
| 1.0981 | 1600 | 0.0138        | -               |
| 1.1325 | 1650 | 0.0163        | -               |
| 1.1668 | 1700 | 0.0166        | -               |
| 1.2011 | 1750 | 0.0129        | -               |
| 1.2354 | 1800 | 0.0149        | -               |
| 1.2697 | 1850 | 0.0168        | -               |
| 1.3040 | 1900 | 0.0152        | -               |
| 1.3384 | 1950 | 0.0148        | -               |
| 1.3727 | 2000 | 0.0132        | -               |
| 1.4070 | 2050 | 0.0153        | -               |
| 1.4413 | 2100 | 0.0135        | -               |
| 1.4756 | 2150 | 0.0105        | -               |
| 1.5100 | 2200 | 0.0195        | -               |
| 1.5443 | 2250 | 0.0145        | -               |
| 1.5786 | 2300 | 0.0137        | -               |
| 1.6129 | 2350 | 0.013         | -               |
| 1.6472 | 2400 | 0.0179        | -               |
| 1.6815 | 2450 | 0.0119        | -               |
| 1.7159 | 2500 | 0.0112        | -               |
| 1.7502 | 2550 | 0.011         | -               |
| 1.7845 | 2600 | 0.0147        | -               |
| 1.8188 | 2650 | 0.0102        | -               |
| 1.8531 | 2700 | 0.0115        | -               |
| 1.8874 | 2750 | 0.0112        | -               |
| 1.9218 | 2800 | 0.0064        | -               |
| 1.9561 | 2850 | 0.0099        | -               |
| 1.9904 | 2900 | 0.0097        | -               |
| 2.0247 | 2950 | 0.0097        | -               |
| 2.0590 | 3000 | 0.0087        | -               |
| 2.0933 | 3050 | 0.0089        | -               |
| 2.1277 | 3100 | 0.0096        | -               |
| 2.1620 | 3150 | 0.0098        | -               |
| 2.1963 | 3200 | 0.009         | -               |
| 2.2306 | 3250 | 0.0119        | -               |
| 2.2649 | 3300 | 0.0065        | -               |
| 2.2992 | 3350 | 0.0088        | -               |
| 2.3336 | 3400 | 0.0073        | -               |
| 2.3679 | 3450 | 0.0092        | -               |
| 2.4022 | 3500 | 0.0095        | -               |
| 2.4365 | 3550 | 0.0072        | -               |
| 2.4708 | 3600 | 0.0065        | -               |
| 2.5051 | 3650 | 0.0085        | -               |
| 2.5395 | 3700 | 0.0072        | -               |
| 2.5738 | 3750 | 0.0087        | -               |
| 2.6081 | 3800 | 0.0088        | -               |
| 2.6424 | 3850 | 0.0097        | -               |
| 2.6767 | 3900 | 0.0085        | -               |
| 2.7111 | 3950 | 0.0077        | -               |
| 2.7454 | 4000 | 0.0071        | -               |
| 2.7797 | 4050 | 0.0098        | -               |
| 2.8140 | 4100 | 0.0062        | -               |
| 2.8483 | 4150 | 0.0083        | -               |
| 2.8826 | 4200 | 0.0073        | -               |
| 2.9170 | 4250 | 0.0084        | -               |
| 2.9513 | 4300 | 0.0093        | -               |
| 2.9856 | 4350 | 0.0082        | -               |

### Framework Versions
- Python: 3.12.13
- SetFit: 1.1.3
- Sentence Transformers: 5.6.1
- Transformers: 4.57.6
- PyTorch: 2.13.0+cu130
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->