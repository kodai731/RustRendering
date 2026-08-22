---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'query: 被写体との距離を広げて'
- text: 'query: スピードを上げて'
- text: 'query: what frame are we on'
- text: 'query: セーブして'
- text: 'query: 標準のカメラ位置に戻す'
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
- **Number of Classes:** 36 classes
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
| camera_direction                   | <ul><li>'query: 追いかけながら回り込むように'</li><li>'query: 広告風に迫力あるアングルで'</li><li>'query: ドラマチックに見せて'</li></ul>                                                                        |

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
| Word count   | 2   | 3.3295 | 11  |

| Label                              | Training Sample Count |
|:-----------------------------------|:----------------------|
| camera_direction                   | 43                    |
| camera_shot:crane_down             | 40                    |
| camera_shot:crane_up               | 40                    |
| camera_shot:dolly_in               | 40                    |
| camera_shot:dolly_out              | 40                    |
| camera_shot:look_at_selection      | 40                    |
| camera_shot:orbit_around_selection | 40                    |
| describe_selection                 | 40                    |
| focus_camera:model                 | 40                    |
| focus_camera:reset                 | 35                    |
| focus_camera:selection             | 42                    |
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
| set_playback_speed:fast            | 43                    |
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
| 0.0006 | 1    | 0.3893        | -               |
| 0.0290 | 50   | 0.3499        | -               |
| 0.0579 | 100  | 0.2766        | -               |
| 0.0869 | 150  | 0.1869        | -               |
| 0.1158 | 200  | 0.1586        | -               |
| 0.1448 | 250  | 0.136         | -               |
| 0.1737 | 300  | 0.1211        | -               |
| 0.2027 | 350  | 0.101         | -               |
| 0.2316 | 400  | 0.0911        | -               |
| 0.2606 | 450  | 0.0824        | -               |
| 0.2895 | 500  | 0.0736        | -               |
| 0.3185 | 550  | 0.0682        | -               |
| 0.3474 | 600  | 0.061         | -               |
| 0.3764 | 650  | 0.0541        | -               |
| 0.4053 | 700  | 0.0485        | -               |
| 0.4343 | 750  | 0.046         | -               |
| 0.4632 | 800  | 0.039         | -               |
| 0.4922 | 850  | 0.0407        | -               |
| 0.5211 | 900  | 0.0373        | -               |
| 0.5501 | 950  | 0.0362        | -               |
| 0.5790 | 1000 | 0.0318        | -               |
| 0.6080 | 1050 | 0.0295        | -               |
| 0.6369 | 1100 | 0.0293        | -               |
| 0.6659 | 1150 | 0.029         | -               |
| 0.6948 | 1200 | 0.0238        | -               |
| 0.7238 | 1250 | 0.024         | -               |
| 0.7528 | 1300 | 0.0272        | -               |
| 0.7817 | 1350 | 0.0218        | -               |
| 0.8107 | 1400 | 0.0234        | -               |
| 0.8396 | 1450 | 0.0236        | -               |
| 0.8686 | 1500 | 0.0213        | -               |
| 0.8975 | 1550 | 0.0199        | -               |
| 0.9265 | 1600 | 0.0215        | -               |
| 0.9554 | 1650 | 0.0208        | -               |
| 0.9844 | 1700 | 0.019         | -               |
| 1.0133 | 1750 | 0.0188        | -               |
| 1.0423 | 1800 | 0.0173        | -               |
| 1.0712 | 1850 | 0.0142        | -               |
| 1.1002 | 1900 | 0.0158        | -               |
| 1.1291 | 1950 | 0.0172        | -               |
| 1.1581 | 2000 | 0.0147        | -               |
| 1.1870 | 2050 | 0.0156        | -               |
| 1.2160 | 2100 | 0.0136        | -               |
| 1.2449 | 2150 | 0.0141        | -               |
| 1.2739 | 2200 | 0.0156        | -               |
| 1.3028 | 2250 | 0.0117        | -               |
| 1.3318 | 2300 | 0.0102        | -               |
| 1.3607 | 2350 | 0.015         | -               |
| 1.3897 | 2400 | 0.0114        | -               |
| 1.4186 | 2450 | 0.0122        | -               |
| 1.4476 | 2500 | 0.011         | -               |
| 1.4765 | 2550 | 0.0128        | -               |
| 1.5055 | 2600 | 0.0129        | -               |
| 1.5345 | 2650 | 0.0098        | -               |
| 1.5634 | 2700 | 0.0137        | -               |
| 1.5924 | 2750 | 0.0134        | -               |
| 1.6213 | 2800 | 0.0109        | -               |
| 1.6503 | 2850 | 0.0076        | -               |
| 1.6792 | 2900 | 0.0095        | -               |
| 1.7082 | 2950 | 0.0122        | -               |
| 1.7371 | 3000 | 0.0105        | -               |
| 1.7661 | 3050 | 0.0104        | -               |
| 1.7950 | 3100 | 0.0065        | -               |
| 1.8240 | 3150 | 0.0096        | -               |
| 1.8529 | 3200 | 0.0079        | -               |
| 1.8819 | 3250 | 0.008         | -               |
| 1.9108 | 3300 | 0.0077        | -               |
| 1.9398 | 3350 | 0.0066        | -               |
| 1.9687 | 3400 | 0.0067        | -               |
| 1.9977 | 3450 | 0.0087        | -               |
| 2.0266 | 3500 | 0.0089        | -               |
| 2.0556 | 3550 | 0.0068        | -               |
| 2.0845 | 3600 | 0.0082        | -               |
| 2.1135 | 3650 | 0.0088        | -               |
| 2.1424 | 3700 | 0.0092        | -               |
| 2.1714 | 3750 | 0.0089        | -               |
| 2.2003 | 3800 | 0.0113        | -               |
| 2.2293 | 3850 | 0.0078        | -               |
| 2.2583 | 3900 | 0.0068        | -               |
| 2.2872 | 3950 | 0.0068        | -               |
| 2.3162 | 4000 | 0.0078        | -               |
| 2.3451 | 4050 | 0.0067        | -               |
| 2.3741 | 4100 | 0.007         | -               |
| 2.4030 | 4150 | 0.0082        | -               |
| 2.4320 | 4200 | 0.0096        | -               |
| 2.4609 | 4250 | 0.0079        | -               |
| 2.4899 | 4300 | 0.0061        | -               |
| 2.5188 | 4350 | 0.0053        | -               |
| 2.5478 | 4400 | 0.0086        | -               |
| 2.5767 | 4450 | 0.0074        | -               |
| 2.6057 | 4500 | 0.0079        | -               |
| 2.6346 | 4550 | 0.0064        | -               |
| 2.6636 | 4600 | 0.0062        | -               |
| 2.6925 | 4650 | 0.0063        | -               |
| 2.7215 | 4700 | 0.0064        | -               |
| 2.7504 | 4750 | 0.0085        | -               |
| 2.7794 | 4800 | 0.0067        | -               |
| 2.8083 | 4850 | 0.009         | -               |
| 2.8373 | 4900 | 0.0082        | -               |
| 2.8662 | 4950 | 0.0058        | -               |
| 2.8952 | 5000 | 0.0049        | -               |
| 2.9241 | 5050 | 0.0081        | -               |
| 2.9531 | 5100 | 0.006         | -               |
| 2.9820 | 5150 | 0.0057        | -               |

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