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
- **Number of Classes:** 29 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                      | Examples                                                                                                                                                  |
|:---------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| list_objects               | <ul><li>"query: what's in this scene"</li><li>'query: give me a list of all objects'</li><li>'query: enumerate the scene contents'</li></ul>              |
| describe_selection         | <ul><li>'query: what did I select'</li><li>'query: give me details on the selection'</li><li>'query: explain the selected object'</li></ul>               |
| get_playback_state         | <ul><li>'query: how fast is it playing'</li><li>'query: tell me the playback status'</li><li>'query: are we paused'</li></ul>                             |
| take_screenshot            | <ul><li>'query: screenshot please'</li><li>'query: save a picture of the viewport'</li><li>'query: snap the current frame'</li></ul>                      |
| play_animation             | <ul><li>'query: begin the animation'</li><li>'query: hit play'</li><li>'query: let the animation roll'</li></ul>                                          |
| pause_animation            | <ul><li>'query: pause it'</li><li>'query: freeze the playback here'</li><li>'query: hold the current frame'</li></ul>                                     |
| stop_animation             | <ul><li>'query: stop and rewind'</li><li>'query: end playback and reset to frame zero'</li><li>'query: abort the animation and return to start'</li></ul> |
| set_playback_speed:slow    | <ul><li>'query: slow the playback down'</li><li>'query: reduce the play speed'</li><li>'query: playback at slow speed'</li></ul>                          |
| set_playback_speed:normal  | <ul><li>'query: set the speed back to normal'</li><li>'query: default playback speed please'</li><li>'query: restore standard play speed'</li></ul>       |
| set_playback_speed:fast    | <ul><li>'query: speed up the playback'</li><li>'query: play it faster'</li><li>'query: increase the play speed'</li></ul>                                 |
| seek_time:start            | <ul><li>'query: go to frame zero'</li><li>'query: move the playhead to the beginning'</li><li>'query: rewind to the first frame'</li></ul>                |
| seek_time:end              | <ul><li>'query: seek to the last frame'</li><li>'query: move the playhead to the end'</li><li>'query: jump to the finish of the timeline'</li></ul>       |
| seek_time:next_key         | <ul><li>'query: advance to the following keyframe'</li><li>'query: step forward to the next key'</li><li>'query: jump ahead one keyframe'</li></ul>       |
| seek_time:prev_key         | <ul><li>'query: step back to the previous key'</li><li>'query: jump to the earlier keyframe'</li><li>'query: seek to the prior key'</li></ul>             |
| toggle_loop                | <ul><li>'query: enable loop playback'</li><li>'query: switch looping off'</li><li>'query: flip the loop setting'</li></ul>                                |
| select_object              | <ul><li>'query: select the sphere'</li><li>'query: choose the object called Robot'</li><li>'query: pick Light01'</li></ul>                                |
| set_object_visibility:show | <ul><li>'query: make the sphere visible again'</li><li>'query: unhide the wall'</li><li>'query: show Robot'</li></ul>                                     |
| set_object_visibility:hide | <ul><li>'query: hide the wall'</li><li>'query: make the sphere invisible'</li><li>'query: turn off visibility for Light01'</li></ul>                      |
| focus_camera:selection     | <ul><li>'query: zoom to what I selected'</li><li>'query: center the view on the selected object'</li><li>'query: look at the selection'</li></ul>         |
| focus_camera:model         | <ul><li>'query: fit the entire model on screen'</li><li>'query: zoom out to see the whole character'</li><li>'query: frame everything'</li></ul>          |
| focus_camera:reset         | <ul><li>'query: put the camera back to default'</li><li>'query: restore the original view'</li><li>'query: reset the viewport camera'</li></ul>           |
| undo                       | <ul><li>'query: take that back'</li><li>'query: cancel the last operation'</li><li>'query: roll back what I just did'</li></ul>                           |
| redo                       | <ul><li>'query: reapply what I undid'</li><li>'query: bring back the change I reverted'</li><li>'query: restore the undone change'</li></ul>              |
| save_scene                 | <ul><li>'query: write the scene to disk'</li><li>'query: store my progress'</li><li>'query: please save'</li></ul>                                        |
| generate_motion:walk       | <ul><li>'query: create a walking animation'</li><li>'query: make the character walk'</li><li>'query: generate a walk cycle'</li></ul>                     |
| generate_motion:run        | <ul><li>'query: generate a running animation'</li><li>'query: make the character run'</li><li>'query: create a run cycle'</li></ul>                       |
| generate_motion:idle       | <ul><li>'query: make an idle pose animation'</li><li>'query: generate a standing idle'</li><li>'query: create an idle loop'</li></ul>                     |
| generate_motion:jump       | <ul><li>'query: create a jump animation'</li><li>'query: make the character jump'</li><li>'query: generate a jumping motion'</li></ul>                    |
| generate_motion:turn       | <ul><li>'query: generate a turning animation'</li><li>'query: make the character turn around'</li><li>'query: create a turn motion'</li></ul>             |

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
| Word count   | 2   | 3.1180 | 10  |

| Label                      | Training Sample Count |
|:---------------------------|:----------------------|
| describe_selection         | 40                    |
| focus_camera:model         | 40                    |
| focus_camera:reset         | 35                    |
| focus_camera:selection     | 40                    |
| generate_motion:idle       | 38                    |
| generate_motion:jump       | 40                    |
| generate_motion:run        | 34                    |
| generate_motion:turn       | 38                    |
| generate_motion:walk       | 40                    |
| get_playback_state         | 40                    |
| list_objects               | 38                    |
| pause_animation            | 40                    |
| play_animation             | 40                    |
| redo                       | 24                    |
| save_scene                 | 40                    |
| seek_time:end              | 40                    |
| seek_time:next_key         | 40                    |
| seek_time:prev_key         | 38                    |
| seek_time:start            | 40                    |
| select_object              | 37                    |
| set_object_visibility:hide | 32                    |
| set_object_visibility:show | 37                    |
| set_playback_speed:fast    | 40                    |
| set_playback_speed:normal  | 39                    |
| set_playback_speed:slow    | 40                    |
| stop_animation             | 39                    |
| take_screenshot            | 40                    |
| toggle_loop                | 40                    |
| undo                       | 24                    |

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
| 0.0007 | 1    | 0.279         | -               |
| 0.0366 | 50   | 0.3358        | -               |
| 0.0732 | 100  | 0.25          | -               |
| 0.1097 | 150  | 0.1722        | -               |
| 0.1463 | 200  | 0.1449        | -               |
| 0.1829 | 250  | 0.1136        | -               |
| 0.2195 | 300  | 0.0978        | -               |
| 0.2560 | 350  | 0.083         | -               |
| 0.2926 | 400  | 0.0684        | -               |
| 0.3292 | 450  | 0.064         | -               |
| 0.3658 | 500  | 0.0549        | -               |
| 0.4023 | 550  | 0.0505        | -               |
| 0.4389 | 600  | 0.0411        | -               |
| 0.4755 | 650  | 0.0388        | -               |
| 0.5121 | 700  | 0.037         | -               |
| 0.5486 | 750  | 0.0418        | -               |
| 0.5852 | 800  | 0.0315        | -               |
| 0.6218 | 850  | 0.0308        | -               |
| 0.6584 | 900  | 0.0304        | -               |
| 0.6950 | 950  | 0.0288        | -               |
| 0.7315 | 1000 | 0.0248        | -               |
| 0.7681 | 1050 | 0.0247        | -               |
| 0.8047 | 1100 | 0.026         | -               |
| 0.8413 | 1150 | 0.0211        | -               |
| 0.8778 | 1200 | 0.0223        | -               |
| 0.9144 | 1250 | 0.0218        | -               |
| 0.9510 | 1300 | 0.0203        | -               |
| 0.9876 | 1350 | 0.0192        | -               |
| 1.0241 | 1400 | 0.0173        | -               |
| 1.0607 | 1450 | 0.0159        | -               |
| 1.0973 | 1500 | 0.0158        | -               |
| 1.1339 | 1550 | 0.0157        | -               |
| 1.1704 | 1600 | 0.0144        | -               |
| 1.2070 | 1650 | 0.0123        | -               |
| 1.2436 | 1700 | 0.015         | -               |
| 1.2802 | 1750 | 0.0123        | -               |
| 1.3168 | 1800 | 0.0147        | -               |
| 1.3533 | 1850 | 0.0123        | -               |
| 1.3899 | 1900 | 0.0139        | -               |
| 1.4265 | 1950 | 0.0103        | -               |
| 1.4631 | 2000 | 0.0147        | -               |
| 1.4996 | 2050 | 0.0122        | -               |
| 1.5362 | 2100 | 0.0084        | -               |
| 1.5728 | 2150 | 0.0152        | -               |
| 1.6094 | 2200 | 0.0144        | -               |
| 1.6459 | 2250 | 0.012         | -               |
| 1.6825 | 2300 | 0.013         | -               |
| 1.7191 | 2350 | 0.0101        | -               |
| 1.7557 | 2400 | 0.0113        | -               |
| 1.7922 | 2450 | 0.0092        | -               |
| 1.8288 | 2500 | 0.0095        | -               |
| 1.8654 | 2550 | 0.0086        | -               |
| 1.9020 | 2600 | 0.0098        | -               |
| 1.9386 | 2650 | 0.0109        | -               |
| 1.9751 | 2700 | 0.0078        | -               |
| 2.0117 | 2750 | 0.0101        | -               |
| 2.0483 | 2800 | 0.0068        | -               |
| 2.0849 | 2850 | 0.0077        | -               |
| 2.1214 | 2900 | 0.0056        | -               |
| 2.1580 | 2950 | 0.0081        | -               |
| 2.1946 | 3000 | 0.0073        | -               |
| 2.2312 | 3050 | 0.0093        | -               |
| 2.2677 | 3100 | 0.0076        | -               |
| 2.3043 | 3150 | 0.007         | -               |
| 2.3409 | 3200 | 0.0061        | -               |
| 2.3775 | 3250 | 0.0068        | -               |
| 2.4140 | 3300 | 0.0103        | -               |
| 2.4506 | 3350 | 0.0084        | -               |
| 2.4872 | 3400 | 0.0099        | -               |
| 2.5238 | 3450 | 0.0049        | -               |
| 2.5604 | 3500 | 0.0065        | -               |
| 2.5969 | 3550 | 0.0085        | -               |
| 2.6335 | 3600 | 0.0104        | -               |
| 2.6701 | 3650 | 0.0056        | -               |
| 2.7067 | 3700 | 0.008         | -               |
| 2.7432 | 3750 | 0.0074        | -               |
| 2.7798 | 3800 | 0.0067        | -               |
| 2.8164 | 3850 | 0.0069        | -               |
| 2.8530 | 3900 | 0.0101        | -               |
| 2.8895 | 3950 | 0.0057        | -               |
| 2.9261 | 4000 | 0.0073        | -               |
| 2.9627 | 4050 | 0.0058        | -               |
| 2.9993 | 4100 | 0.0049        | -               |

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