---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'query: build a jump-up animation'
- text: 'query: 再生を開始させて'
- text: 'query: give me details on the selection'
- text: 'query: アニメの終わりへ移動'
- text: 'query: タイムラインの頭に戻して'
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
preds = model("query: 再生を開始させて")
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
| Word count   | 2   | 3.1739 | 10  |

| Label                      | Training Sample Count |
|:---------------------------|:----------------------|
| describe_selection         | 24                    |
| focus_camera:model         | 24                    |
| focus_camera:reset         | 24                    |
| focus_camera:selection     | 24                    |
| generate_motion:idle       | 24                    |
| generate_motion:jump       | 24                    |
| generate_motion:run        | 24                    |
| generate_motion:turn       | 24                    |
| generate_motion:walk       | 24                    |
| get_playback_state         | 24                    |
| list_objects               | 24                    |
| pause_animation            | 24                    |
| play_animation             | 24                    |
| redo                       | 24                    |
| save_scene                 | 24                    |
| seek_time:end              | 24                    |
| seek_time:next_key         | 24                    |
| seek_time:prev_key         | 24                    |
| seek_time:start            | 24                    |
| select_object              | 24                    |
| set_object_visibility:hide | 24                    |
| set_object_visibility:show | 24                    |
| set_playback_speed:fast    | 24                    |
| set_playback_speed:normal  | 24                    |
| set_playback_speed:slow    | 24                    |
| stop_animation             | 24                    |
| take_screenshot            | 24                    |
| toggle_loop                | 24                    |
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
| 0.0011 | 1    | 0.3819        | -               |
| 0.0575 | 50   | 0.3271        | -               |
| 0.1149 | 100  | 0.2122        | -               |
| 0.1724 | 150  | 0.1498        | -               |
| 0.2299 | 200  | 0.1186        | -               |
| 0.2874 | 250  | 0.0932        | -               |
| 0.3448 | 300  | 0.075         | -               |
| 0.4023 | 350  | 0.0671        | -               |
| 0.4598 | 400  | 0.0578        | -               |
| 0.5172 | 450  | 0.0468        | -               |
| 0.5747 | 500  | 0.0427        | -               |
| 0.6322 | 550  | 0.0398        | -               |
| 0.6897 | 600  | 0.0367        | -               |
| 0.7471 | 650  | 0.0388        | -               |
| 0.8046 | 700  | 0.0362        | -               |
| 0.8621 | 750  | 0.0335        | -               |
| 0.9195 | 800  | 0.0298        | -               |
| 0.9770 | 850  | 0.026         | -               |
| 1.0345 | 900  | 0.0224        | -               |
| 1.0920 | 950  | 0.0201        | -               |
| 1.1494 | 1000 | 0.0218        | -               |
| 1.2069 | 1050 | 0.0236        | -               |
| 1.2644 | 1100 | 0.0187        | -               |
| 1.3218 | 1150 | 0.0263        | -               |
| 1.3793 | 1200 | 0.0183        | -               |
| 1.4368 | 1250 | 0.0164        | -               |
| 1.4943 | 1300 | 0.0198        | -               |
| 1.5517 | 1350 | 0.0158        | -               |
| 1.6092 | 1400 | 0.016         | -               |
| 1.6667 | 1450 | 0.0149        | -               |
| 1.7241 | 1500 | 0.0165        | -               |
| 1.7816 | 1550 | 0.0194        | -               |
| 1.8391 | 1600 | 0.0148        | -               |
| 1.8966 | 1650 | 0.0174        | -               |
| 1.9540 | 1700 | 0.0169        | -               |
| 2.0115 | 1750 | 0.0138        | -               |
| 2.0690 | 1800 | 0.0132        | -               |
| 2.1264 | 1850 | 0.0119        | -               |
| 2.1839 | 1900 | 0.0132        | -               |
| 2.2414 | 1950 | 0.0105        | -               |
| 2.2989 | 2000 | 0.0108        | -               |
| 2.3563 | 2050 | 0.0145        | -               |
| 2.4138 | 2100 | 0.0116        | -               |
| 2.4713 | 2150 | 0.0113        | -               |
| 2.5287 | 2200 | 0.0139        | -               |
| 2.5862 | 2250 | 0.0089        | -               |
| 2.6437 | 2300 | 0.0143        | -               |
| 2.7011 | 2350 | 0.0146        | -               |
| 2.7586 | 2400 | 0.0109        | -               |
| 2.8161 | 2450 | 0.0103        | -               |
| 2.8736 | 2500 | 0.0109        | -               |
| 2.9310 | 2550 | 0.011         | -               |
| 2.9885 | 2600 | 0.0109        | -               |

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