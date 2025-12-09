---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:25820
- loss:CosineSimilarityLoss
base_model: sentence-transformers/multi-qa-mpnet-base-cos-v1
widget:
- source_sentence: LAST DAILY SHUT DOWN IDLE OVER RIDE SPEED CHECK - ENGINE RPM
  sentences:
  - DURING THE LAST DAILY SHUTDOWN THE IDLE-OVERRIDE SPEED WAS CHECKED AND ENGINE
    RPM RECORDED.
  - AT THE LAST DAILY SHUT-DOWN WE RAN AN IDLE OVERRIDE SPEED CHECK AND NOTED THE
    ENGINE RPM.
  - THE ROCKER COVER FOR CYLINDER TWO AND CYLINDER ONE IS MISSING A PAIR OF SCREWS.
- source_sentence: LEFT ENGINE BAFFLE ACCESS HOLE ELONGATED.
  sentences:
  - INSPECTION REVEALED A BROKEN BAFFLE WIRE.
  - 'ACCESS HOLE ON THE #1 ENGINE BAFFLE SHOWS ELONGATION.'
  - 'LEAK OBSERVED AT THE EXHAUST PUSHROD SEAL OF THE LEFT #2 CYLINDER.'
- source_sentence: WORN FRONT BAFFLE SEALS OBSERVED DURING INSPECTION.
  sentences:
  - INSPECTION REVEALED WEAR ON THE FRONT BAFFLE SEALING STRIPS.
  - NOTED A SOILED ENGINE SCROLL; RECOMMEND CLEANING.
  - 'CAM LOCK ON THE #1 ENGINE BAFFLE COVER MUST BE FIXED OR REPLACED.'
- source_sentence: THE RIGHT FORWARD UPPER BAFFLE SEAL NEEDS TO BE RESECURED.
  sentences:
  - FOUND THE VALVE ON CYLINDER 3 STICKING DURING THE INSPECTION.
  - FOUND THE FORWARD UPPER BAFFLE SEAL LOOSE ON THE LEFT — SECURE IT AGAIN.
  - 'THE RIGHT BAFFLE IS LOOSE — APPEARS THE #3 CYLINDER BAFFLE SEAL IS FAILING.'
- source_sentence: 'THE #2 CYLINDER ROCKER COVER IS MISSING A SCREW.'
  sentences:
  - INSPECTION SHOWS ABRASION MARKS ON THE BLACK BAFFLE SEAL AT THE LOWER FORWARD
    ENGINE AREA BY THE AIR FILTER.
  - 'ROCKER COVER GASKET LEAKING ON CYLINDERS #2 AND #3.'
  - THE LEFT-SIDE ROCKER COVER ON CYLINDER 3 IS WITHOUT A FASTENER.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/multi-qa-mpnet-base-cos-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) on the conserta-avioes dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) <!-- at revision d51b22a1dfa8184e9258074e56e2875e50612dca -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - conserta-avioes
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'THE #2 CYLINDER ROCKER COVER IS MISSING A SCREW.',
    'THE LEFT-SIDE ROCKER COVER ON CYLINDER 3 IS WITHOUT A FASTENER.',
    'INSPECTION SHOWS ABRASION MARKS ON THE BLACK BAFFLE SEAL AT THE LOWER FORWARD ENGINE AREA BY THE AIR FILTER.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9553, -0.1387],
#         [ 0.9553,  1.0000, -0.1357],
#         [-0.1387, -0.1357,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
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

### Training Dataset

#### conserta-avioes

* Dataset: conserta-avioes
* Size: 25,820 training samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                         | sentence2                                                                         | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | string                                                                            | int                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.61 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 16.61 tokens</li><li>max: 34 tokens</li></ul> | <ul><li>0: ~49.90%</li><li>1: ~50.10%</li></ul> |
* Samples:
  | sentence1                                                                   | sentence2                                                      | label          |
  |:----------------------------------------------------------------------------|:---------------------------------------------------------------|:---------------|
  | <code>COULDN'T GET THE AIRPLANE TO START.</code>                            | <code>ENGINE WOULD NOT CRANK DURING THE START ATTEMPT.</code>  | <code>1</code> |
  | <code>FOUND THE LEFT BAFFLE SEAL BY THE PROP GOVERNOR HANGING LOOSE.</code> | <code>RIGHT-SIDE PROP GOVERNOR BAFFLE SEAL NOT SECURED.</code> | <code>1</code> |
  | <code>THERE'S OIL DRIPPING OVER THE RIGHT MAGS.</code>                      | <code>FOUND OIL SEEPING ABOVE THE RIGHT MAGNETOS.</code>       | <code>1</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Evaluation Dataset

#### conserta-avioes

* Dataset: conserta-avioes
* Size: 3,220 evaluation samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                         | sentence2                                                                        | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | string                                                                           | int                                             |
  | details | <ul><li>min: 4 tokens</li><li>mean: 15.48 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 16.4 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>0: ~49.20%</li><li>1: ~50.80%</li></ul> |
* Samples:
  | sentence1                                                                                       | sentence2                                                                          | label          |
  |:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------|
  | <code>PISTON PINS FOR CYLINDERS 3 AND 4 WERE DAMAGED DURING REMOVAL.</code>                     | <code>THE NO.3/NO.4 PISTON PINS WERE HARMED DURING DISASSEMBLY.</code>             | <code>1</code> |
  | <code>RIGHT ENGINE CYLINDER NO. 2 EXHIBITS A POSSIBLE CRACK ADJACENT TO THE INTAKE PORT.</code> | <code>POSSIBLE FRACTURE AT THE INTAKE-PORT AREA OF LEFT ENGINE CYLINDER #2.</code> | <code>1</code> |
  | <code>#2 CYLINDER HEAD TEMPERATURE FITTING NEEDS TO BE REPLACED.</code>                         | <code>CYLINDER #1 HEAD TEMPERATURE FITTING (CHT) NEEDS TO BE REPLACED.</code>      | <code>1</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `learning_rate`: 5e-06
- `num_train_epochs`: 4
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-06
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch      | Step     | Training Loss | Validation Loss |
|:----------:|:--------:|:-------------:|:---------------:|
| 0.0620     | 100      | 0.0533        | 0.0320          |
| 0.1239     | 200      | 0.0387        | 0.0282          |
| 0.1859     | 300      | 0.0358        | 0.0261          |
| 0.2478     | 400      | 0.0327        | 0.0243          |
| 0.3098     | 500      | 0.0315        | 0.0233          |
| 0.3717     | 600      | 0.031         | 0.0226          |
| 0.4337     | 700      | 0.0292        | 0.0215          |
| 0.4957     | 800      | 0.026         | 0.0209          |
| 0.5576     | 900      | 0.029         | 0.0199          |
| 0.6196     | 1000     | 0.0269        | 0.0192          |
| 0.6815     | 1100     | 0.0253        | 0.0192          |
| 0.7435     | 1200     | 0.0247        | 0.0190          |
| 0.8055     | 1300     | 0.0234        | 0.0185          |
| 0.8674     | 1400     | 0.0239        | 0.0181          |
| 0.9294     | 1500     | 0.0218        | 0.0179          |
| 0.9913     | 1600     | 0.0207        | 0.0180          |
| 1.0533     | 1700     | 0.0193        | 0.0176          |
| 1.1152     | 1800     | 0.0179        | 0.0175          |
| 1.1772     | 1900     | 0.0201        | 0.0174          |
| 1.2392     | 2000     | 0.0196        | 0.0172          |
| 1.3011     | 2100     | 0.019         | 0.0172          |
| 1.3631     | 2200     | 0.0185        | 0.0172          |
| 1.4250     | 2300     | 0.0196        | 0.0172          |
| 1.4870     | 2400     | 0.0166        | 0.0169          |
| 1.5489     | 2500     | 0.0172        | 0.0166          |
| 1.6109     | 2600     | 0.0195        | 0.0166          |
| 1.6729     | 2700     | 0.0182        | 0.0165          |
| 1.7348     | 2800     | 0.0194        | 0.0162          |
| 1.7968     | 2900     | 0.0162        | 0.0161          |
| 1.8587     | 3000     | 0.0171        | 0.0160          |
| 1.9207     | 3100     | 0.0176        | 0.0162          |
| 1.9827     | 3200     | 0.0181        | 0.0162          |
| 2.0446     | 3300     | 0.0157        | 0.0161          |
| 2.1066     | 3400     | 0.0156        | 0.0159          |
| 2.1685     | 3500     | 0.0152        | 0.0159          |
| **2.2305** | **3600** | **0.0157**    | **0.0156**      |
| 2.2924     | 3700     | 0.0165        | 0.0155          |
| 2.3544     | 3800     | 0.014         | 0.0155          |
| 2.4164     | 3900     | 0.0119        | 0.0156          |
| 2.4783     | 4000     | 0.0157        | 0.0157          |
| 2.5403     | 4100     | 0.0148        | 0.0155          |
| 2.6022     | 4200     | 0.0155        | 0.0153          |
| 2.6642     | 4300     | 0.0143        | 0.0154          |
| 2.7261     | 4400     | 0.016         | 0.0155          |
| 2.7881     | 4500     | 0.0168        | 0.0156          |
| 2.8501     | 4600     | 0.0144        | 0.0156          |
| 2.9120     | 4700     | 0.0141        | 0.0156          |
| 2.9740     | 4800     | 0.0115        | 0.0157          |
| 3.0359     | 4900     | 0.0131        | 0.0157          |
| 3.0979     | 5000     | 0.013         | 0.0156          |
| 3.1599     | 5100     | 0.0132        | 0.0155          |
| 3.2218     | 5200     | 0.0134        | 0.0155          |
| 3.2838     | 5300     | 0.0145        | 0.0155          |
| 3.3457     | 5400     | 0.0152        | 0.0155          |
| 3.4077     | 5500     | 0.0122        | 0.0154          |
| 3.4696     | 5600     | 0.0149        | 0.0154          |
| 3.5316     | 5700     | 0.0164        | 0.0154          |
| 3.5936     | 5800     | 0.0109        | 0.0154          |
| 3.6555     | 5900     | 0.0134        | 0.0153          |
| 3.7175     | 6000     | 0.0129        | 0.0153          |
| 3.7794     | 6100     | 0.0124        | 0.0153          |
| 3.8414     | 6200     | 0.0124        | 0.0153          |
| 3.9033     | 6300     | 0.0119        | 0.0153          |
| 3.9653     | 6400     | 0.0126        | 0.0153          |

* The bold row denotes the saved checkpoint.

### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.2
- PyTorch: 2.9.0+cu126
- Accelerate: 1.12.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
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