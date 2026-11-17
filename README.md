## 1. Run Offline Validation First

Run the following from `code/`:

```bash
python3 scripts/demo.py
python3 -m unittest discover -s tests -v
python3 -m mrmem --help
```

The demo reproduces the two-step trajectory in Supplementary Section S2: first retrieve summaries of preferences and short walks, then refine only the walking constraints while retaining the preference summary. The demo uses whitespace-based token counting and validates execution semantics only. Actual experiments use the **answer model's tokenizer** to measure terminal memory length and the **policy tokenizer** to measure action length.

The tests include actual PyTorch parameter updates, as well as action-probability, KL-direction, and gradient checks using a randomly initialized tiny GPT-2. They neither download models nor access the judge API. They are not performance experiments on the paper's datasets.

## 2. Environment and Models

The implementation reuses the project's existing dependencies, with no new dependencies or automatic installation steps. The core requires Python ≥3.10, PyTorch, and Transformers; statistical analysis uses NumPy. The demo executor can run using only the standard library.

`configs/paper.json` specifies the paper's hyperparameters and model roles. Replace `models.*.path` with local weight paths. The default is `local_only=true`, so models are not downloaded automatically. The policy starts from **Qwen3-8B-Base** by default, the answer model is Qwen3-8B, and the embedding model is Qwen3-Embedding-8B.

The current trainer is a single-process reference implementation for full-parameter training, using per-trajectory accumulation to form effective minibatches. The policy, reference, answer, and embedding models can be placed on separate GPUs. Full-parameter training of an 8B model requires sufficient GPU memory for weights, gradients, Adam states, and activations. The four device assignments in the configuration illustrate model roles and do not guarantee that each model will fit on a 48 GB GPU. First validate the data and workflow with a local 0.6B or 4B model, then run according to available hardware capacity. Do not treat this implementation's throughput or latency as a reproduction of the paper's server measurements.

The judge model is accessed through an OpenAI-compatible `/chat/completions` endpoint, without an additional SDK. Set `judge.base_url` to your judge service and use the paper's GPT-4 Turbo model. Credentials are read from the `MRMEM_JUDGE_API_KEY` environment variable and are not written to code, configuration files, or logs.

## 3. Data Contract

Recommended layout:

```text
data/
  questions.jsonl
  repositories/<owner_id>.json
  splits/locomo_fixed.json
  teacher_traces.jsonl
  demonstrations.jsonl
models/
checkpoints/
runs/
```

For LoCoMo, `owner_id` identifies a conversation; for PersonaMem, it identifies a user. Each repository follows the structure in [examples/repository.json](examples/repository.json):

- `sources`: immutable original messages, including IDs, ISO 8601 timestamps, and text.
- `records`: `id, category, topic, attribute, value, summary, source_ids`; category is one of `episodic/profile/domain`.
- `parents`: unique IDs, summaries, and child record IDs; each `(category,topic,attribute)` corresponds to one parent, and each leaf has exactly one parent.
- `owner_id`: must match the questions and repository filename.

Each question line follows [examples/questions.jsonl](examples/questions.jsonl), with `id, owner_id, dataset, query, answer, category, objective` and optional `list_answer, support_ids`. `objective` is a **training correctness-evaluation mask fixed in advance**. It must be provided explicitly and must not be determined dynamically based on whether the model answers correctly. `support_ids` are IDs of extracted leaf records, not original message IDs. For PersonaMem, include the answer choices in query and store the gold option label in answer. During data preparation, convert the original benchmark fields and the mappings to extracted supporting evidence into this contract.

During training, objective=true uses normalized EM; otherwise, a binary judge is called. During evaluation, LoCoMo always uses the judge, while PersonaMem uses option-label EM. LoCoMo category 5 is retained in the training data and excluded only during evaluation.

Generate splits:

```bash
python3 -m mrmem splits --output data/splits/locomo_fixed.json
python3 -m mrmem splits --fold 0 --output data/splits/locomo_fold0.json
python3 -m mrmem splits --dataset personamem --owners data/user_ids.json --output data/splits/personamem.json
```

The fixed LoCoMo split matches the supplementary material. In ten-fold evaluation, the test owners are the paper's ten conversations in order; the inner validation owner is the next conversation cyclically, and the remaining eight are used for training. The paper does not specify the validation owner for each fold, so this is an implementation decision. The original PersonaMem user ID split is unavailable; by default, users are randomly split into 126/18/36 using seed 2027. Replace the manifest when the original split becomes available.

## 4. Construct and Standardize Repositories

Existing verified repositories of extracted records can be loaded directly. To construct a repository from original messages, create a configuration:

```json
{
  "sources": "data/raw/conv-26.json",
  "extractor": {"path": "models/Qwen3-8B", "device": "cuda:0", "dtype": "bfloat16"},
  "embedding": {"path": "models/Qwen3-Embedding-8B", "device": "cuda:1", "dtype": "bfloat16"},
  "max_tokens": 4096,
  "output": "data/repositories/conv-26.json"
}
```

The original message file has the form `{"owner_id":"conv-26","sources":[...]}`. Run `python3 -m mrmem construct --config configs/construction.json`. Construction accesses historical messages only, without accessing questions or answers. Temporal/semantic segmentation, episodic candidate thresholds, and candidate counts follow the paper. Invalid model output raises an error immediately to avoid silently writing incorrect source links.

Standardization to 500 records and growth to 1,000/1,500 records both use `resize`:

```json
{
  "repository": "data/natural/conv-26.json",
  "size": 500,
  "seed": 2027,
  "protected_ids": ["r000001", "r000008"],
  "donors": [],
  "allowed_donor_owners": [],
  "random_subsample": false,
  "output": "data/repositories/conv-26.json"
}
```

Run `python3 -m mrmem resize --config configs/resize.json`. `protected_ids` should include the supporting leaf records for all annotated questions belonging to the owner, **including test annotations**. This is the paper's evidence-preserving construction protocol for the factorial experiments and must not be presented as an annotation-free deployment workflow. System experiments on natural repositories do not require this standardization step.

When padding a repository, explicitly list the permitted donor owners: in ten-fold evaluation, pad each test repository from that fold's train+dev owners; in the fixed split, the two test conversations only pad each other; PersonaMem does not use padding across users. Growth experiments sample from PersonaMem **training users**. The tool checks donors against the provided allowlist; the experiment configuration is responsible for ensuring that the allowlist matches the split. During import, source/record IDs are namespaced and parents with the same `(M,D,L)` are merged. Each output is accompanied by a `.manifest.json` recording added/removed IDs, the seed, and the repository hash.

For evidence-sensitivity experiments, set `random_subsample=true` with seed `training_seed + fold_index`, and explicitly set `allow_missing_support=true` in the training/evaluation configuration. Questions whose supporting evidence was removed must be retained, not filtered out. Reuse the SFT checkpoint from the evidence-protected condition, but retrain FL/HL.

## 5. SFT → GRPO

See [examples/teacher_traces.jsonl](examples/teacher_traces.jsonl) for the format of human-reviewed teacher traces. Each trace stores only the owner, question ID, and action list; answers are not stored in the policy input. `compile-demos` replays trajectories using the actual executor and regenerates each step's policy input using only query/metadata/history/evidence:

```bash
python3 -m mrmem compile-demos --config configs/paper.json --traces data/teacher_traces.jsonl --output data/demonstrations.jsonl
python3 -m mrmem sft --config configs/sft_run.json
python3 -m mrmem grpo --config configs/grpo_run.json
```

Both run configurations can be created by copying and editing `paper.json`:

1. SFT: point policy to the base weights, set `output` to a new directory, and populate `split/demonstrations`. Train for at most 3 epochs for the full system, or at most 2 epochs for `interface.protocol=scope`. Evaluate validation loss each epoch, save `best/last`, and stop early after one epoch without improvement. Training uses only train owners, and validation uses only dev owners.
2. GRPO: change the policy path to `SFT_output/best` for the corresponding fold and seed, and set `output` to a new directory. The frozen reference is automatically loaded from **the same starting checkpoint**. Use 1000 updates, 8 groups × 8 trajectories, 4 epochs/batch, minibatch size 8, 100 warmup steps, learning rate 1e-6, clipping .2, KL weight .01, and length weight .7.
3. Training from the base model: skip SFT and point policy to the base model; the frozen reference automatically uses the base model as well.
4. no-length: set `training.length_weight=0` and train independently.

Scope-only demonstration completions have the form `{"subspace_ids":[...]}`. Teacher traces for shared multi-format SFT must cover all formats used, including flat, hierarchical, leaf_only, and opaque. Each trace can override the defaults through `interface`. First replay all formats into one demonstrations.jsonl, then train one shared scope SFT model for the same fold and seed before launching separate FL/HL GRPO runs. **Do not train FL-SFT and HL-SFT separately and describe them as a shared initialization.**

`compile-demos` validates format and executability. Alignment with semantic support, human-review flags, and demonstration coverage must be verified during actual data preparation. This repository does not present synthetic trajectories as the paper's GPT-4 Turbo demonstrations with human review.

SFT uses cross-entropy over action tokens, excluding prompt tokens from the loss. GRPO sums log-probabilities over all generated tokens across the two actions, including any actually generated EOS token, and computes the **trajectory probability ratio** before clipping. Old-policy log-probabilities are saved before updates. KL is the sum of full-vocabulary `KL(ref || policy)` over action tokens, not a sampling approximation. Reward standard deviation is computed as the population standard deviation. Groups with zero variance have advantage 0, but the KL term remains active.

## 6. Batch Experiments and Evaluation

```bash
python3 scripts/make_experiments.py --folds tenfold --conditions FC HC FL HL FL_SFT HL_SFT --output experiments/coarse
python3 scripts/make_experiments.py --folds tenfold --granularity fine --conditions FC HC FL HL --output experiments/fine
python3 scripts/make_experiments.py --folds tenfold --conditions hierarchical_leaf_only opaque_masked --output experiments/components
python3 scripts/make_experiments.py --conditions MRMem flat_two_stage fixed_coarse fixed_fine parent_fine adaptive no_length_reward from_base --output experiments/system
```

The output `tasks.json` contains executable command arrays and the corresponding run configurations. The generator **only writes configurations**; it does not automatically launch GPU training or judge requests. Execute the generated commands to run the experiments. By default, it looks for `checkpoints/fold_N/seed_N/scope_sft/best` or `full_sft/best`, using `fixed/` for the fixed split. Train the SFT checkpoints for each fold/seed first, as described in Section 5.

The experiment matrix generator is specific to LoCoMo. For PersonaMem, specify the generated user split in a separate configuration. Factorial experiments use standardized repositories. Write system results on natural repositories, factorial results on standardized repositories, and fine-granularity conditions to separate run directories; do not mix them in analysis.

`--grid` generates the validation grid specified in the paper. Select configurations independently for FL/HL in each fold, without using test scores. Select HC's `k_topic` from 1..5 using dev accuracy. For the main RRF comparator, select the highest accuracy; you can use `select --tolerance 0`. For length-weight candidates, prefer the lowest memory cost among candidates within 0.01 of the highest accuracy (default tolerance .01). The generator does not select configurations based on test results. When running `FC/HC` alone without a grid, `k_topic=3` is the default, not a claim that it is optimal in every fold.

GRPO validates every 50 updates, an adjustable implementation choice. Select `best` by validation accuracy, breaking exact ties by memory cost. `last` is the final update. Selection across hyperparameters with an accuracy tolerance and checkpoint selection within a single training run are separate steps.

Set policy in the evaluation configuration to the selected checkpoint, specify a separate output, and run:

```bash
python3 -m mrmem evaluate --config configs/eval_run.json --partition test
python3 -m mrmem evaluate --config configs/dev_run.json --partition dev
python3 -m mrmem select --candidates data/validation_candidates.json --output runs/selected.json
```

`method=rrf` is the main conventional comparator: FC applies RRF to all leaves, while HC applies parent RRF followed by descendant RRF, with a fixed top-5. Use `zero_shot=true` with a base checkpoint for temperature 0; learned and SFT-only evaluation retain temperature .7. Fixed fine granularity displays parent summaries and all leaf metadata; parent_fine displays only parent summaries. The two-stage interface initially displays only parent summaries, then displays all child leaf summaries/metadata within the selected parent scopes in the second step.

Adaptive granularity requires an additional trained `models.granularity_gate`. This model takes query＋parent metadata and outputs exactly `coarse` or `fine`. It can be trained through the generic SFT entry point using binary text labels. Annotations may come only from train/dev owners. At inference time, the frozen gate is called first, and its decision determines whether to display parent metadata or parent＋all-leaf metadata. The gate's input/output counts toward online tokens. The test-set fine-granularity proportion is determined by actual model outputs, not hardcoded to the paper's 22.4%.

Outputs include `predictions.jsonl`, `metrics.json`, and `provenance.json`. Each question records the owner, seed, condition, prediction, correctness, evidence IDs, terminal memory tokens, cumulative policy/answer tokens, call counts, invalid actions, and latency. Latency includes the local implementation's complete per-question processing and cannot be directly compared with the paper's dedicated latency measurements.

## 7. Paired Statistics

```bash
python3 -m mrmem analyze --predictions runs/FC/predictions.jsonl runs/HC/predictions.jsonl runs/FL/predictions.jsonl runs/HL/predictions.jsonl --output runs/factorial.json
python3 -m mrmem analyze --predictions runs/system/predictions.jsonl runs/fine/predictions.jsonl --weights '{"MRMem":1,"fixed_fine":-1}' --output runs/disclosure_gap.json
python3 -m mrmem analyze --predictions runs/system/predictions.jsonl runs/fine/predictions.jsonl --weights '{"MRMem":1,"fixed_fine":-1}' --metric memory_tokens --output runs/token_gap.json
```

Provide the actual paths to per-question files for all folds and all three seeds. The analyzer requires identical owner/question/seed coverage across cells and refuses to silently take their intersection. Run and record deterministic comparators with the three corresponding seed labels as well, to ensure strict pairing.

First compute the paired three-seed mean for each question, then cluster by owner and perform 10,000 paired bootstrap replicates weighted by QA count. The fixed split with two owners yields point estimates only, without presenting confidence intervals that appear reliable. When both `FL_SFT/HL_SFT` are provided, the analyzer automatically computes point estimates for the SFT interaction, RL increment, and RL share; it does not fabricate a confidence interval for the ratio. The sign test is labeled nominal and does not eliminate dependencies caused by shared training histories across the ten folds.
