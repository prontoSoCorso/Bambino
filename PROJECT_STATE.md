# PROJECT_STATE.md

**Project:** BAMBINO — AI-Based Infant Hearing Screening
**Reference paper:** *A Benchmark Study for Reporting Feasibility in AI-Based Infant Hearing Screening: Exploring the Limits of Passive Sensing* (AIME 2026, Oral Long, accepted with minor revisions)
**Repository root:** `/home/phd2/Scrivania/CorsoRepo/Bambino`
**Old Data root:** `/home/phd2/Scrivania/CorsoRepo/Bambino/data/raw`
**NEW Full Data root:** `/home/phd2/Scrivania/CorsoRepo/Bambino/_0_main_project/data/full/raw` (Claude Code MUST use this path for the new full dataset)
**Last updated:** 2026-05-05

---

## 1. General Specs

### 1.1 Clinical Problem Statement

The project targets the **diagnostic blind spot for infants with a developmental age of 3–7 months** in audiological screening. This cohort is:

- **Too mature** for the primitive startle/Moro reflexes leveraged by neonatal screening (UNHS, ABR/OAE).
- **Too immature** for Visual Reinforcement Audiometry (VRA), which requires sufficient neck-muscle control and the cognitive ability to associate sound with a reinforced visual reward (typically achieved by ≥7 months).

The current clinical fallback is **Behavioural Observation Audiometry (BOA)**: an audiologist presents suprathreshold sounds and visually scores subtle reflexive behaviours (eye widening, sucking pause, micro head turns). BOA is effective in expert hands but is **subjective, non-standardised, and exhibits high inter-observer variability**. The engineering target is a passive, camera-only screening system that reproduces or augments the clinician's BOA judgement at scale, identifying infants who fail to respond and warrant referral for definitive diagnostic testing.

### 1.2 Sensing Scope

- **Sensor:** Single Sony HDR-CX625 camera, **25 fps**, frontal mount below the central monitor.
- **Cohort:** 46 healthy infants (mean 4.68 months; central 95% range 3.68–6.44). Risk-factor exclusions enforced for a normative sample.
- **Stimulus protocol:**
  - Loudspeakers at **±45° azimuth**, 70 dB SPL, 4 s sounds.
  - Four stimulus macro-categories: 1 kHz warble tone, 1 kHz narrowband noise, 1 kHz filtered speech/music, generic/personalised "other" sounds.
  - **Control = 500 ms silent intervals** (no stimulus).
  - **Passive paradigm (no visual reward).** Central monitor displays continuous moving colourful shapes to anchor frontal head pose for camera capture.
  - ≈20 trials per infant. Class prior **Stimulus 80.25% / Control 19.75%**.
- **Feature extraction:** **OpenFace 2.0** → **38 channels per frame**, partitioned into:
  - `g` (gaze) — **8 channels** (gaze direction X/Y/Z, gaze angles, both eyes).
  - `h` (head pose) — **13 channels** (translation Tx/Ty/Tz, rotation Rx/Ry/Rz, scale, etc.).
  - `f` (facial action units) — **17 channels** (brow raise, cheek raise, blink, jaw drop, …).
- **Trial window:** 12 s total (–2 s pre-stimulus → +10 s post-stimulus). The current supervised pipeline feeds **only the 0–10 s post-stimulus segment** (250 frames @ 25 fps) to classifiers.
- **Quality control:** OpenFace per-frame confidence; trials are dropped if the **first 52 frames (≈2 s)** have confidence ≤50% or are explicitly flagged `low_confidence_for_trial`. Implemented in [DataUtils/OpenFaceDataset.py](DataUtils/OpenFaceDataset.py).
- **Validation regime:** Strict **Subject-Independent Hold-out**. Each infant resides exclusively in one of {train, val, test}. No cross-split leakage.

### 1.3 Repository Topology (active modules only; `zNO_*` deprecated and excluded)

```
Bambino/
├── config.py                          # Global seeds, paths, modality dims, dataset tiers
├── DataUtils/                         # OpenFaceInstance, OpenFaceDataset, BoaOpenFaceDataset
├── _utils_/                           # dataset_utils, models_utils, plot_utils
├── _0_main_project/data/full/raw/     # NEW location for the full raw dataset
├── data/                              # Legacy data dir (raw → preprocessed → normalized → augmented)
├── _01_preprocessing/
│   ├── preprocessing_temporal_series.ipynb
│   ├── augment_time_series.ipynb      # 6 augmentation primitives, class-aware
│   └── dataset_stats.ipynb
├── _02_stats_and_normalization/
│   ├── main_normalization.py          # per-modality z-norm using train statistics
│   └── compute_umap.ipynb             # UMAP across all modality subsets
├── _03_train/
│   ├── logistic_regression/log_regr.ipynb
│   ├── minirocket_and_metadata/minirocket_xgb.ipynb
│   ├── FiLM/                          # config_film.py, film_cnn.ipynb
│   ├── InceptionTime/                 # config_inception.py, inceptionTime.ipynb
│   ├── resnet_on_images/              # image_creation.py (GASF), resnet.ipynb
│   ├── moment/                        # config_moment.py, main_moment.ipynb (5 heads)
│   └── moment_anomaly_detection/      # config_ad.py, main_ad.ipynb (preliminary AD scaffold)
├── _04_test/best_models/              # best_lstmfcn.pth, rocket_model.joblib, …
├── paper_results/
│   ├── performance_comparison/        # performance_comparison.csv, final_test.ipynb
│   └── UMAP/                          # frozen figures from the paper
├── data/                              # 5 tiers: raw → preprocessed → normalized → augmented → augmented_normalized
├── logs/, optuna/                     # training logs, Optuna SQLite studies
└── requirements.txt                   # torch 2.7.0+cu128, sklearn 1.5.0, optuna 3.6.1, …
```

Data tiers are pickled `BoaOpenFaceDataset` objects (`training_set.pt`, `validation_set.pt`, `test_set.pt`) at every stage, allowing any model to attach at the appropriate preprocessing depth.

---

## 2. Current Specs

### 2.1 Implemented Models (Supervised Binary Classification)

The benchmark currently spans 8 architectures × {augmented, non-augmented} = 22 logged configurations (CNN 1D/2D baselines reproduced from prior VRA work, no augmentation, for cross-study comparability).

| Family | Path | Heads / Variants | Augmentation toggle |
|---|---|---|---|
| LogReg (Lasso) on hand-crafted features | [_03_train/logistic_regression/](_03_train/logistic_regression/) | L1-penalised LR over ~17 statistical/temporal/complexity descriptors per channel (~650 features/trial) | Yes |
| MiniRocket + XGBoost / HistGB | [_03_train/minirocket_and_metadata/](_03_train/minirocket_and_metadata/) | Random convolutional kernels → gradient boosting (with age/sex metadata fusion) | Yes |
| FiLM-CNN | [_03_train/FiLM/](_03_train/FiLM/) | 1D CNN conditioned on (age, sex) via FiLM blocks | Yes |
| InceptionTime | [_03_train/InceptionTime/](_03_train/InceptionTime/) | Deep ensemble of inception modules | Yes |
| ResNet-18 (GASF) | [_03_train/resnet_on_images/](_03_train/resnet_on_images/) | Per-modality PCA→1D → Gramian Angular Summation Field → ImageNet-pretrained ResNet-18 fine-tune | Yes |
| MOMENT foundation model | [_03_train/moment/](_03_train/moment/) | 5 classifier heads on frozen MOMENT-1-large embeddings: MLP, HistGB, LogReg, PCA+TabPFN, TabPFN | Yes |
| CNN 1D / CNN 2D (legacy, prior VRA work) | reproduced in `paper_results/` | Raw waveform / spectrogram baselines | No (intentional, for comparability) |
| MOMENT Anomaly Detection (preliminary) | [_03_train/moment_anomaly_detection/](_03_train/moment_anomaly_detection/) | MAE-style reconstruction fine-tune; MSE as anomaly score | N/A (formulation is unsupervised) |

All supervised heads use:
- Subject-grouped splits enforced at `BoaOpenFaceDataset` construction.
- Per-modality z-normalisation with parameters fit on the training set ([_utils_/dataset_utils.py](_utils_/dataset_utils.py), `compute_normalization_params`).
- Validation-driven threshold selection, learning-rate scheduling, early stopping (per-config in `config_<model>.py`).
- Bootstrap-based CI computation (200 resamples, 70% of test set per resample) in [paper_results/performance_comparison/final_test.ipynb](paper_results/performance_comparison/final_test.ipynb).

### 2.2 Headline Result: No Supervised Model Exceeds Chance

Across all 22 configurations, **Balanced Accuracy is confined to 0.50–0.60** and **all 95% bootstrap CIs cross 0.50**. The CNN 1D/2D baselines that succeeded on the older VRA cohort (7–24 months) **collapse to majority-class prediction** (BA = 0.500, Specificity = 0.000) on this 3–7-month cohort. The three configurations with point AUCs above chance — LogReg+aug (0.568), ResNet-GASF+aug (0.642), MOMENT+TabPFN no-aug (0.591) — are not statistically robust and are not reproducible across augmentation toggles (e.g. MOMENT+TabPFN drops from 0.591 → 0.514 when augmentation is added).

**Model complexity is uncorrelated with performance.** A 650-feature L1 logistic regression matches or exceeds InceptionTime, FiLM-CNN, and the frozen MOMENT foundation model. The only marginal lift comes from 2D image-domain encoding (ResNet-GASF), suggesting that whatever weak signal exists manifests as a brief non-stationary excursion in variance/entropy rather than as a structured, time-locked head-turn morphology with consistent timing.

### 2.3 Manifold Diagnosis (UMAP)

[_02_stats_and_normalization/compute_umap.ipynb](_02_stats_and_normalization/compute_umap.ipynb) and [paper_results/UMAP/](paper_results/UMAP/) document the failure mode:

- **Raw feature space:** Stimulus and Control trials are **topologically inseparable** — complete overlap, no marginal cluster structure.
- **Learned embedding space (best deep model, ResNet-GASF):** Clusters do form, but their structure is **driven by subject identity, not stimulus class**. Stimulus and Control remain interleaved within each subject-shaped cluster.

This is interpreted as an **informational limit**, not a model-capacity failure: the discriminating signal is not present in the post-stimulus video feed at sufficient magnitude under the current passive protocol.

### 2.4 Theoretical Pivot: Supervised Classification → Subject-Specific Anomaly Detection

The paper's revised Discussion and Conclusion (camera-ready) reframe the negative result. The supervised binary formulation is identified as a **structurally limiting assumption** for three converging reasons:

1. **Trial-wise habituation.** With no reinforcement (passive paradigm), reflexive responses extinguish rapidly across the session. Most trials labelled "Stimulus" are behaviourally indistinguishable from "Control" because the infant has already habituated. This injects **structured label noise that is invisible to any supervised learner**, regardless of capacity.
2. **High inter-subject variability.** Both response timing and idiosyncratic baseline movements differ markedly between infants. A population-level decision boundary cannot reconcile these baselines.
3. **Visual-attentional competition.** The central attention-holding video and the ±45° (rather than 90°) speaker geometry suppress both the magnitude and the consistency of the oculomotor head-turn reflex.

The mandated reformulation is to **model each infant's baseline behavioural manifold $M_i$ from pre-stimulus windows and silent controls, and score post-stimulus windows by their deviation from $M_i$.** This converts the output from a binary class into a **continuous novelty score** that aligns with BOA's native decision logic and that a clinician can threshold.

A preliminary AD scaffold already exists at [_03_train/moment_anomaly_detection/](_03_train/moment_anomaly_detection/) (MAE-style MOMENT reconstruction with MSE scoring); it is not yet subject-conditioned and predates the formal pivot.

### 2.5 Known Engineering Debt

- The augmentation pipeline at [_01_preprocessing/augment_time_series.ipynb](_01_preprocessing/augment_time_series.ipynb) is built around the supervised binary objective and around the 0–10 s post-stimulus window only; it has no notion of pre-stimulus baselines, subject-conditioned manifolds, or trial-order semantics.
- Pre-stimulus frames (–2 s → 0 s) are loaded by the QC filter but **discarded before classifier input**. They are not currently exposed as a per-trial baseline tensor in any dataset getter.
- `trial_id` is present on every `OpenFaceInstance` ([DataUtils/OpenFaceInstance.py](DataUtils/OpenFaceInstance.py)) but is currently used only for descriptive bucketing (`categorize_trial_id`). It is **not propagated as a model input or as a sample weight**.
- Five `zNO_*` directories (`zNO_ROCKET`, `zNO_LSTMFCN`, `zNO_ts2vec`, `zNO_MOMENT_tabPFN`, `zNO_test_results_after_training`) remain in the tree as deprecated artefacts.

---

## 3. Future Implementations

The next phase is driven by the AIME 2026 camera-ready Discussion and the project status update of 2026-04-29. It splits cleanly into three workstreams: **(A) Anomaly-detection reformulation**, **(B) Temporal-window ablation with per-trial baselines and trial-order information**, and **(C) Augmentation pipeline overhaul** to support both. All three are prerequisites for the joint BAMBINO + VRA journal submission currently under planning.

### 3.1 Workstream A — Subject-Specific Anomaly Detection Architecture

**Objective.** Replace the supervised binary head with an unsupervised novelty-scoring framework that consumes per-subject baseline windows and emits a continuous anomaly score per post-stimulus window.

**Architectural blueprint.**

1. **Causal Baseline-Manifold Construction per infant $i$ at trial $t$:**
   - **CRITICAL (No Temporal Leakage):** To score a post-stimulus window at trial $t$, the baseline representation $M_{i,t}$ must be built **strictly causally**. It must *only* ingest data that occurred prior to the stimulus onset of trial $t$.
   - **Sources:** (a) the 2 s pre-stimulus segments of trials strictly $< t$, and (b) full silent control trials strictly $< t$.
   - **Implementation:** Implement a cumulative/rolling baseline rather than a static session-wide baseline. Do not aggregate late-session control trials to score early-session stimuli.
2. **Scoring function.** For each post-stimulus window $w$ from infant $i$, emit $s(w \mid M_i) \in \mathbb{R}$ (deviation from $M_i$), thresholded downstream.
3. **Candidate encoder/scorer instantiations** (to be benchmarked against each other):
   - **One-class SVM** on the existing ~650 hand-crafted statistical/temporal/complexity descriptors per trial (immediate baseline; reuses the LogReg feature pipeline at [_03_train/logistic_regression/](_03_train/logistic_regression/)).
   - **Subject-conditioned autoencoder** (reconstruction-error scoring). Extend the existing scaffold at [_03_train/moment_anomaly_detection/](_03_train/moment_anomaly_detection/) to condition on subject embeddings (FiLM-style) so a single model amortises across infants instead of the current per-infant fine-tuning.
   - **Self-supervised contrastive encoder** trained on baseline segments only, using InfoNCE / SimCLR-style positive pairs drawn from within-subject baselines and negative pairs across subjects; score = distance to per-subject baseline centroid in the learned space.
4. **Outputs and evaluation.** Each method emits a per-trial novelty score. Evaluation switches from `balanced_accuracy` / `roc_auc` over hard labels to:
   - Per-trial AUROC of novelty score vs. nominal Stimulus/Control label (anchored to **early-session trials only** to control for habituation, see §3.2).
   - Per-infant pass/fail at clinician-tunable thresholds, calibrated against the BOA decision logic.
   - 95% bootstrap CIs (existing infrastructure in [paper_results/performance_comparison/final_test.ipynb](paper_results/performance_comparison/final_test.ipynb) is reusable).

**Module layout.** Create `_03_train/anomaly_detection/` containing:
- `config_ad_subject.py` — encoder family, baseline-window length, scoring function, subject-conditioning strategy.
- `manifold_utils.py` — `BaselineCorpusBuilder`, `SubjectConditionedScorer`, `build_baseline_corpus(dataset, infant_id)`.
- `main_ocsvm.ipynb`, `main_subject_ae.ipynb`, `main_contrastive.ipynb` — one notebook per encoder family.
- `eval_ad.py` — habituation-aware evaluation harness (see §3.2).

The existing [_03_train/moment_anomaly_detection/](_03_train/moment_anomaly_detection/) becomes one entry under this umbrella (rename `main_ad.ipynb` → `main_moment_ad.ipynb` and migrate).

### 3.2 Workstream B — Temporal Window Ablations: 2 s Pre-Stimulus Baseline + Trial-Order Information

**Objective.** Decouple "stimulus response" from "infant baseline" at the per-trial level, and explicitly model session-level habituation as either a covariate or a sample-weight prior.

#### 3.2.1 Per-trial 2 s pre-stimulus baseline tensor

**Engineering tasks.**

1. **Expose the pre-stimulus segment in the dataset getter.** Modify [DataUtils/BoaOpenFaceDataset.py](DataUtils/BoaOpenFaceDataset.py) `__getitem__` so that each sample returns:
   ```
   (x_dict_post, x_dict_pre, y_label, extras)
   ```
   where `x_dict_pre` is the (–2 s → 0 s) window (50 frames @ 25 fps) per modality, and `x_dict_post` is the existing 0–10 s window (250 frames). Mirror the change in `OpenFaceDataset.__getitem__` to keep the parent consistent.
2. **Propagate the QC contract.** The current confidence filter at [DataUtils/OpenFaceDataset.py](DataUtils/OpenFaceDataset.py) drops trials where confidence ≤50% over the **first 52 frames**. That window currently coincides with the start of the post-stimulus segment; redefine it to the **pre-stimulus 50 frames** so QC is anchored to baseline integrity. Re-run preprocessing → all five data tiers must be regenerated.
3. **Baseline normalisation modes** (configurable, ablation axis):
   - `mode = "global"` — current behaviour: per-modality z-norm with population statistics from the training set.
   - `mode = "per_trial"` — subtract per-trial pre-stimulus mean, divide by per-trial pre-stimulus std (channel-wise). Implement in [_utils_/dataset_utils.py](_utils_/dataset_utils.py) as `apply_per_trial_baseline_normalization(dataset)`.
   - `mode = "per_subject"` — z-norm with statistics aggregated across all baseline windows of the same infant.
4. **Downstream consumer updates.** For each active model family, add a `baseline_mode` switch in the model's `config_*.py`:
   - LogReg / MiniRocket / GASF: re-extract features from the baseline-normalised post-stimulus window.
   - FiLM-CNN, InceptionTime: optionally concatenate the pre-stimulus window as additional context channels OR pass aggregated baseline statistics through the FiLM conditioning vector.
   - MOMENT: feed both windows independently and contrast embeddings (post-stimulus embedding − baseline embedding) before the classifier head.
   - Anomaly detection (Workstream A): the pre-stimulus window IS the primary input for $M_i$ construction.

#### 3.2.2 Trial-order information

**Hypothesis.** Response reliability decreases monotonically with trial number due to habituation; a trial late in the session carries less informative label content than an early one. `trial_id` is already on every instance ([DataUtils/OpenFaceInstance.py](DataUtils/OpenFaceInstance.py)) and must now influence training.

**Two mutually exclusive instantiations, to be A/B'd:**

1. **Explicit input feature.**
   - Append a normalised `trial_index_in_session ∈ [0, 1]` scalar (and optionally `log(1 + trial_index)`) to the metadata vector that already carries (age, sex) for FiLM-CNN, MiniRocket+meta, MOMENT heads, and LogReg.
   - For sequence models without a metadata path, broadcast the scalar across the time dimension as an additional input channel (38 → 39).
2. **Trial-decaying sample weights.**
   - Define `w(trial_idx) = exp(-λ · trial_idx)` (or a piecewise linear / sigmoid schedule), with `λ` selected via validation-set BA on the **first ⅓ of trials only** (where response is expected to be most reliable).
   - Plug into:
     - sklearn / XGBoost / HistGB heads via `sample_weight=` (LogReg, MiniRocket, MOMENT-HistGB / MOMENT-LogReg).
     - PyTorch trainers (FiLM-CNN, InceptionTime, ResNet-GASF, MOMENT-MLP) by replacing the existing `WeightedRandomSampler` from [_utils_/models_utils.py](_utils_/models_utils.py) `get_balanced_sampler()` with a sampler that composes class-balance weights × habituation-decay weights, OR by passing per-sample weights into the loss reduction.
   - Extend [_utils_/models_utils.py](_utils_/models_utils.py) with `get_habituation_aware_sampler(dataset, decay_lambda, balance=True)`.
   - **CRITICAL CONSTRAINT:** Habituation-decay weights apply **ONLY to supervised post-stimulus models** (which evaluate the degrading response). For Anomaly Detection baseline construction (Workstream A), all causally valid pre-stimulus and control windows must maintain equal weight (e.g., 1.0) regardless of their position in the session, as late-session fatigue is a valid baseline state, not an error.

**Habituation-aware evaluation.** Existing test-set metrics are blind to where in the session a trial sits. Add to `eval_ad.py` (and reuse for supervised models): trial-bucketed metric reporting (early ⅓ / mid ⅓ / late ⅓ of session), so a model that only detects fresh responses is correctly credited rather than averaged into the habituated noise.

#### 3.2.3 Ablation matrix

Run as a single Optuna study group (extend [_03_train/optuna_analysis.py](_03_train/optuna_analysis.py)) over the cross-product:

| Axis | Levels |
|---|---|
| Baseline normalisation mode | `global`, `per_trial`, `per_subject` |
| Pre-stimulus window length | 1 s, 2 s, 4 s if available |
| Trial-order encoding | none, scalar feature, decay-weighted samples, both |
| Model family | LogReg, MiniRocket, FiLM-CNN, MOMENT-TabPFN, ResNet-GASF, anomaly-detection (per Workstream A) |

Persist studies under `optuna/optuna_studies/baseline_ablation_<model>.db` and aggregate via the existing `optuna_analysis.py` infrastructure.

### 3.3 Workstream C — Augmentation Pipeline Overhaul

The current pipeline at [_01_preprocessing/augment_time_series.ipynb](_01_preprocessing/augment_time_series.ipynb) implements 6 primitives (jitter σ=0.05, scaling 0.8–1.2, pad-shift ±25 frames, time-warp max-warp 0.2, time-masking 5–20 frames, with a protective constraint preserving the first 0.6 s of stimulus onset). It is built for the supervised binary objective on the post-stimulus window. It must be re-architected for the new framework.

**Required changes.**

1. **Independent Temporal Augmentation (Preventing Stimulus Bleed).** - Spatial and magnitude augmentations (e.g., scale, jitter, channel dropout) must share the same random state/seed across the paired pre-stimulus and post-stimulus windows to maintain spatial continuity. 
   - **Temporal augmentations (time-warping, shifting) must NEVER cross the $t=0$ boundary.** Apply temporal transforms independently to the pre-stimulus (–2 s → 0 s) tensor and the post-stimulus (0 s → 10 s) tensor. **Do not** concatenate the 12 s window before temporal warping, as this would shift the physical stimulus onset away from the split index, destroying the AD premise.
2. **Drop the stimulus-onset protective constraint.** It encoded the assumption that t=0 carries a sharp time-locked transient. The current data refute this assumption (Section 2.2). For supervised stimulus-response models, replace it with a **soft constraint**: time-warp grids must remain monotonic and bounded such that the temporal location of any frame shifts by at most ±100 ms.
3. **Anomaly-detection augmentations apply to the baseline corpus only.** When training a subject-conditioned baseline model (Workstream A), augmentations are applied **exclusively to baseline windows** (pre-stimulus + control trials). Post-stimulus windows are scored, never augmented. Add a `purpose: {"supervised", "ad_baseline"}` flag to the augmentation entry point; route accordingly.
4. **Habituation-Preserving Augmentation.** - Augmented copies must inherit the `trial_id` of their source trial verbatim.
   - When trial-decaying sample weights are active (for supervised models only), augmented copies receive the same weight as their source (scaled by `1/n_aug`).
   - For AD baseline generation, augmented pre-stimulus/control windows maintain a flat weight.
   - The augmentation report at `data/augmented/training_set_augmentation_report.csv` must include the source `trial_id` and the inherited weight.
5. **Subject-stratified augmentation budgets.** Replace the current global `ORIGINAL_POS_PER_TRIAL=4 / ORIGINAL_NEG_PER_TRIAL=1` heuristic with a per-subject budget that equalises augmented sample count **across infants** rather than across the population, preventing high-trial-count infants from dominating the training distribution.
6. **New primitives motivated by anomaly-detection.** Add to the primitive set:
   - **Channel dropout** (zero a random subset of the 38 channels per epoch) — forces baseline models to learn redundant representations of the manifold.
   - **Magnitude warping** (smooth multiplicative envelope) — distinct from time-warp; targets the variance/entropy excursion that the paper identifies as the most plausible signal carrier.
7. **Deterministic, versioned augmentation.** Every augmented copy must store a deterministic seed and primitive composition string in its metadata, making the augmented dataset reproducible from `data/normalized/` without re-running the notebook.

### 3.4 Out-of-Scope (Upstream Protocol Recommendations)

These are not engineering tasks for this repository but are documented for alignment with the Manchester clinical team and for the journal-paper Discussion:

- Replace the central attention-holding video with a **static neutral image immediately prior to stimulus onset** to release peripheral attention (mitigates the competition effect at 3–7 months).
- Move loudspeakers to **±90° azimuth** with supportive infant seating to maximise head-turn magnitude and SNR.
- **Vary acoustic content across trials within a session** (rather than repeating the same stimulus class) to counteract habituation at the protocol level. Without this, no algorithmic intervention can fully recover the lost signal in late-session trials.

### 3.5 Sequencing & Acceptance Criteria

| Order | Task | Done when |
|---|---|---|
| B.1 | Pre-stimulus window exposed in `BoaOpenFaceDataset.__getitem__`; QC anchored to baseline | All five data tiers regenerate without errors; existing supervised models train unchanged using `x_dict_post` only |
| C.1–C.4 | Augmentation overhaul: joint pre/post, AD-baseline-only mode, habituation-preserving weights | Augmentation report includes `trial_id`, `weight`, `purpose`; reproducibility verified from a fixed seed |
| B.2 | Trial-order encoded (both as scalar feature and as decay weights), habituation-bucketed evaluation | Trial-bucketed metrics reported in `final_test.ipynb`; LogReg+aug result reproduced under both encodings |
| B.3 | Baseline-normalisation ablation matrix run across active models | Optuna studies persisted under `optuna/optuna_studies/baseline_ablation_*.db`; aggregated table in `paper_results/` |
| A.1 | One-class SVM on existing 650-feature pipeline (cheapest AD baseline) | Per-subject novelty AUROC + 95% CI reported on test set |
| A.2 | Subject-conditioned autoencoder (extension of `moment_anomaly_detection/`) | Single model amortising across infants; novelty AUROC ≥ one-class SVM baseline |
| A.3 | Self-supervised contrastive encoder trained on baseline segments | Novelty AUROC ≥ subject-conditioned AE on early-session bucket |

A model from Workstream A is considered **promising** (i.e. worth pursuing for the journal submission) only if its bootstrap 95% CI for early-session-bucket AUROC strictly excludes 0.50 — the bar that the supervised benchmark failed to clear.
