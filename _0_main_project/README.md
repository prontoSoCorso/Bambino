# BAMBINO — Production Pipeline

Modular Python rewrite of the legacy notebook-driven prototype, aligned with
[../PROJECT_STATE.md](../PROJECT_STATE.md). All exploratory `.ipynb` logic in
the parent repo has been collapsed into a single, dataclass-configured,
PyTorch-Lightning entrypoint.

## Layout

```
_0_main_project/
├── main.py                       # Single training/eval entrypoint
├── requirements.txt
├── data/full/raw/                # NEW full raw dataset root
├── results/                      # TB + CSV logs, checkpoints, metrics.json
├── src/
│   ├── configs/                  # Dataclasses: base, data, augmentation, trainer, models
│   ├── data/                     # Dataset, DataModule, normalization, augmentation, features
│   ├── models/                   # LogReg, InceptionTime, MOMENT, Anomaly Detector
│   └── utils/                    # Plotting (palette-locked), metrics, samplers, seeding
└── tests/                        # pytest suite — causal baseline + temporal augmentation
```

## Quickstart

```bash
pip install -r requirements.txt

# Supervised baseline (LogReg on hand-crafted features)
python main.py --model logreg --run-id logreg_v1

# Deep TS classifier with augmentation + per-trial baseline norm
python main.py --model inception_time --run-id incept_v1 \
        --baseline-norm-mode per_trial --use-pre-stim-context

# Subject-conditioned anomaly detector (the active research direction)
python main.py --model anomaly_detector --run-id ad_causal_v1 \
        --baseline-norm-mode per_subject

# Run the test suite
pytest tests/ -v
```

## Architectural contracts

All four are enforced in code and verified in `tests/`.

1. **Subject-independent splits.** No infant in more than one of
   `{train, val, test}`. See [src/data/splits.py](src/data/splits.py) and
   `tests/test_splits.py`.

2. **Causal baseline.** To score a post-stim window of trial `t` for infant
   `i`, only data with `(pt_id == i AND trial_id < t)` may be ingested into
   the baseline manifold. See `build_causal_baseline()` in
   [src/models/anomaly_detector.py](src/models/anomaly_detector.py) and
   `tests/test_causal_baseline.py`.

3. **Temporal augmentation cannot cross t=0.** Time-warp / pad-shift / mask
   are applied INDEPENDENTLY to the pre-stimulus and post-stimulus halves of
   each trial. Spatial / magnitude augs share an RNG so channel realisations
   match across the boundary. See [src/data/augmentation.py](src/data/augmentation.py)
   and `tests/test_temporal_augmentation.py`.

4. **Habituation weights are model-class-specific.** Decay weights apply
   ONLY to supervised post-stimulus models. AD baseline construction uses
   FLAT weights (1.0) regardless of trial position — late-session fatigue is
   a valid baseline state. See [src/utils/samplers.py](src/utils/samplers.py).

## Plot palette (enforced)

| Role | Hex | Name |
|---|---|---|
| Primary / positive class | `#882255` | Wine |
| Secondary / negative class | `#4477AA` | Blue |
| Tertiary A | `#44AA99` | Teal |
| Tertiary B | `#DDCC77` | Sand |
| Accent | `#CC6677` | Rose |
| Reference / background | `#98A4B0` | Grey |

All plotting is centralised in [src/utils/plotting.py](src/utils/plotting.py).
A test (`tests/test_palette.py`) pins the constants to these values.

## Logging

- **TensorBoard**: `results/<run_id>/tb_logs/`
- **CSV**: `results/<run_id>/csv_logs/`
- **Checkpoints**: `results/<run_id>/checkpoints/`
- **Final metrics (LogReg)**: `results/<run_id>/metrics.json`

No cloud loggers (W&B, MLflow) — explicitly per the refactor brief.
