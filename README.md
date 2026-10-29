# Cauliflower Classification

Research code and reproducibility records for **Comparative Evaluation of CNNs and Transformers for Cauliflower Disease Classification Using an Audited and Harmonized Dataset**.

This project compares eight pretrained image classifiers using an audited, harmonized dataset and a separate five-fold source-held-out evaluation. It also includes statistical comparisons, inference-efficiency measurements and Grad-CAM analysis.

## Study overview

- Five public repositories; 5,394 downloaded images.
- After label harmonization, duplicate review and exclusions: 3,914 retained images across 14 operational classes.
- Main fixed split: 2,756 training, 579 validation and 579 test images; split seed 42.
- Offline augmentation increases the training set to 7,191 images by bringing each class to at least 500 images without downsampling larger classes.
- Main benchmark: 54 model–experiment conditions, each trained with seeds 42, 43 and 44: 162 runs.
- Source-held-out evaluation: four models × five folds × three seeds: 60 runs at 224 × 224 pixels, using the three or four eligible shared classes in each fold.

The labels are harmonized repository labels; the audit does not provide independent pathological confirmation.

## Original datasets

Download the original datasets from their providers. Preserve the source identifiers below when reconstructing the dataset.

| Source | Dataset | Version / link |
| --- | --- | --- |
| S1 | Cauliflower Diseases Identification Image Dataset | Version 2: https://doi.org/10.17632/c2hrwntzgh.2 |
| S2 | Cauliflower Dataset | Version 1: https://doi.org/10.17632/x26px3xnmy.1 |
| S3 | Cauli-Net (Data on Cauliflower Diseases and Healthy) | Version 1: https://doi.org/10.17632/y6hk67rhs5.1 |
| S4 | VegNet | Version 3: https://doi.org/10.17632/t5sssfgn2v.3 |
| S5 | Cauliflower Image Dataset | https://www.kaggle.com/datasets/sharifashik/cauliflower-image-dataset |

Raw and augmented image folders are excluded from the GitHub package to keep it small. Dataset access, attribution and redistribution remain subject to each provider's terms. Use the retained manifest and recorded source paths and hashes to identify the images used in this study; do not assume that a newer repository download is identical.

## Repository organization

The publication package groups existing files as follows. Names within copied report folders are preserved. No separate `dataset_sources.csv` or `eligibility_reports` folder is required.

```text
CauliflowerV2/
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/                         # Final scripts used for the study
├── records/                         # Audit, harmonization and split records
│   └── source_held_out/
│       ├── retained_manifest_input.csv
│       ├── protocol.json
│       ├── eligibility_summary.csv
│       └── ... five fold folders containing reports only
└── results/
    ├── main/                        # Final baseline–E6 summaries
    ├── source_held_out/              # Five-fold results and run records
    ├── test_prediction/              # Prediction collection and metadata
    ├── efficiency_log/               # Efficiency collection and metadata
    └── analysis/
        ├── statistics/
        ├── efficiency/
        └── gradcam/
```

`records/` contains the existing audit reports, manually reviewed exact- and near-duplicate decisions, source-label mapping, retained manifest, exclusion records, main split reports and offline-augmentation reports. In each source-held-out fold, retain `reports/` and `offline_augmentation_reports/`; omit the image directories. The reviewed audit decisions are inputs to reproduction and must be retained alongside the code.

The publication layout is separate from the original execution layout. Scripts may expect `Dataset/`, `Results/` or `Result/`, or recorded Windows paths. Keep those local runtime directories outside version control and pass explicit path arguments where supported. Check `--help` before running a copied script. Moving files into this publication layout does not automatically update saved paths or script defaults.

## Environment

Main experiments used Windows 11, Python 3.12.10 and an NVIDIA GeForce RTX 5060 Ti with 16 GB memory. Install the study dependencies from the supplied `requirements.txt`, including timm 1.0.30. Use the appropriate PyTorch/CUDA installation for the recorded environment and retain the exported environment records with the results.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The saved statistical analysis was produced separately with Python 3.12.14 and NumPy 2.3.5; see `analysis_protocol.json`. Hardware, library versions and deterministic settings can affect timing and numerical results.

## Main experimental settings

| Experiment | Input size | Offline augmentation | Online augmentation |
| --- | --- | --- | --- |
| Baseline | 224 × 224 | No | No |
| E1 | 224 × 224 | Yes | No |
| E2 | 224 × 224 | Yes | Yes |
| E3 | 300 × 300 | Yes | No |
| E4 | 300 × 300 | Yes | Yes |
| E5 | 384 × 384 | Yes | No |
| E6 | 384 × 384 | Yes | Yes |

Models: `resnet50`, `densenet121`, `mobilenet_v3_large`, `efficientnet_v2_s`, `xception`, `convnext_tiny`, `vit_base_patch16_224` and `twins_svt_base`.

ViT is not evaluated in E3 or E4. Its identifier names the pretrained 224-pixel checkpoint; its actual E5/E6 input is 384 × 384, with positional-embedding interpolation. It does not use a separately pretrained 384-pixel checkpoint.

EfficientNetV2-S uses BF16 automatic mixed precision throughout baseline–E6; the other models use FP16 automatic mixed precision. Inference-efficiency benchmarks and Grad-CAM use FP32. Use only the final BF16 EfficientNet runs and the final E5 reruns when collecting results.

Training uses batch size 32, AdamW, learning rate 0.0001, weight decay 0.0001, label smoothing 0.1 and a maximum of 100 epochs. Checkpoints are selected using validation accuracy only, with early-stopping patience 10. Run-specific configurations and histories are the detailed records of each training run.

## Reproduce dataset preparation and training

1. Download the five source datasets and retain their source directory identities.
2. Run `01_audit_downloads.py`. Apply the published manually reviewed duplicate and label-conflict decisions. Do not replace the review decisions with automatic similarity thresholds alone.
3. Run the final cleaning/harmonization script using the source-label mapping. Check the retained manifest and the 3,914-image total.
4. Run the main split script with split seed 42. Verify the published assignments and the 2,756/579/579 counts before augmentation.
5. Run `04_augment_train.py` using the saved policy and seed. Verify the augmentation manifest, parent-image linkage and 7,191-image training total. Validation and test images are not augmented.
6. Run `05_train_experiments.py` for the seven non-EfficientNet models and `05_train_experiments_amp_memory_cleanup.py` with `--amp-dtype bfloat16` for EfficientNetV2-S. Use seeds 42, 43 and 44 and the experiment settings above. Omit ViT from E3/E4.
7. Combine only final completed runs using the report-combination script. Retain class mappings, configuration files, histories, checkpoint-selection records, predictions and efficiency logs.
8. Prepare source-held-out folds using `09_prepare_source_held_out.py`, then augment their training partitions using `10_augment_source_held_out.py` and train using `11_train_source_held_out.py`.

The source-held-out models are ResNet50, MobileNetV3-Large, ViT-Base-Patch16 and Twins-SVT-Base. Training uses offline augmentation to a minimum of 500 images per class plus online augmentation. Each fold has a fixed 85%/15% development split, split seed 42, and a source-exclusive held-out test set. Photographs associated with the held-out source and another source are excluded from that fold. Report all five folds, including limited class supports.

For preparation and training scripts, use their final published filenames and `--help` to supply the local input/output paths. The original cleaning and main-splitting filenames are preserved rather than renamed here. Before releasing this repository, record the exact commands used for these steps alongside the corresponding scripts or run configurations.

## Reproduce statistical analysis from saved predictions

`12_collect_test_predictions.py` collects the final predictions. Preserve `collection_manifest.csv`, `collection_summary.json`, each experiment's `class_to_index.json`, and the 162 model/seed `test_predictions.csv` files. Package that collection as `results/test_prediction.zip`, retaining its internal directory structure.

```bash
python scripts/13_analyze_test_predictions.py --input results/test_prediction.zip --output results/analysis/statistics_reproduced --bootstrap 10000 --permutations 50000 --seed 20261025
```

The output directory must not already exist. Expected outputs:

- `confidence_intervals.csv`: 108 accuracy/macro-F1 interval records for 54 conditions.
- `paired_comparisons.csv`: 118 tests, with Holm adjustment across the complete family.
- `class_supports.csv`: supports of the 14 test classes.
- `analysis_protocol.json`: validation checks, analysis settings, software and limitations.

The bootstrap is stratified by true class and keeps the three seed predictions together for each sampled image. Intervals are conditional on the fitted runs and the fixed test-class supports; they do not quantify uncertainty over new training runs or new sources. These outputs concern the main benchmark, not the source-held-out evaluation.

## Reproduce efficiency summaries

`07_collect_efficiency_logs.py` collects the final `efficiency.json` files. Retain the collected `results/efficiency_log/` tree and both collection metadata files.

```bash
python scripts/08_summarize_efficiency.py --input results/efficiency_log --output results/analysis/efficiency_reproduced --apply
```

Use a new output directory. Expected outputs are `efficiency_per_run.csv`, `efficiency_summary.csv`, `efficiency_summary_long.csv`, `operator_coverage.csv`, `unsupported_operators.csv`, `audit_flags.csv` and `summary.json`.

This step summarizes saved measurements; it does not run new GPU workloads. The training runner performs the actual benchmarks. Measurements use FP32, 20 warm-up iterations and 100 timed iterations, with batch sizes 1 and 32. Latency is GPU forward-pass time and excludes data loading, preprocessing and host-to-device transfer. Reported seed SDs describe variation across three seed-level measurements. Memory is peak allocated PyTorch memory in MiB, not total device usage. Computation estimates are partial because unsupported operations are not added to the totals; GMACs must not be described as complete model operation counts.

## Reproduce Grad-CAM figures

Grad-CAM uses the four source-held-out model architectures, but their **main E6 seed-42 checkpoints**, in FP32. Preserve the full `gradcam_E6_seed42/` output tree, including `protocol.json`, `selection_manifest.csv`, `selection_counts.csv`, `cam_audit.csv`, per-image metadata, numeric maps and rendered outputs.

Rebuild the main comparison and additional panels from saved outputs:

```bash
python scripts/15_build_gradcam_comparison.py --input results/analysis/gradcam/gradcam_E6_seed42 --output results/analysis/gradcam/comparison_reproduced --correct-rows 2 --error-rows 2 --selection-seed 42 --apply
```

To regenerate the maps themselves, run `14_generate_gradcam.py` with the original E6 run root, reconstructed split images and the four `best.pt` checkpoints. The run root must include `experiment.json`, `class_to_index.json` and each model/seed's `config.json`, `status.json`, `test_evaluation.json` and `test_predictions.csv`. The script verifies checkpoint hashes. The README does not imply that checkpoints or raw images are included; regeneration from weights requires obtaining them or rerunning training.

The main figure displays two common-correct examples and two examples with an error from at least one model, selected from 20 candidates. Keep `main_figure_rows.csv` and `figure_protocol.json` with the figure. Maps target predicted-class logits; additional ground-truth maps are generated for errors. Maps are normalized individually and are qualitative attributions, not validated lesion segmentations.

## Availability and interpretation

Keep scripts, manually reviewed decisions, manifests, split assignments and analysis outputs under version control. Exclude raw/augmented image directories, caches, failed runs and superseded runs. Large checkpoints may be distributed separately with documented hashes and links. A final public release should contain the files described here; this README alone does not establish their availability.

Reported mixed-source performance should be interpreted together with the source-held-out results. Different folds have different class sets and supports. Nonsignificant paired comparisons do not establish equivalence. Partial computation estimates and desktop GPU measurements do not establish deployment performance on other devices.

## Citation

Please cite the manuscript title above and the original dataset providers when using this work. Publication metadata and a release DOI can be added after they are assigned; none is claimed here.

## License

The MIT license in `LICENSE` applies to original software contributed by this project's copyright holders. Third-party datasets, photographs, pretrained weights, dependencies and incorporated third-party code retain their respective licenses. The MIT license does not grant rights to those materials or automatically license the manuscript and dataset-derived images.
