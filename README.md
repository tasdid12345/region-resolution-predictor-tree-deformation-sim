# Predicting per-region model resolution for a simulated deformable tree

## Motivation
"Planning-Query-Guided Model Generation for Model-Based Deformable Object Manipulation" (2025) proposes a diffusion-based generator that reads start and goal pointclouds and predicts a resolution for each region of a graph-neural-network dynamics model, so that planning is faster with little loss in task performance. This experiment asks whether a simpler, feature-based predictor can do the resolution-prediction step better than a generic neural baseline.

## Question
Can a tuned gradient-boosting predictor on per-region start-to-goal deformation features beat a standard MLP on raw start and goal pointclouds at predicting per-region model resolution for a simulated deformable tree?

## Dataset
Simulated in the script with numpy (no download). It is NOT the paper's data, which is not public. Random 2D branching trees with 8 regions of about 200 points each; goal pointclouds are smooth random bends of the start. Each region's resolution (1, 2 or 4 control points) is the smallest one whose piecewise-linear approximation error of the displacement stays under a threshold. Training labels are randomly flipped to mimic noisy supervision; test labels are clean. Fixed seed 42, 2,000 train and 500 test trees.

## Baseline vs improved
- Baseline: scikit-learn MLP on flattened start and goal pointclouds, predicting all region resolutions (stand-in for the diffusion generator, which cannot be built with the allowed packages; not the paper's method).
- Improved: one LightGBM classifier shared across regions on local deformation features (displacement mean and spread, curvature, branch length, region id, etc.).
- Both are tuned by cross-validation on the training set only, with the same budget, and evaluated once on the test set with seeds 42, 43, 44. Main metric: per-region accuracy; secondary: under-resolved share and mean predicted resolution.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to results.json (and a chart to figures/).

## Results (latest run)

On simulated 2D deformable-tree data (500 clean test labels), a standard MLP on raw start and goal pointclouds, standing in for your diffusion generator, reached 41% per-region resolution accuracy, while a tuned LightGBM on local deformation features reached 94% on the same test set.

| Metric | Value |
|---|---|
| baseline_score | 0.4078 |
| baseline_score_std | 0.004596 |
| baseline_under_resolved_share | 0.3079 |
| baseline_under_resolved_share_std | 0.0156 |
| baseline_mean_resolution | 2.307 |
| baseline_mean_resolution_std | 0.05351 |
| improved_score | 0.9413 |
| improved_score_std | 0.0002041 |
| improved_under_resolved_share | 0.02667 |
| improved_under_resolved_share_std | 0.0002357 |
| improved_mean_resolution | 2.399 |
| improved_mean_resolution_std | 0.0006236 |
| improvement | 0.5335 |
| test_true_mean_resolution | 2.389 |
| train_label_noise_rate | 0.2579 |
| majority_class_test_accuracy | 0.4903 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data, not the paper's; labels come from a geometric approximation-error rule, not closed-loop planning performance, so results do not test the paper's speed/performance claim. Features such as displacement curvature are close to the labelling rule, which favours the feature model. The diffusion generator is replaced by an MLP. Threshold is the 70th percentile of linear-fit error on train trees. Seeds only affect model training; the simulated data are fixed. Tuning uses noisy train labels. Under-resolved share and mean resolution are proxies only.

## Reference

This project is motivated by the following paper:

> A LaGrassa, Z Huang, D Berenson (2025). *Planning-Query-Guided Model Generation for Model-Based Deformable Object Manipulation*. arXiv preprint arXiv. https://arxiv.org/abs/2508.19199

Paper author: Prof. Alex LaGrassa, Stevens Institute of Technology (alagrass@stevens.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
