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

## Reference

This project is motivated by the following paper:

> A LaGrassa, Z Huang, D Berenson (2025). *Planning-Query-Guided Model Generation for Model-Based Deformable Object Manipulation*. arXiv preprint arXiv. https://arxiv.org/abs/2508.19199

Paper author: Prof. Alex LaGrassa, Stevens Institute of Technology (alagrass@stevens.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
