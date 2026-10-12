"""Simulated deformable-tree resolution prediction: MLP on raw pointclouds vs tuned LightGBM on region deformation features."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold, ParameterGrid, ParameterSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, N_TREES, N_TEST, R, P = 42, 2500, 500, 8, 200  # R regions, P points per region
RES = np.array([1, 2, 4])                              # candidate control-point counts
NOISE = 0.25                                           # train label flip probability
rng = np.random.default_rng(SEED)
t = np.linspace(0, 1, P)

# ---------- simulation ----------
def make_tree():
    pts = np.zeros((R, P, 2))
    for i in range(R):
        if i == 0:
            start, ang = np.zeros(2), np.pi / 2 + rng.normal(0, 0.3)
            length = rng.uniform(1.5, 2.5)
        else:
            par = rng.integers(0, i)
            start = pts[par, rng.integers(20, 180)]
            d = pts[par, -1] - pts[par, 0]
            ang = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)
            length = rng.uniform(0.8, 1.8)
        a = ang + rng.normal(0, 0.6) * t                       # gently curved branch
        steps = length / P * np.stack([np.cos(a), np.sin(a)], 1)
        pts[i] = start + np.cumsum(steps, 0) - steps[0]
    return pts

def bend(pts):
    """Smooth random bend: sum of 3 sinusoidal displacement waves."""
    x = pts.reshape(-1, 2)
    th, amp = rng.uniform(0, 2 * np.pi, 3), rng.uniform(0.1, 0.6)
    w = rng.uniform(0.5, 3.0, 3)[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
    ud = rng.uniform(0, 2 * np.pi, 3)
    A = amp * rng.uniform(0.3, 1, 3)[:, None] * np.stack([np.cos(ud), np.sin(ud)], 1)
    d = np.sin(x @ w.T + rng.uniform(0, 2 * np.pi, 3)) @ A
    return pts + d.reshape(pts.shape)

print("Simulating trees ...")
S = np.stack([make_tree() for _ in range(N_TREES)])
G = np.stack([bend(s) for s in S])
D = G - S                                                      # (N, R, P, 2) displacements

# piecewise-linear (hat basis) least-squares projection residual per resolution
def hat_proj(k):
    B = np.ones((P, 1)) if k == 1 else np.maximum(0, 1 - np.abs(t[:, None] - np.linspace(0, 1, k)) * (k - 1))
    return B @ np.linalg.pinv(B)
def approx_err(k):
    res = D - np.einsum("ij,nrjc->nric", hat_proj(k), D)
    return np.sqrt((res ** 2).sum(-1).mean(-1))               # (N, R) RMS error
E = {k: approx_err(k) for k in RES}

# ---------- split once (stratified on tree-level difficulty bins) ----------
strat = pd.qcut(E[2].mean(1), 4, labels=False)
tr, te = train_test_split(np.arange(N_TREES), test_size=N_TEST, random_state=SEED, stratify=strat)

# resolution threshold from TRAIN trees only; label = smallest resolution under threshold
thr = np.quantile(E[2][tr], 0.7)
Yc = np.full((N_TREES, R), 2)
Yc[E[2] < thr] = 1
Yc[E[1] < thr] = 0
print("Clean class shares (train):", np.bincount(Yc[tr].ravel(), minlength=3) / Yc[tr].size)
if (np.bincount(Yc.ravel(), minlength=3) == 0).any():
    sys.exit("degenerate label distribution")

# noisy training labels: flip to an adjacent resolution; test labels stay clean
Yn = Yc.copy()
flip = rng.random(Yc.shape) < NOISE
step = np.where(Yc == 0, 1, np.where(Yc == 2, -1, rng.choice([-1, 1], Yc.shape)))
Yn[flip] += step[flip]

# ---------- features ----------
RAW = np.concatenate([S.reshape(N_TREES, -1), G.reshape(N_TREES, -1)], 1)   # baseline input
prof = D.reshape(N_TREES, R, 8, P // 8, 2).mean(3)                           # coarse displacement profile
nrm = lambda a: np.linalg.norm(a, axis=-1)
blen = nrm(np.diff(S, axis=2)).sum(-1)
glen = nrm(np.diff(G, axis=2)).sum(-1)
FEAT = np.stack([
    nrm(D.mean(2)),                                          # displacement mean
    np.sqrt(D.var(2).sum(-1)),                               # displacement spread
    nrm(prof[:, :, 2:] - 2 * prof[:, :, 1:-1] + prof[:, :, :-2]).mean(-1),  # curvature of displacement
    nrm(prof[:, :, -1] - prof[:, :, 0]),                     # end-to-end displacement change
    blen, glen / blen,                                       # branch length, stretch ratio
    np.broadcast_to(np.arange(R), (N_TREES, R)),             # region id
], -1).astype(float)
F = FEAT.shape[-1]

# ---------- models: fit(idx, labels, cfg, seed) -> predict(idx) -> (n, R) classes ----------
def fit_mlp(idx, Y, cfg, seed):
    ind = np.eye(3)[Y].reshape(len(idx), R * 3)             # multilabel one-hot indicator
    m = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=cfg["h"], alpha=cfg["a"], early_stopping=True, max_iter=200, random_state=seed)
    ).fit(RAW[idx], ind)
    return lambda j: m.predict_proba(RAW[j]).reshape(len(j), R, 3).argmax(2)

def fit_lgb(idx, Y, cfg, seed):
    m = LGBMClassifier(**cfg, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=2
                       ).fit(FEAT[idx].reshape(-1, F), Y.reshape(-1))   # one model shared by regions
    return lambda j: m.predict(FEAT[j].reshape(-1, F)).reshape(len(j), R)

def cv_select(fit, cfgs):
    """3-fold CV on train trees (noisy labels, the only ones available); returns best config."""
    scores = []
    for c in cfgs:
        accs = []
        for a, b in KFold(3, shuffle=True, random_state=SEED).split(tr):
            pred = fit(tr[a], Yn[tr[a]], c, SEED)(tr[b])
            accs.append((pred == Yn[tr[b]]).mean())
        scores.append(np.mean(accs))
        print(f"  cfg {c} cv_acc={scores[-1]:.4f}")
    return cfgs[int(np.argmax(scores))]

def evaluate(pred, true):
    return {"acc": float((pred == true).mean()),
            "under": float((pred < true).mean()),           # under-resolved share
            "mean_res": float(RES[pred].mean())}            # speed proxy (fewer control points = faster)

# ---------- tuning (equal budget: 6 configs x 3 folds each) ----------
mlp_cfgs = list(ParameterGrid({"h": [(64,), (128,), (256,)], "a": [1e-4, 1e-2]}))
lgb_cfgs = list(ParameterSampler({
    "n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1], "num_leaves": [7, 15, 31],
    "min_child_samples": [20, 50, 100], "subsample": [0.7, 1.0], "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [0, 5]}, n_iter=6, random_state=SEED))
print("Tuning MLP baseline ...")
best_mlp = cv_select(fit_mlp, mlp_cfgs)
print("Tuning LightGBM ...")
best_lgb = cv_select(fit_lgb, lgb_cfgs)

# ---------- final: train on full train set, evaluate once on test, seeds 42-44 ----------
res = {"baseline": [], "improved": []}
for s in (42, 43, 44):
    print(f"Seed {s}: fitting and testing ...")
    res["baseline"].append(evaluate(fit_mlp(tr, Yn[tr], best_mlp, s)(te), Yc[te]))
    res["improved"].append(evaluate(fit_lgb(tr, Yn[tr], best_lgb, s)(te), Yc[te]))
mean = lambda k, f: float(np.mean([r[f] for r in res[k]]))
std = lambda k, f: float(np.std([r[f] for r in res[k]]))
metrics = {}
for k, name in (("baseline", "baseline"), ("improved", "improved")):
    metrics[f"{name}_score"] = mean(k, "acc")
    metrics[f"{name}_score_std"] = std(k, "acc")
    metrics[f"{name}_under_resolved_share"] = mean(k, "under")
    metrics[f"{name}_under_resolved_share_std"] = std(k, "under")
    metrics[f"{name}_mean_resolution"] = mean(k, "mean_res")
    metrics[f"{name}_mean_resolution_std"] = std(k, "mean_res")
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
metrics["test_true_mean_resolution"] = float(RES[Yc[te]].mean())
metrics["train_label_noise_rate"] = float((Yn[tr] != Yc[tr]).mean())
metrics["majority_class_test_accuracy"] = float(np.bincount(Yc[te].ravel()).max() / Yc[te].size)
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

os.makedirs("figures", exist_ok=True)
plt.bar(["MLP raw clouds", "LightGBM features"], [metrics["baseline_score"], metrics["improved_score"]],
        yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]])
plt.ylabel("Per-region resolution accuracy (test)")
plt.tight_layout(); plt.savefig("figures/accuracy.png", dpi=120)

out = {
    "question": "Can a tuned gradient-boosting predictor on per-region start-to-goal deformation features beat a standard MLP on raw start and goal pointclouds at predicting per-region model resolution for a simulated deformable tree?",
    "method": "Simulated 2D branching trees with smooth random bends; resolution label = smallest control-point count (1, 2, 4) whose piecewise-linear fit of the region displacement is under a threshold; train labels randomly flipped, test labels clean; both models tuned by 3-fold CV on train only; seeds 42-44.",
    "dataset": {"name": "Simulated deformable 2D trees (8 regions, 200 points each)", "source_url": "generated in script (numpy); no download",
                "n_rows": N_TREES, "n_train": int(len(tr)), "n_test": int(len(te)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "Per-region resolution accuracy on clean test labels",
    "higher_is_better": True,
    "baseline": "scikit-learn MLPClassifier on flattened start+goal pointclouds predicting all region resolutions (stand-in for the paper's diffusion generator), CV-tuned over 6 configs.",
    "improved": "One LightGBM classifier shared across regions on local deformation features (displacement mean/spread, curvature, end-to-end change, branch length, stretch, region id), CV-tuned over 6 configs.",
    "metrics": metrics,
    "notes": "Simulated data, not the paper's; labels come from a geometric approximation-error rule, not closed-loop planning performance, so results do not test the paper's speed/performance claim. Features such as displacement curvature are close to the labelling rule, which favours the feature model. The diffusion generator is replaced by an MLP. Threshold is the 70th percentile of linear-fit error on train trees. Seeds only affect model training; the simulated data are fixed. Tuning uses noisy train labels. Under-resolved share and mean resolution are proxies only.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Wrote results.json")
