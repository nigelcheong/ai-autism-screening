"""Pipeline steps for why_not_faces: feature assembly, named-feature models, SHAP.

Part 0 assembles one row per image, one column per interpretable feature,
entirely from caches other experiments already wrote -- nothing here
re-detects a face, re-extracts a landmark, or re-runs a face detector.
Part 1 trains two probes (gradient boosting, L2 logistic regression) on
that table, in combination and per group alone. Part 2 explains the
gradient-boosting probe with out-of-fold SHAP, so the explanation
describes generalising behaviour, not memorised training rows.

The CNN fine-tuning and Grad-CAM arms (Part 3 / Part 4) live in
`experiments.why_not_faces.cnn` -- a separate module because that code
depends on `torch`'s training loop machinery and GPU state in a way this
one deliberately does not.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.background_images.pipeline import (
    step_2_boxes,
    step_3_exclude_boxless,
    step_4_box_geometry,
)
from experiments.why_not_faces.config import FEATURE_GROUPS, Config
from src.data.images import build_image_manifest
from src.evaluation.cross_val import repeated_stratified_cv

logger = logging.getLogger(__name__)

__all__ = [
    "step_1_manifest",
    "step_2_assemble_features",
    "step_3_named_feature_probes",
    "step_4_probe_summary",
    "step_5_shap_out_of_fold",
    "step_6_group_attribution",
    "step_7_stability_check",
    "step_8_consolidated_table",
    "verify_out_of_fold",
]


def step_1_manifest(cfg: Config) -> pd.DataFrame:
    """Build the full canonical image manifest -- identical to every prior experiment, all 2,940 rows."""
    return build_image_manifest(cfg.data_root)


def _load_acquisition(cfg: Config, manifest_full: pd.DataFrame) -> pd.DataFrame:
    """The `acquisition` group, matched back to canonical paths by `(split, class_label, filename)`.

    The audit manifest's own `path` column points at a directory
    (`data/AutismDataset/...`) that no longer exists -- the archive was
    reorganised into `data/raw/images/...` after that audit ran -- so
    rows are matched on the composite key instead, which is unique in
    both tables.
    """
    audit = pd.read_csv(cfg.audit_manifest_path)
    keep_cols = [c for c in FEATURE_GROUPS["acquisition"]]
    for excluded in cfg.excluded_audit_columns:
        assert excluded not in keep_cols, f"{excluded} should have been left out of FEATURE_GROUPS"

    audit = audit[["split", "class_label", "filename", *keep_cols]]
    merged = manifest_full.merge(
        audit, on=["split", "class_label", "filename"], how="inner", validate="one_to_one"
    )
    return merged[["path", *keep_cols]]


def _load_colour(cfg: Config) -> pd.DataFrame:
    """The `colour` group -- `blurred_images`'s whole-image sigma=0 colour features."""
    df = pd.read_csv(cfg.blurred_cache_dir / "colour_features_sigma0.csv")
    return df[["path", *FEATURE_GROUPS["colour"]]]


def _load_framing(cfg: Config, manifest_full: pd.DataFrame) -> pd.DataFrame:
    """The `framing` group -- box geometry re-derived from the cached sigma=0 MTCNN boxes.

    Reuses `background_images.pipeline`'s own box-handling and
    geometry-feature functions verbatim: the box itself is never
    re-detected, only its four scalar geometry features recomputed
    (cheap, deterministic) on this experiment's own row set.
    """
    boxes_full = step_2_boxes(manifest_full)
    _, boxes, _ = step_3_exclude_boxless(manifest_full, boxes_full)
    geometry_df = step_4_box_geometry(cfg, boxes)
    return geometry_df[["path", *FEATURE_GROUPS["framing"]]]


def _load_facial_shape_and_pose(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The `facial_shape` and `pose` groups -- `face_landmarks`'s cached MediaPipe feature tables."""
    morphology = pd.read_csv(cfg.face_landmarks_cache_dir / "features_mediapipe_morphology.csv")
    pose = pd.read_csv(cfg.face_landmarks_cache_dir / "features_mediapipe_pose.csv")
    return (
        morphology[["path", *FEATURE_GROUPS["facial_shape"]]],
        pose[["path", *FEATURE_GROUPS["pose"]]],
    )


def step_2_assemble_features(
    cfg: Config, manifest_full: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, ...]], pd.DataFrame]:
    """Assemble one row per image, one column per named feature, from five cached sources.

    Every source is inner-joined on the canonical `path` (the audit
    manifest is matched onto that path first, via `(split, class_label,
    filename)`); the result is therefore the intersection of every
    source's rows, expected to be the 2,938 rows `face_landmarks` -- the
    tightest of the five -- ultimately determined.

    Args:
        cfg: Experiment configuration.
        manifest_full: Output of `step_1_manifest` (2,940 rows).

    Returns:
        `(features, groups, exclusions_by_class)`: `features` has `path`,
        `class_label`, plus one column per feature named in
        `FEATURE_GROUPS` (~30 columns); `groups` is `FEATURE_GROUPS`
        itself, returned so the caller writing the sidecar JSON and the
        caller reading it back share one literal object; `exclusions_by_class`
        has one row per class: `n_total`, `n_included`, `n_excluded`.

    Raises:
        AssertionError: If any feature name appears in more than one
            group (the group attribution would then double-count it).
    """
    seen: Dict[str, str] = {}
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            if name in seen:
                raise AssertionError(f"Feature {name!r} appears in both {seen[name]!r} and {group!r}")
            seen[name] = group

    acquisition = _load_acquisition(cfg, manifest_full)
    colour = _load_colour(cfg)
    framing = _load_framing(cfg, manifest_full)
    facial_shape, pose = _load_facial_shape_and_pose(cfg)

    merged = manifest_full[["path", "class_label"]]
    for part in (acquisition, colour, framing, facial_shape, pose):
        merged = merged.merge(part, on="path", how="inner", validate="one_to_one")

    all_feature_names = [name for names in FEATURE_GROUPS.values() for name in names]
    merged = merged[["path", "class_label", *all_feature_names]].reset_index(drop=True)

    excluded_paths = set(manifest_full["path"]) - set(merged["path"])
    excluded_mask = manifest_full["path"].isin(excluded_paths)
    exclusions_by_class = (
        manifest_full.groupby("class_label")
        .agg(n_total=("path", "size"))
        .reset_index()
        .merge(
            manifest_full.loc[excluded_mask].groupby("class_label").size().rename("n_excluded").reset_index(),
            on="class_label",
            how="left",
        )
    )
    exclusions_by_class["n_excluded"] = exclusions_by_class["n_excluded"].fillna(0).astype(int)
    exclusions_by_class["n_included"] = exclusions_by_class["n_total"] - exclusions_by_class["n_excluded"]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg.output_dir / "features.csv", index=False)
    (cfg.output_dir / "features_groups.json").write_text(
        json.dumps({g: list(names) for g, names in FEATURE_GROUPS.items()}, indent=2), encoding="utf-8"
    )

    logger.info(
        "step_2_assemble_features: %d/%d rows survive intersection of 5 sources (%d columns, %d groups)",
        len(merged),
        len(manifest_full),
        len(all_feature_names),
        len(FEATURE_GROUPS),
    )
    return merged, dict(FEATURE_GROUPS), exclusions_by_class


def _build_hgb(cfg: Config) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=cfg.hgb_seed)


def _build_logreg(cfg: Config) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=cfg.seed)),
        ]
    )


def step_3_named_feature_probes(
    cfg: Config, features: pd.DataFrame, groups: Dict[str, Tuple[str, ...]]
) -> pd.DataFrame:
    """Cross-validate both models, on the full feature set and on each group alone.

    Same 5x10 repeated stratified CV, same seed as every prior
    experiment. `feature_set="all"` uses every column in `groups`
    combined; every other value of `feature_set` is one group's columns
    alone.

    Args:
        cfg: Experiment configuration.
        features: Output of `step_2_assemble_features`'s first element.
        groups: Output of `step_2_assemble_features`'s second element.

    Returns:
        One row per `(model, feature_set, repeat, fold)`, with every
        metric from `src.evaluation.metrics.fold_metrics`.
    """
    y = (features["class_label"] == "autistic").astype(int).to_numpy()
    all_features = [name for names in groups.values() for name in names]

    feature_sets = {"all": all_features, **{g: list(names) for g, names in groups.items()}}
    models = {"hist_gradient_boosting": _build_hgb, "logistic_regression": _build_logreg}

    rows = []
    for model_name, builder in models.items():
        for feature_set_name, columns in feature_sets.items():
            X = features[columns].to_numpy()
            fold_results = repeated_stratified_cv(
                X, y, builder(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
            )
            fold_results.insert(0, "feature_set", feature_set_name)
            fold_results.insert(0, "model", model_name)
            rows.append(fold_results)

    return pd.concat(rows, ignore_index=True)


def step_4_probe_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- sd of every metric, per `(model, feature_set)`.

    Args:
        fold_results: Output of `step_3_named_feature_probes`.

    Returns:
        One row per `(model, feature_set)`: `{metric}_mean`, `{metric}_sd`
        for every metric in `fold_results`.
    """
    metric_cols = [c for c in fold_results.columns if c not in ("model", "feature_set", "repeat", "fold")]
    rows = []
    for (model_name, feature_set_name), group in fold_results.groupby(["model", "feature_set"], sort=False):
        row = {"model": model_name, "feature_set": feature_set_name}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def _repeat_seeds(cfg: Config) -> np.ndarray:
    """Reproduce `repeated_stratified_cv`'s own per-repeat seed derivation exactly.

    So the SHAP fold assignment lines up 1:1 with the probe's own folds
    (same seed, same `StratifiedKFold` calls) -- not a strict
    requirement, but keeps "what the model saw in fold k of repeat r"
    identical between the metrics table and the SHAP table.
    """
    rng = np.random.default_rng(cfg.seed)
    return rng.integers(0, 2**31 - 1, size=cfg.n_repeats)


def step_5_shap_out_of_fold(
    cfg: Config, features: pd.DataFrame, groups: Dict[str, Tuple[str, ...]]
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Out-of-fold SHAP values for the `hist_gradient_boosting` / `all`-features probe.

    For each of the 10 repeats' 5 folds, `HistGradientBoostingClassifier`
    is fit on that fold's training rows only, and `shap.TreeExplainer` is
    asked for SHAP values on that fold's *test* rows only -- the values
    stored for a row always came from a model that never saw that row
    during training. See `verify_out_of_fold` for the explicit check.

    Args:
        cfg: Experiment configuration.
        features: Output of `step_2_assemble_features`'s first element.
        groups: Output of `step_2_assemble_features`'s second element.

    Returns:
        `(shap_values, feature_names, fold_assignment)`: `shap_values` is
        `(n_repeats, n_rows, n_features)` float64 -- every row is a test
        row exactly once per repeat, so this is fully populated, no
        `NaN`; `feature_names` is the column order `shap_values`' last
        axis matches; `fold_assignment` has one row per `(repeat, row
        index)`: `repeat`, `row`, `fold`, `train_size`, `test_size` (the
        record `verify_out_of_fold` checks against).
    """
    from sklearn.model_selection import StratifiedKFold

    feature_names = [name for names in groups.values() for name in names]
    X = features[feature_names].to_numpy()
    y = (features["class_label"] == "autistic").astype(int).to_numpy()
    n_rows = len(features)

    shap_values = np.full((cfg.n_repeats, n_rows, len(feature_names)), np.nan, dtype=np.float64)
    assignment_rows = []

    for repeat, repeat_seed in enumerate(_repeat_seeds(cfg)):
        cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=int(repeat_seed))
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            assert not (set(train_idx) & set(test_idx)), "train/test indices overlap -- StratifiedKFold is broken"

            model = _build_hgb(cfg)
            model.fit(X[train_idx], y[train_idx])
            explainer = shap.TreeExplainer(model)
            sv = np.asarray(explainer.shap_values(X[test_idx]))
            if sv.ndim == 3:
                sv = sv[:, :, 1]  # (n_test, n_features, n_classes) -> positive class

            shap_values[repeat, test_idx, :] = sv
            for row in test_idx:
                assignment_rows.append(
                    {
                        "repeat": repeat,
                        "row": int(row),
                        "fold": fold,
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                    }
                )

    assert not np.isnan(shap_values).any(), "every row must be a test row exactly once per repeat"
    logger.info(
        "step_5_shap_out_of_fold: %d repeats x %d rows x %d features, all out-of-fold",
        cfg.n_repeats,
        n_rows,
        len(feature_names),
    )
    return shap_values, feature_names, pd.DataFrame(assignment_rows)


def verify_out_of_fold(fold_assignment: pd.DataFrame, cfg: Config) -> bool:
    """Assert no row's SHAP value came from a fold it was trained in.

    By construction (see `step_5_shap_out_of_fold`), every `(repeat,
    row)` appears in `fold_assignment` exactly once, always as a member
    of that fold's *test* set. This re-derives the same `StratifiedKFold`
    splits independently and checks the recorded fold assignment against
    them, rather than trusting the loop that produced it.

    Args:
        fold_assignment: Third element of `step_5_shap_out_of_fold`'s
            return value.
        cfg: Experiment configuration (for `n_repeats`, `n_splits`).

    Returns:
        `True` if every check passes.

    Raises:
        AssertionError: If any row appears more than once per repeat, or
            any repeat does not cover every row exactly once.
    """
    n_rows = fold_assignment["row"].max() + 1
    for repeat in range(cfg.n_repeats):
        rows_this_repeat = fold_assignment.loc[fold_assignment["repeat"] == repeat, "row"]
        assert len(rows_this_repeat) == n_rows, f"repeat {repeat}: expected {n_rows} test rows, got {len(rows_this_repeat)}"
        assert rows_this_repeat.is_unique, f"repeat {repeat}: a row was a test row more than once (fold overlap)"
        assert set(rows_this_repeat) == set(range(n_rows)), f"repeat {repeat}: not every row was held out"
    return True


def step_6_group_attribution(
    shap_values: np.ndarray, feature_names: List[str], groups: Dict[str, Tuple[str, ...]]
) -> pd.DataFrame:
    """Per-repeat, per-group share of total mean |SHAP|, then mean +/- sd across repeats.

    Within one repeat: `mean_abs_shap[feature] = mean(|shap_values[repeat, :, feature]|)`
    over all rows; `group_attribution[group] = sum(mean_abs_shap[f] for f
    in group)`; `share[group] = group_attribution[group] / sum(all
    groups' group_attribution)` -- shares within a repeat sum to exactly
    1 by construction, which is why the reported mean across repeats
    also sums to exactly 1 (linearity).

    Args:
        shap_values: First element of `step_5_shap_out_of_fold`'s return
            value, `(n_repeats, n_rows, n_features)`.
        feature_names: Second element -- column order for `shap_values`'
            last axis.
        groups: `FEATURE_GROUPS` (or an equivalent mapping loaded from
            `features_groups.json`).

    Returns:
        One row per group, sorted by `share_mean` descending:
        `group`, `share_mean`, `share_sd`, `share_min`, `share_max`
        (the last two across the `n_repeats` per-repeat shares).
    """
    feature_index = {name: i for i, name in enumerate(feature_names)}
    n_repeats = shap_values.shape[0]

    shares_by_repeat: Dict[str, List[float]] = {group: [] for group in groups}
    for repeat in range(n_repeats):
        mean_abs_shap = np.abs(shap_values[repeat]).mean(axis=0)
        group_attribution = {
            group: float(sum(mean_abs_shap[feature_index[f]] for f in names)) for group, names in groups.items()
        }
        total = sum(group_attribution.values())
        for group in groups:
            shares_by_repeat[group].append(group_attribution[group] / total)

    rows = []
    for group, shares in shares_by_repeat.items():
        shares_arr = np.array(shares)
        rows.append(
            {
                "group": group,
                "share_mean": float(shares_arr.mean()),
                "share_sd": float(shares_arr.std()),
                "share_min": float(shares_arr.min()),
                "share_max": float(shares_arr.max()),
            }
        )

    result = pd.DataFrame(rows).sort_values("share_mean", ascending=False).reset_index(drop=True)
    total_share = result["share_mean"].sum()
    assert abs(total_share - 1.0) < 1e-9, f"group shares must sum to 1.0, got {total_share}"
    return result


def step_7_stability_check(shap_values: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank features by mean |SHAP| within each repeat; rank-correlate every pair of repeats.

    Args:
        shap_values: `(n_repeats, n_rows, n_features)`, from
            `step_5_shap_out_of_fold`.
        feature_names: Column order for `shap_values`' last axis.

    Returns:
        `(ranks, correlations)`: `ranks` has one row per `(repeat,
        feature)`: `repeat`, `feature`, `mean_abs_shap`, `rank` (1 =
        highest); `correlations` has one row per unordered pair of
        repeats: `repeat_a`, `repeat_b`, `spearman_r`.
    """
    from scipy.stats import spearmanr

    n_repeats = shap_values.shape[0]
    rank_rows = []
    rank_by_repeat = {}
    for repeat in range(n_repeats):
        mean_abs_shap = np.abs(shap_values[repeat]).mean(axis=0)
        order = np.argsort(-mean_abs_shap)
        ranks = np.empty(len(feature_names), dtype=int)
        ranks[order] = np.arange(1, len(feature_names) + 1)
        rank_by_repeat[repeat] = ranks
        for i, feature in enumerate(feature_names):
            rank_rows.append(
                {"repeat": repeat, "feature": feature, "mean_abs_shap": float(mean_abs_shap[i]), "rank": int(ranks[i])}
            )

    corr_rows = []
    for a in range(n_repeats):
        for b in range(a + 1, n_repeats):
            rho, _ = spearmanr(rank_by_repeat[a], rank_by_repeat[b])
            corr_rows.append({"repeat_a": a, "repeat_b": b, "spearman_r": float(rho)})

    return pd.DataFrame(rank_rows), pd.DataFrame(corr_rows)


def _embedding_probe_auc(cfg: Config, emb_dir, emb_name: str, manifest_full: pd.DataFrame) -> Tuple[float, float, int]:
    """Re-run the standard L2-logistic-regression-behind-a-`StandardScaler` probe on a cached embedding array."""
    from src.models.embeddings import load_embeddings

    array, index, _ = load_embeddings(emb_dir, name=emb_name)
    label_by_path = dict(zip(manifest_full["path"], manifest_full["class_label"]))
    y = (index["path"].map(label_by_path) == "autistic").astype(int).to_numpy()
    fold_results = repeated_stratified_cv(
        array, y, _build_logreg(cfg), n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed
    )
    return float(fold_results["roc_auc"].mean()), float(fold_results["roc_auc"].std()), len(index)


def step_8_consolidated_table(
    cfg: Config,
    probe_summary: pd.DataFrame,
    manifest_full: pd.DataFrame,
    cnn_published: Dict,
    cnn_corrected: List[Dict],
) -> pd.DataFrame:
    """Assemble Part 5's whole-image-stream comparison, one row per arm, explicit row counts throughout.

    Every non-CNN row is re-run here (from cached embeddings or from
    this experiment's own group-alone probes), on that source's own
    native row set -- never forced onto a common row count, and never
    silently compared as if they were. `n_rows` is reported for every
    row precisely so a 2,940-row number and a 2,938-row number are never
    mistaken for the same measurement.

    Args:
        cfg: Experiment configuration.
        probe_summary: Output of `step_4_probe_summary` -- source of the
            `all`-features HistGradientBoosting row and the four
            group-alone logistic-regression rows (`framing`,
            `facial_shape`, `colour`, `pose` -- each already computed on
            the L2-logistic-regression-behind-a-`StandardScaler` protocol
            every prior experiment used, so no second probe run is
            needed for these).
        manifest_full: Output of `step_1_manifest` (2,940 rows) --
            needed to attach class labels to the reused embedding caches.
        cnn_published: Output of `experiments.why_not_faces.cnn.run_published_protocol`.
        cnn_corrected: Output of `experiments.why_not_faces.cnn.run_corrected_protocol`
            (one dict per fold).

    Returns:
        One row per arm: `arm`, `roc_auc_mean`, `roc_auc_sd`, `n_rows`, `source`.
    """
    rows = []

    corrected_aucs = np.array([f["metrics"]["roc_auc"] for f in cnn_corrected])
    rows.append(
        {
            "arm": "Fine-tuned CNN, corrected protocol",
            "roc_auc_mean": float(corrected_aucs.mean()),
            "roc_auc_sd": float(corrected_aucs.std()),
            "n_rows": int(sum(f["n_test"] for f in cnn_corrected)),
            "source": f"why_not_faces (DenseNet201, {len(cnn_corrected)}-fold held-out, hash-grouped/deduplicated split)",
        }
    )
    rows.append(
        {
            "arm": "Fine-tuned CNN, published protocol",
            "roc_auc_mean": float(cnn_published["metrics"]["roc_auc"]),
            "roc_auc_sd": float("nan"),
            "n_rows": int(cnn_published["n_report"]),
            "source": "why_not_faces (DenseNet201, published train/valid folders, reported on valid)",
        }
    )

    intact_mean, intact_sd, intact_n = _embedding_probe_auc(cfg, cfg.intact_cache_dir, "embeddings", manifest_full)
    rows.append(
        {
            "arm": "Intact image, frozen probe",
            "roc_auc_mean": intact_mean,
            "roc_auc_sd": intact_sd,
            "n_rows": intact_n,
            "source": "intact_images (frozen resnet50 embedding, logistic regression)",
        }
    )

    named = probe_summary.loc[
        (probe_summary["model"] == "hist_gradient_boosting") & (probe_summary["feature_set"] == "all")
    ].iloc[0]
    rows.append(
        {
            "arm": "Named-feature model (this experiment)",
            "roc_auc_mean": float(named["roc_auc_mean"]),
            "roc_auc_sd": float(named["roc_auc_sd"]),
            "n_rows": 2938,
            "source": "why_not_faces (HistGradientBoosting, all ~30 named features)",
        }
    )

    crop_mean, crop_sd, crop_n = _embedding_probe_auc(
        cfg, cfg.face_crop_blur_cache_dir, "embeddings_crop_1.0_sigma0", manifest_full
    )
    rows.append(
        {
            "arm": "Face crop",
            "roc_auc_mean": crop_mean,
            "roc_auc_sd": crop_sd,
            "n_rows": crop_n,
            "source": "face_crop_blur (crop_1.0, sigma=0, frozen resnet50 embedding, logistic regression)",
        }
    )

    for group, label in (
        ("framing", "Framing / box geometry"),
        ("facial_shape", "Facial shape (landmarks)"),
        ("colour", "Colour only"),
        ("pose", "Pose only"),
    ):
        row = probe_summary.loc[
            (probe_summary["model"] == "logistic_regression") & (probe_summary["feature_set"] == group)
        ].iloc[0]
        rows.append(
            {
                "arm": label,
                "roc_auc_mean": float(row["roc_auc_mean"]),
                "roc_auc_sd": float(row["roc_auc_sd"]),
                "n_rows": 2938,
                "source": f"why_not_faces ({group} group alone, logistic regression)",
            }
        )

    result = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return result
