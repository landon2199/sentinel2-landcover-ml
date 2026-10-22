"""Accuracy assessment and area estimation following Olofsson et al. (2014).

Conventions: error matrices have **rows = map class** and **columns = reference class**,
as in Olofsson et al. (2014, Remote Sensing of Environment 148: 42-57).
"""
from __future__ import annotations

import numpy as np

Z95 = 1.96


def error_matrix(y_map, y_ref, classes) -> np.ndarray:
    """Count matrix n[i, j] = samples mapped as classes[i] with reference classes[j]."""
    pos = {c: k for k, c in enumerate(classes)}
    n = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for m, r in zip(np.asarray(y_map), np.asarray(y_ref)):
        n[pos[m], pos[r]] += 1
    return n


def sample_metrics(n: np.ndarray) -> dict:
    """Unweighted metrics of a count matrix (valid for a census or a simple random sample).

    Kappa is reported for comparability with older literature only; see Pontius & Millones (2011).
    """
    n = np.asarray(n, dtype=float)
    total = n.sum()
    diag = np.diag(n)
    rows, cols = n.sum(1), n.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = np.where(rows > 0, diag / rows, np.nan)
        pa = np.where(cols > 0, diag / cols, np.nan)
        f1 = 2 * ua * pa / (ua + pa)
    oa = diag.sum() / total
    pe = (rows * cols).sum() / total**2
    return {
        "overall_accuracy": float(oa),
        "kappa": float((oa - pe) / (1 - pe)),
        "users_accuracy": ua.tolist(),
        "producers_accuracy": pa.tolist(),
        "f1": f1.tolist(),
        "macro_f1": float(np.nanmean(f1)),
        "balanced_accuracy": float(np.nanmean(pa)),
    }


def olofsson(n: np.ndarray, map_counts, pixel_area: float = 1.0) -> dict:
    """Stratified estimators of accuracy and area with standard errors.

    Parameters
    ----------
    n : (q, q) sample count matrix, rows = map strata, columns = reference.
    map_counts : mapped pixels per class (stratum sizes N_i), same order as rows.
    pixel_area : area of one pixel in the output unit (e.g. km2).

    Returns estimated proportions, overall/user's/producer's accuracy and class areas,
    each with a standard error and a 95% confidence half-width (1.96 * SE).
    Equations 2-10 of Olofsson et al. (2014) (Card 1982 for producer's accuracy variance).
    """
    n = np.asarray(n, dtype=float)
    N_i = np.asarray(map_counts, dtype=float)
    n_i = n.sum(axis=1)
    if np.any(n_i == 0):
        raise ValueError("Every map stratum needs at least one sample; strata with no samples: "
                         f"{np.flatnonzero(n_i == 0).tolist()}")
    # Variances divide by n_i - 1; a stratum with a single sample has no variance estimate,
    # so its variance terms (and every CI that uses them) become NaN instead of dividing by zero.
    dof = np.where(n_i > 1, n_i - 1, np.nan)
    if np.any(n_i == 1):
        import warnings

        warnings.warn("Stratum with n = 1: variances and confidence intervals are undefined (NaN).", stacklevel=2)
    W = N_i / N_i.sum()
    frac = n / n_i[:, None]  # n_ij / n_i.
    p = W[:, None] * frac  # estimated area proportions p_ij

    # Overall accuracy
    U = np.diag(frac)
    oa = np.trace(p)
    var_oa = np.sum(W**2 * U * (1 - U) / dof)

    # User's accuracy
    var_ua = U * (1 - U) / dof

    # Producer's accuracy
    p_col = p.sum(axis=0)
    P = np.diag(p) / p_col
    N_hat_j = (N_i[:, None] * frac).sum(axis=0)  # estimated reference pixels per class
    q = len(N_i)
    var_pa = np.empty(q)
    for j in range(q):
        others = [i for i in range(q) if i != j]
        term1 = N_i[j] ** 2 * (1 - P[j]) ** 2 * U[j] * (1 - U[j]) / dof[j]
        term2 = P[j] ** 2 * np.sum(
            N_i[others] ** 2 * frac[others, j] * (1 - frac[others, j]) / dof[others]
        )
        var_pa[j] = (term1 + term2) / N_hat_j[j] ** 2

    # Area of each reference class
    se_pcol = np.sqrt(np.sum(W[:, None] ** 2 * frac * (1 - frac) / dof[:, None], axis=0))
    total_area = N_i.sum() * pixel_area
    area = p_col * total_area
    se_area = se_pcol * total_area

    return {
        "proportions": p.tolist(),
        "weights": W.tolist(),
        "overall_accuracy": float(oa),
        "overall_accuracy_ci95": float(Z95 * np.sqrt(var_oa)),
        "users_accuracy": U.tolist(),
        "users_accuracy_ci95": (Z95 * np.sqrt(var_ua)).tolist(),
        "producers_accuracy": P.tolist(),
        "producers_accuracy_ci95": (Z95 * np.sqrt(var_pa)).tolist(),
        "mapped_area": (N_i * pixel_area).tolist(),
        "estimated_area": area.tolist(),
        "estimated_area_se": se_area.tolist(),
        "estimated_area_ci95": (Z95 * se_area).tolist(),
    }


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    """Group correlated features by season and spectral region for permutation importance."""
    regions = {
        "visible (B02-B04)": ("B02", "B03", "B04"),
        "red edge (B05-B07)": ("B05", "B06", "B07"),
        "NIR (B08, B8A)": ("B08", "B8A"),
        "SWIR (B11, B12)": ("B11", "B12"),
        "indices (NDVI, NDMI, NDWI, MNDWI)": ("NDVI", "NDMI", "NDWI", "MNDWI"),
    }
    groups: dict[str, list[int]] = {}
    for k, name in enumerate(names):
        season, _, rest = name.partition("_")
        label = None
        for region, members in regions.items():
            if rest in members:
                label = f"{season} {region}"
        groups.setdefault(label or name.replace("_", " "), []).append(k)
    return groups


def grouped_permutation_importance(model, X, y, groups: dict[str, list[int]], score, n_repeats=10, seed=0):
    """Drop in ``score(y, model.predict(X))`` when each group of columns is shuffled jointly.

    Shuffling correlated features together avoids the near-zero importances that
    single-feature permutation gives when another feature carries the same information.
    """
    rng = np.random.default_rng(seed)
    base = score(y, model.predict(X))
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - score(y, model.predict(Xp)))
        out[name] = (float(np.mean(drops)), float(np.std(drops)))
    return base, out
