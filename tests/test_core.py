"""Fast unit tests (no network)."""
import numpy as np
import pytest

from landcover.assess import error_matrix, feature_groups, olofsson, sample_metrics
from landcover.features import normalized_difference, spectral_indices
from landcover.train import block_ids, buffer_zone, clustered_sample, harmonize_labels, stratified_sample


def test_normalized_difference():
    a = np.array([0.4, 0.2, 0.0])
    b = np.array([0.1, 0.2, 0.0])
    out = normalized_difference(a, b)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.0)
    assert np.isnan(out[2])  # zero denominator


def test_spectral_indices_vegetation_vs_water():
    veg = {"B03": 0.08, "B04": 0.05, "B08": 0.45, "B8A": 0.45, "B11": 0.20}
    water = {"B03": 0.06, "B04": 0.04, "B08": 0.02, "B8A": 0.02, "B11": 0.01}
    v = {k: np.array(x) for k, x in spectral_indices({k: np.array(x) for k, x in veg.items()}).items()}
    w = {k: np.array(x) for k, x in spectral_indices({k: np.array(x) for k, x in water.items()}).items()}
    assert v["NDVI"] > 0.7 and w["NDVI"] < 0
    assert w["NDWI"] > 0 > v["NDWI"]
    assert w["MNDWI"] > 0 > v["MNDWI"]


def test_block_ids():
    # 10 x 10 grid of 20 m pixels starting at (500000, 3400000); 100 m blocks -> 2 x 2 blocks
    transform = (20.0, 0.0, 500000.0, 0.0, -20.0, 3400000.0)
    b = block_ids((10, 10), transform, 100.0)
    assert len(np.unique(b)) == 4
    assert b[0, 0] == b[4, 4] != b[0, 5]
    assert b[5, 0] != b[0, 0]
    assert (b[:5, :5] == b[0, 0]).all()


def test_buffer_zone():
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    z = buffer_zone(m, 1)
    assert z.sum() == 4 and not z[3, 3]


def test_harmonize_labels_merge_and_drop():
    wc = np.array([10] * 50 + [30] * 45 + [20] * 3 + [60] * 2).reshape(10, 10)
    labels, rep = harmonize_labels(wc, np.ones_like(wc, bool), min_fraction=0.05)
    assert rep["merged"] == {20: 30} and rep["dropped"] == [60]
    assert set(np.unique(labels)) == {0, 10, 30}


def test_stratified_sample_caps_per_class():
    strata = np.array([1] * 100 + [2] * 5).reshape(-1, 1)
    idx = stratified_sample(strata, np.ones_like(strata, bool), 10, np.random.default_rng(0))
    vals = strata.ravel()[idx]
    assert (vals == 1).sum() == 10 and (vals == 2).sum() == 5


# Olofsson et al. (2014), Remote Sensing of Environment 148:42-57, worked example (Tables 8-9):
# strata deforestation, forest gain, stable forest, stable non-forest; 30 m Landsat pixels.
OLOFSSON_N = np.array(
    [
        [66, 0, 5, 4],
        [0, 55, 8, 12],
        [1, 0, 153, 11],
        [2, 1, 9, 313],
    ]
)
OLOFSSON_MAPPED_PX = np.array([200_000, 150_000, 3_200_000, 6_450_000])


def test_olofsson_worked_example():
    r = olofsson(OLOFSSON_N, OLOFSSON_MAPPED_PX, pixel_area=0.09)  # hectares
    # Published: deforestation area 21,158 ha +/- 6,158 ha (95% CI)
    assert r["estimated_area"][0] == pytest.approx(21_158, abs=5)
    assert r["estimated_area_ci95"][0] == pytest.approx(6_158, abs=5)
    # Published: forest gain 11,686 ha +/- 3,756 ha
    assert r["estimated_area"][1] == pytest.approx(11_686, abs=5)
    assert r["estimated_area_ci95"][1] == pytest.approx(3_756, abs=5)
    # Overall 0.95, deforestation user's 0.88 and producer's 0.75
    assert r["overall_accuracy"] == pytest.approx(0.95, abs=0.005)
    assert r["users_accuracy"][0] == pytest.approx(0.88, abs=0.005)
    assert r["producers_accuracy"][0] == pytest.approx(0.75, abs=0.005)
    # Area estimates must sum to the total mapped area.
    assert sum(r["estimated_area"]) == pytest.approx(OLOFSSON_MAPPED_PX.sum() * 0.09)


def test_sample_metrics_perfect_and_matrix():
    n = error_matrix([1, 1, 2, 2], [1, 1, 2, 2], [1, 2])
    m = sample_metrics(n)
    assert m["overall_accuracy"] == 1.0 and m["kappa"] == 1.0


def test_clustered_sample_makes_patches():
    labels = np.ones((50, 50), dtype=np.uint8)
    idx, seeds = clustered_sample(labels, labels > 0, 50, 5, np.random.default_rng(1))
    assert seeds.shape == idx.shape and (seeds >= 0).all()
    assert len(np.unique(seeds)) < idx.size  # several pixels share a patch seed
    assert 50 <= idx.size <= 50 + 24  # stops once the target is reached
    r, c = np.divmod(idx, 50)
    # patches: most picked pixels have a picked 4-neighbour
    picked = set(zip(r.tolist(), c.tolist()))
    with_neighbour = sum(any((a + da, b + db) in picked for da, db in ((0, 1), (1, 0), (0, -1), (-1, 0))) for a, b in picked)
    assert with_neighbour / len(picked) > 0.9


def test_feature_groups():
    names = ["summer_B02", "summer_B03", "summer_NDVI", "winter_B11", "NDVI_summer_minus_winter"]
    g = feature_groups(names)
    assert g["summer visible (B02-B04)"] == [0, 1]
    assert g["winter SWIR (B11, B12)"] == [3]
    assert len(g) == 4


def test_olofsson_guards_small_strata():
    n = np.array([[5, 0], [0, 1]])
    with pytest.warns(UserWarning):
        r = olofsson(n, [100, 50])
    assert np.isnan(r["overall_accuracy_ci95"]) and r["overall_accuracy"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        olofsson(np.array([[5, 0], [0, 0]]), [100, 50])


def test_cross_validate_buffer_drops_neighbours():
    from sklearn.dummy import DummyClassifier

    from landcover.train import cross_validate

    X = np.zeros((40, 1))
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(4), 10)
    coords = np.column_stack([np.arange(40) * 10.0, np.zeros(40)])
    out = cross_validate(DummyClassifier(), X, y, groups, n_splits=4, coords=coords, buffer_m=25)
    assert 0 <= out["overall_accuracy"] <= 1
