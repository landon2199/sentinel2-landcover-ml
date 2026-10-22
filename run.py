"""End-to-end run: data -> features -> sampling -> models -> random vs spatial CV -> assessment -> figures.

Example:
    python run.py                       # Bryan-College Station, Texas, 2021, 20 m
    python run.py --bbox -96.45 30.52 -96.20 30.75 --year 2021 --resolution 20 --seed 42
"""
from __future__ import annotations

import landcover  # noqa: F401  (sets MKL/DLL environment before numpy loads)

import argparse
import json
import time
from pathlib import Path

import numpy as np

from landcover import assess, data, features, plots, train

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", nargs=4, type=float, default=[-96.45, 30.52, -96.20, 30.75],
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="WGS 84 bounding box")
    p.add_argument("--place", default="Bryan–College Station, Texas", help="name used in figure titles")
    p.add_argument("--year", type=int, default=2021, help="label year (WorldCover exists for 2020 and 2021)")
    p.add_argument("--resolution", type=float, default=20.0, help="pixel size in metres")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-km", type=float, default=2.0, help="spatial block size for CV and the test split")
    p.add_argument("--test-fraction", type=float, default=0.2, help="share of blocks held out for testing")
    p.add_argument("--buffer-m", type=float, default=200.0, help="training pixels this close to test blocks are dropped")
    p.add_argument("--n-train", type=int, default=2000, help="training pixels per class")
    p.add_argument("--n-test", type=int, default=500, help="test pixels per mapped class (Olofsson sample)")
    p.add_argument("--cv-buffer-m", type=float, default=0.0,
                   help="optional: drop training samples this close to the held-out fold in spatial CV")
    p.add_argument("--cluster-px", type=int, default=5, help="patch width (pixels) for the clustered training design")
    p.add_argument("--min-class-fraction", type=float, default=0.005, help="rarer classes are merged or dropped")
    p.add_argument("--max-cloud", type=float, default=10.0, help="scene cloud cover limit (%%)")
    p.add_argument("--no-winter", action="store_true", help="skip the winter composite")
    p.add_argument("--cache", type=Path, default=ROOT / "data", help="folder for cached rasters (not in git)")
    return p.parse_args()


def cache_paths(args):
    tag = f"{'_'.join(f'{v:.3f}' for v in args.bbox)}_{args.year}_{args.resolution:g}m{'_nowinter' if args.no_winter else ''}"
    return (args.cache / f"features_{tag}.tif", args.cache / f"worldcover_{tag}.tif",
            args.cache / f"scenes_{tag}.json")


def _prepare_inputs(args, status_file):
    """Child-process entry: build the cache, report back, then exit hard.

    On some Windows setups a process that has made GDAL network reads never finishes
    exiting, and threaded reads were also seen to stall scikit-learn later in the same
    process (both look like a deadlock when threads start or stop). Doing the download here
    keeps the analysis process clean. The status goes to a file, because even a queue's
    feeder thread can stall, and the parent terminates this process instead of waiting.
    """
    import os
    import sys
    import traceback

    code = 0
    try:
        get_inputs(args)
    except Exception:  # noqa: BLE001 - report and signal failure to the parent
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    Path(status_file).write_text(str(code))
    os._exit(code)


def get_inputs(args):
    """Feature stack and WorldCover labels on one grid, cached as GeoTIFFs."""
    data.configure_gdal()
    f_path, l_path, meta_path = cache_paths(args)
    if f_path.exists() and l_path.exists() and meta_path.exists():
        print(f"Using cached inputs in {args.cache}")
        stack, names, grid = data.read_stack(f_path)
        wc = data.read_stack(l_path)[0][0]
        return stack, names, wc, grid, json.loads(meta_path.read_text())

    lon, lat = (args.bbox[0] + args.bbox[2]) / 2, (args.bbox[1] + args.bbox[3]) / 2
    epsg = data.utm_epsg(lon, lat)
    y = args.year
    windows = {"summer": f"{y}-04-01/{y}-09-30"}
    if not args.no_winter:
        windows["winter"] = f"{y - 1}-12-01/{y}-02-28"
    seasons, scenes, grid = {}, {}, None
    for season, window in windows.items():
        items = data.search_sentinel2(args.bbox, window, max_cloud=args.max_cloud)
        print(f"{season}: {len(items)} Sentinel-2 L2A items ({window})")
        bands, n_clear, g = data.load_composite(items, args.bbox, epsg, args.resolution)
        grid = grid or g
        seasons[season] = bands
        scenes[season] = {
            "window": window,
            "items": sorted(it.id for it in items),
            "median_clear_observations": float(np.median(n_clear)),
            "pixels_without_clear_observation": int((n_clear == 0).sum()),
        }
    stack, names = features.build_feature_stack(seasons)
    wc_items = data.search_worldcover(args.bbox, y)
    wc = data.load_worldcover(wc_items, grid)
    scenes["worldcover_items"] = [it.id for it in wc_items]
    scenes["crs"] = grid.crs
    data.save_stack(f_path, stack, names, grid)
    data.save_stack(l_path, wc, ["worldcover"], grid, nodata=0)
    meta_path.write_text(json.dumps(scenes, indent=2))
    return stack, names, wc, grid, scenes


def main():
    args = parse_args()
    data.configure_gdal()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    if not all(p.exists() for p in cache_paths(args)):
        # Download and composite in a separate process. On Windows, GDAL/odc-stac worker
        # threads left behind in this process can deadlock scikit-learn's thread pools.
        import multiprocessing as mp

        status_file = args.cache / "prepare_status.txt"
        status_file.unlink(missing_ok=True)
        args.cache.mkdir(parents=True, exist_ok=True)
        proc = mp.get_context("spawn").Process(target=_prepare_inputs, args=(args, str(status_file)))
        proc.start()
        while not status_file.exists() and proc.is_alive():
            time.sleep(1)
        time.sleep(1)
        status = int(status_file.read_text() or 1) if status_file.exists() else (proc.exitcode or 1)
        if proc.is_alive():
            proc.terminate()
        status_file.unlink(missing_ok=True)
        if status != 0:
            raise SystemExit("Data preparation failed; see the traceback above.")
    stack, names, wc, grid, scenes = get_inputs(args)
    t_data = time.time() - t0
    print(f"Inputs ready: {stack.shape[0]} features on a {grid.shape[0]} x {grid.shape[1]} grid ({t_data:.0f} s)")

    valid = np.isfinite(stack).all(axis=0) & (wc > 0)
    labels, label_report = train.harmonize_labels(wc, valid, args.min_class_fraction)
    classes = label_report["classes"]
    print("Classes:", [data.WORLDCOVER[c][0] for c in classes], "| merged:", label_report["merged"],
          "| dropped:", label_report["dropped"])

    # ---- spatial split: whole blocks held out for testing, a buffer removed around them
    blocks = train.block_ids(grid.shape, grid.transform, args.block_km * 1000)
    usable = labels > 0
    test_mask = train.split_test_blocks(blocks, usable, args.test_fraction, args.seed)
    buffer = train.buffer_zone(test_mask, int(round(args.buffer_m / args.resolution)))
    train_area = usable & ~test_mask & ~buffer
    test_pop = usable & test_mask

    X_all = stack.reshape(stack.shape[0], -1).T
    y_all = labels.ravel()
    tr_idx = train.stratified_sample(labels, train_area, args.n_train, rng)
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    groups = blocks.ravel()[tr_idx]
    print(f"Training sample (scattered pixels): {tr_idx.size} pixels from {np.unique(groups).size} blocks")
    # Same size, but drawn as polygon-like patches, the way training data are often digitized.
    cl_idx, cl_seed = train.clustered_sample(labels, train_area, args.n_train, args.cluster_px,
                                             np.random.default_rng(args.seed))
    print(f"Training sample (clustered {args.cluster_px}x{args.cluster_px} patches): {cl_idx.size} pixels "
          f"in {np.unique(cl_seed).size} patches")
    # Spatial CV groups come from the pixel itself (scattered) or from its patch seed (clustered),
    # so a patch that straddles a block edge stays in one fold.
    designs = {"scattered": (tr_idx, tr_idx), "clustered": (cl_idx, cl_seed)}

    # ---- random vs spatial cross-validation on each training design
    models = train.make_models(args.seed)
    cv, sweeps = {}, {}
    for design, (idx, group_px) in designs.items():
        Xd, yd = X_all[idx], y_all[idx]
        xy = train.pixel_coords(idx, grid.shape, grid.transform)

        def spatial(model, km):
            g = train.block_ids(grid.shape, grid.transform, km * 1000).ravel()[group_px]
            return train.cross_validate(model, Xd, yd, g, seed=args.seed, coords=xy, buffer_m=args.cv_buffer_m)

        for name in (models if design == "scattered" else ["random_forest"]):
            key = f"{name}|{design}"
            cv[key] = {
                "random_cv": train.cross_validate(models[name], Xd, yd, None, seed=args.seed),
                "spatial_cv": spatial(models[name], args.block_km),
            }
            print(f"{key}: random CV BA {cv[key]['random_cv']['balanced_accuracy']:.3f} | "
                  f"spatial CV BA {cv[key]['spatial_cv']['balanced_accuracy']:.3f}")
        key = f"random_forest|{design}"
        sweeps[design] = [(0, cv[key]["random_cv"])]
        for km in (0.5, 1.0, 2.0, 4.0):
            res_km = cv[key]["spatial_cv"] if km == args.block_km else spatial(models["random_forest"], km)
            sweeps[design].append((km, res_km))
            print(f"  RF {design} block CV {km:g} km: BA {res_km['balanced_accuracy']:.3f}")

    # ---- fit on the full training sample, map, and assess on the held-out blocks
    test_flat = np.flatnonzero(test_pop.ravel())
    ref_test = y_all[test_flat]
    from sklearn.base import clone

    rf_clustered = clone(models["random_forest"]).fit(X_all[cl_idx], y_all[cl_idx])
    cv["random_forest|clustered"]["test"] = assess.sample_metrics(
        assess.error_matrix(rf_clustered.predict(X_all[test_flat]), ref_test, classes))
    results = {}
    pred_map = None
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        if name == "random_forest":
            pred_map = train.predict_raster(model, stack, valid)
            pred_test = pred_map.ravel()[test_flat]
        else:
            pred_test = model.predict(X_all[test_flat])
        census = assess.sample_metrics(assess.error_matrix(pred_test, ref_test, classes))
        cv[f"{name}|scattered"]["test"] = census

        # Olofsson et al. (2014): stratified random sample by MAP class inside the test blocks
        test_map = np.zeros(labels.size, dtype="uint8")
        test_map[test_flat] = pred_test
        s_idx = train.stratified_sample(test_map.reshape(labels.shape), test_pop, args.n_test,
                                        np.random.default_rng(args.seed + 1))
        n = assess.error_matrix(test_map[s_idx], y_all[s_idx], classes)
        map_counts = [int((pred_test == c).sum()) for c in classes]
        px_km2 = (args.resolution / 1000) ** 2
        olof = assess.olofsson(n, map_counts, pixel_area=px_km2)
        ref_counts = [int((ref_test == c).sum()) for c in classes]
        olof["worldcover_area_census"] = [c * px_km2 for c in ref_counts]
        olof["ci_covers_census"] = [
            bool(abs(e - c) <= h) for e, c, h in
            zip(olof["estimated_area"], olof["worldcover_area_census"], olof["estimated_area_ci95"])
        ]
        results[name] = {"census": census, "olofsson": olof, "sample_counts": n.tolist(),
                         "sample_metrics": assess.sample_metrics(n), "sample_size": int(n.sum()),
                         "sample_index": s_idx}
        print(f"{name}: held-out census OA {census['overall_accuracy']:.3f}, BA {census['balanced_accuracy']:.3f} | "
              f"Olofsson OA {olof['overall_accuracy']:.3f} +/- {olof['overall_accuracy_ci95']:.3f}")

    # ---- grouped permutation importance (Random Forest, held-out Olofsson sample)
    from sklearn.metrics import balanced_accuracy_score

    s_idx = results["random_forest"]["sample_index"]
    groups_fi = assess.feature_groups(names)
    _, pi = assess.grouped_permutation_importance(models["random_forest"], X_all[s_idx], y_all[s_idx], groups_fi,
                                                  balanced_accuracy_score, n_repeats=10, seed=args.seed)
    pi_sorted = sorted(pi.items(), key=lambda kv: kv[1][0], reverse=True)

    # ---- figures
    res = args.resolution / 1000
    extent = [0, grid.shape[1] * res, 0, grid.shape[0] * res]
    ref_show = np.where(valid, labels, 0)
    pred_show = np.where(ref_show > 0, pred_map, 0)
    figs = ROOT / "figures"
    plots.map_comparison(pred_show, ref_show, classes, extent, test_mask, figs / "map_comparison.png",
                         subtitle=f"{args.place}, {args.year}. Black outlines: {args.block_km:g} km blocks held out for testing.")
    rf = results["random_forest"]
    plots.confusion_figure(rf["sample_counts"], classes, rf["olofsson"]["users_accuracy"],
                           rf["olofsson"]["producers_accuracy"], figs / "confusion_matrix.png",
                           f"Held-out test sample (n = {rf['sample_size']}), UA/PA area-weighted")
    plots.cv_comparison(cv, sweeps, figs / "cv_comparison.png", args.cluster_px)
    plots.importance_figure([k for k, _ in pi_sorted], [v[0] for _, v in pi_sorted], [v[1] for _, v in pi_sorted],
                            figs / "feature_importance.png")
    plots.hero_figure(pred_show, ref_show, classes, extent, cv,
                      rf["olofsson"]["overall_accuracy"], rf["olofsson"]["overall_accuracy_ci95"],
                      figs / "hero.png", args.place, args.cluster_px, args.year)
    data.save_stack(args.cache / "rf_classification.tif", pred_map, ["rf_class"], grid, nodata=0)

    # ---- metrics.json
    def r(x, k=4):
        if isinstance(x, list):
            return [r(v, k) for v in x]
        return round(float(x), k) if isinstance(x, (float, np.floating)) else x

    class_names = [data.WORLDCOVER[c][0] for c in classes]
    out = {
        "run": {
            "bbox_wgs84": args.bbox, "place": args.place, "year": args.year, "resolution_m": args.resolution,
            "seed": args.seed, "crs": grid.crs, "grid_shape": list(grid.shape),
            "block_km": args.block_km, "cv_buffer_m": args.cv_buffer_m, "test_fraction_of_blocks": args.test_fraction, "buffer_m": args.buffer_m,
            "n_train_per_class": args.n_train, "n_test_per_class": args.n_test,
            "runtime_seconds": round(time.time() - t0, 1), "data_seconds": round(t_data, 1),
            "scenes": scenes,
        },
        "labels": {
            "classes": classes, "class_names": class_names,
            "worldcover_fractions_before_merge": label_report["fractions_before"],
            "merged": {str(k): v for k, v in label_report["merged"].items()},
            "dropped": label_report["dropped"],
            "share_of_valid_pixels_excluded": r(1 - usable.sum() / max(valid.sum(), 1)),
        },
        "samples": {"train": int(tr_idx.size), "train_blocks": int(np.unique(groups).size),
                    "train_clustered": int(cl_idx.size), "cluster_px": args.cluster_px,
                    "test_blocks": int(np.unique(blocks[test_mask]).size), "test_population_pixels": int(test_pop.sum())},
        "cross_validation": {m: {k: {kk: r(vv) for kk, vv in v.items() if not isinstance(vv, list)}
                                 for k, v in d.items()} for m, d in cv.items()},
        "block_size_sweep_random_forest": {
            design: [{"block_km": km, **{k: r(v) for k, v in d.items() if not isinstance(v, list)}} for km, d in sw]
            for design, sw in sweeps.items()
        },
        "test": {},
        "grouped_permutation_importance": [
            {"group": k, "features": [names[i] for i in groups_fi[k]], "mean_drop_balanced_accuracy": r(v[0]),
             "sd": r(v[1])} for k, v in pi_sorted
        ],
    }
    for name, d in results.items():
        o, c = d["olofsson"], d["census"]
        out["test"][name] = {
            "census_of_test_blocks": {
                "overall_accuracy": r(c["overall_accuracy"]), "kappa": r(c["kappa"]),
                "balanced_accuracy": r(c["balanced_accuracy"]), "macro_f1": r(c["macro_f1"]),
                "per_class": {nm: {"users_accuracy": r(u), "producers_accuracy": r(p), "f1": r(f)}
                              for nm, u, p, f in zip(class_names, c["users_accuracy"], c["producers_accuracy"], c["f1"])},
            },
            "olofsson_2014": {
                "sample_size": d["sample_size"], "sample_counts_rows_map_cols_reference": d["sample_counts"],
                "overall_accuracy": r(o["overall_accuracy"]), "overall_accuracy_ci95": r(o["overall_accuracy_ci95"]),
                "per_class": {
                    nm: {
                        "users_accuracy": r(o["users_accuracy"][k]), "users_accuracy_ci95": r(o["users_accuracy_ci95"][k]),
                        "producers_accuracy": r(o["producers_accuracy"][k]),
                        "producers_accuracy_ci95": r(o["producers_accuracy_ci95"][k]),
                        "mapped_area_km2": r(o["mapped_area"][k], 2),
                        "estimated_area_km2": r(o["estimated_area"][k], 2),
                        "estimated_area_ci95_km2": r(o["estimated_area_ci95"][k], 2),
                        "worldcover_census_area_km2": r(o["worldcover_area_census"][k], 2),
                        "ci_covers_census": o["ci_covers_census"][k],
                    }
                    for k, nm in enumerate(class_names)
                },
            },
        }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"Done in {time.time() - t0:.0f} s. Metrics: results/metrics.json, figures: figures/")


if __name__ == "__main__":
    main()
