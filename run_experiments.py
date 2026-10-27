"""Reproducible validation search and one final test comparison."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

# Bound BLAS threads before importing numeric libraries; small ALS solves use one thread.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from data_utils import ROOT, make_matrix, prepare_data
from evaluation import evaluate
from models.item_knn import ItemKNN
from models.ials import IALS
from models.custom_ials import CustomIALS
from models.ease import EASE
from models.fm import FactorizationMachine

RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
CLASSES = {"ItemKNN": ItemKNN, "iALS": IALS, "EASE": EASE,
           "FM": FactorizationMachine, "Custom iALS": CustomIALS, "FM IDs only": FactorizationMachine}
# Deliberately small, predetermined grids; NDCG@10 alone selects settings.
GRIDS = {
    "ItemKNN": [{"neighbors": n, "shrinkage": s} for n, s in ((50, 0), (100, 0), (100, 10))],
    "iALS": [{"factors": f, "reg": r, "alpha": 20, "iterations": 12}
             for f, r in ((32, 1.0), (32, 10.0), (64, 10.0))],
    "EASE": [{"reg": r} for r in (100, 500, 1000)],
    "FM": [{"factors": 32, "epochs": e, "lr": 0.003, "metadata": True} for e in (5, 10)],
    "Custom iALS": [{"factors": f, "reg": r, "alpha": 20, "iterations": 12, "beta": 0.5}
                    for f, r in ((32, 1.0), (32, 10.0), (64, 10.0))],
    "FM IDs only": [{"factors": 32, "epochs": e, "lr": 0.003, "metadata": False} for e in (5, 10)],
}


def signature(bundle):
    files = [ROOT / "data_utils.py", ROOT / "evaluation.py", ROOT / "run_experiments.py"]
    files += sorted((ROOT / "models").glob("*.py"))
    blob = b"".join(f.read_bytes() for f in files)
    blob += json.dumps(bundle["audit"]["sha256"], sort_keys=True).encode()
    blob += json.dumps(GRIDS, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def timed_fit(name, params, history, frame, metadata):
    start = time.perf_counter()
    model = CLASSES[name](**params)
    threads = 1 if name in ("iALS", "Custom iALS") else 4
    with threadpool_limits(limits=threads):
        model.fit(history, frame, metadata)
    return model, time.perf_counter() - start


def run_validation(bundle, force=False):
    RESULTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    fingerprint = signature(bundle)
    path = RESULTS / "validation.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows = existing.get("trials", []) if existing.get("signature") == fingerprint and not force else []
    history = make_matrix(bundle["train"], bundle["shape"])
    for name, configs in GRIDS.items():
        for params in configs:
            if any(r["model"] == name and r["params"] == params for r in rows):
                continue
            print(f"Validation: {name} {params}", flush=True)
            model, seconds = timed_fit(name, params, history, bundle["train"], bundle["metadata"])
            with threadpool_limits(limits=4):
                metrics, _ = evaluate(model, history, bundle["validation"], bundle["eligible_users"])
            row = {"model": name, "params": params, "fit_seconds": seconds, **metrics}
            if hasattr(model, "loss_history"):
                row["loss_history"] = model.loss_history
            rows.append(row)
            save_json(path, {"signature": fingerprint, "trials": rows})
            print(row, flush=True)
            del model
            gc.collect()
    selected = {}
    for name in GRIDS:
        best = max((r for r in rows if r["model"] == name), key=lambda r: r["NDCG@10"])
        selected[name] = best["params"]
    save_json(RESULTS / "selected.json", {"signature": fingerprint, "params": selected})
    return pd.DataFrame(rows), selected


def run_final(bundle, selected, force=False):
    path = RESULTS / "test.json"
    fingerprint = signature(bundle)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows = previous.get("models", []) if previous.get("signature") == fingerprint and not force else []
    # Final models have access to all warm train+validation events, never test.
    frame = pd.concat([bundle["train"], bundle["validation"]], ignore_index=True)
    history = make_matrix(frame, bundle["shape"])
    for name, params in selected.items():
        if any(r["model"] == name and r["params"] == params for r in rows):
            continue
        print(f"Final test: {name} {params}", flush=True)
        model, seconds = timed_fit(name, params, history, frame, bundle["metadata"])
        with threadpool_limits(limits=4):
            metrics, per_user = evaluate(model, history, bundle["test"], bundle["eligible_users"])
        row = {"model": name, "params": params, "fit_seconds": seconds, **metrics}
        if hasattr(model, "loss_history"):
            row["loss_history"] = model.loss_history
        rows.append(row)
        save_json(path, {"signature": fingerprint, "models": rows})
        save_json(ARTIFACTS / (name.replace(" ", "_") + "_test_users.json"), per_user)
        print(row, flush=True)
        del model
        gc.collect()
    return pd.DataFrame(rows)


def run(force=False):
    bundle = prepare_data()
    save_json(RESULTS / "data_audit.json", bundle["audit"])
    versions = {n: importlib.metadata.version(n) for n in
                ("numpy", "scipy", "pandas", "torch", "matplotlib", "threadpoolctl")}
    save_json(RESULTS / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
              "logical_cpus": os.cpu_count(), "torch_cuda": torch.cuda.is_available(),
              "versions": versions, "seed": 42})
    print(json.dumps(bundle["audit"], ensure_ascii=False, indent=2), flush=True)
    validation, selected = run_validation(bundle, force)
    test = run_final(bundle, selected, force)
    print(test[["model", "NDCG@10", "MAP@10", "Recall@10", "users"]].to_string(index=False), flush=True)
    return bundle, validation, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute, ignoring matching local results")
    run(parser.parse_args().force)
