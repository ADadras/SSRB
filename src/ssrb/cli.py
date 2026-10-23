"""Command line interface for preparing, fitting and evaluating SSRB."""
import argparse
import csv
import json
import platform
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="ssrb")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    q = sub.add_parser("check-profile")
    from .profiles import PROFILE_NAMES
    q.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    q = sub.add_parser("prepare-bsds")
    q.add_argument("--data-root", required=True)
    q.add_argument("--output", required=True)
    for dataset in ("ima", "lidc", "riga"):
        q = sub.add_parser("prepare-"+dataset)
        q.add_argument("--zip-path" if dataset == "riga" else "--data-root", required=True)
        q.add_argument("--output", required=True)
    q = sub.add_parser("bootstrap-representation")
    q.add_argument("--units", required=True, help="CSV of paired, visible-choice-averaged analysis units")
    q.add_argument("--output", required=True)
    q = sub.add_parser("analyze-representation")
    for option in ("evaluations", "output"):
        q.add_argument("--"+option, required=True)
    q = sub.add_parser("analyze-assignment")
    q.add_argument("--profile", choices=("visible", "lidc", "capacity"), required=True)
    for option in ("evaluations", "output"):
        q.add_argument("--"+option, required=True)
    q = sub.add_parser("make-visible-manifests")
    for name in ("source", "data-root", "output"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("extract-features")
    for name in ("root", "manifest", "output", "dino-repo", "weights"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--batch-size", type=int, default=16)
    q.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3")
    q = sub.add_parser("import-features")
    q.add_argument("--source", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--manifest")
    q.add_argument("--split", choices=["train", "val", "test"])
    q = sub.add_parser("seal-inputs")
    q.add_argument("--root", required=True)
    q.add_argument("--manifest", required=True)
    q = sub.add_parser("verify-features")
    for option in ("source", "reference", "endpoint", "split", "backbone"):
        q.add_argument("--"+option, required=True)
    for name in ("train-semantic", "train-ssrb"):
        q = sub.add_parser(name)
        for option in ("root", "manifest", "output"):
            q.add_argument("--"+option, required=True)
        q.add_argument("--config")
        q.add_argument("--cache-inputs", action="store_true", help="cache the same decoded visible inputs in memory")
        q.add_argument("--isolate", action="store_true", help="require Linux Landlock read isolation")
        q.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch operations")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--device", default="cpu")
        if name == "train-ssrb":
            q.add_argument("--anchor", required=True)
    q = sub.add_parser("train-comparison")
    from .comparisons import METHODS
    q.add_argument("--method", choices=METHODS, required=True)
    for name in ("root", "manifest", "output"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--config")
    q.add_argument("--anchor")
    q.add_argument("--cimd-root")
    q.add_argument("--isolate", action="store_true")
    q.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch operations")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="cpu")
    q = sub.add_parser("infer-comparison")
    for name in ("checkpoint", "root", "manifest", "output"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--feature-file")
    q.add_argument("--cimd-root")
    q.add_argument("--device", default="cpu")
    q.add_argument("--batch-size", type=int)
    q.add_argument("--sampling-seed", type=int)
    q.add_argument("--deterministic", action="store_true")
    for name in ("align-heads", "infer-posthoc"):
        q = sub.add_parser(name)
        for option in ("semantic-checkpoint", "raw-checkpoint", "root", "manifest", "output"):
            q.add_argument("--"+option, required=True)
        q.add_argument("--device", default="cpu")
        q.add_argument("--batch-size", type=int, default=12)
        if name == "infer-posthoc":
            q.add_argument("--feature-file", required=True)
            q.add_argument("--alignment")
    q = sub.add_parser("infer-sam2")
    for name in ("root", "manifest", "output", "sam-repo", "weights"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    q.add_argument("--single-predictions")
    q.add_argument("--retain-full-pool", action="store_true")
    q = sub.add_parser("evaluate-sam2-pool")
    for option in ("predictions", "root", "manifest", "output"):
        q.add_argument("--"+option, required=True)
    q = sub.add_parser("analyze-sharing")
    for option in ("evaluations", "output"):
        q.add_argument("--"+option, required=True)
    q.add_argument("--replicates", type=int, default=10000)
    q.add_argument("--seed", type=int, default=20260814)
    q = sub.add_parser("train-sharing")
    for option in ("root", "manifest", "output", "method"):
        q.add_argument("--"+option, required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="cpu")
    q.add_argument("--deterministic", action="store_true")
    q.add_argument("--isolate", action="store_true")
    q = sub.add_parser("benchmark")
    for option in ("root", "manifest", "output", "method"):
        q.add_argument("--"+option, required=True)
    for option in ("checkpoint", "dino-repo", "dino-weights", "cimd-root", "sam-repo", "sam-weights"):
        q.add_argument("--"+option)
    q.add_argument("--device", default="cuda")
    q.add_argument("--seed", type=int)
    q.add_argument("--subset-size", type=int, default=8)
    q.add_argument("--warmup-runs", type=int, default=100)
    q.add_argument("--timed-runs", type=int, default=500)
    q.add_argument("--smoke", action="store_true")
    for name in ("train-partial", "infer-partial"):
        q = sub.add_parser(name)
        for option in ("root", "manifest", "output", "dino-repo", "weights"):
            q.add_argument("--"+option, required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--deterministic", action="store_true")
        if name == "train-partial":
            q.add_argument("--seed", type=int, default=0)
        else:
            q.add_argument("--checkpoint", required=True)
            q.add_argument("--batch-size", type=int, default=12)
    q = sub.add_parser("plot-paper")
    q.add_argument("--profile", choices=("representation", "budget", "probunet", "mechanism", "qualitative", "bsds-construction", "assignment-examples"), required=True)
    for option in ("source", "output"):
        q.add_argument("--"+option, required=True)
    q = sub.add_parser("analyze-residuals")
    for option in ("diagnostics", "output"):
        q.add_argument("--"+option, required=True)
    q.add_argument("--replicates", type=int, default=10000)
    q.add_argument("--seed", type=int, default=20260815)
    q = sub.add_parser("diagnose-residuals")
    for option in ("checkpoint", "root", "manifest", "feature-file", "output"):
        q.add_argument("--"+option, required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--batch-size", type=int, default=12)
    q = sub.add_parser("analyze-paper")
    q.add_argument("--evaluations", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--profile", choices=["controls", "probunet", "dinov2", "partial"], required=True)
    q.add_argument("--replicates", type=int, default=10000)
    q.add_argument("--seed", type=int)
    q = sub.add_parser("plot-results")
    q.add_argument("--summary", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("plot-banks")
    for name in ("predictions", "root", "manifest", "output"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--image-id", action="append", required=True)
    q = sub.add_parser("infer")
    q.add_argument("--deterministic", action="store_true")
    for name in ("checkpoint", "root", "manifest", "feature-file", "output"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--batch-size", type=int, default=12)
    for name in ("make-extent-banks", "combine-ensemble"):
        q = sub.add_parser(name)
        q.add_argument("--predictions", required=True, action="append" if name=="combine-ensemble" else "store")
        q.add_argument("--output", required=True)
        if name == "make-extent-banks":
            q.add_argument("--root", required=True)
            q.add_argument("--manifest", required=True)
    q = sub.add_parser("evaluate")
    for name in ("predictions", "root", "manifest", "output"):
        q.add_argument("--"+name, required=True)
    q.add_argument("--ged", action="store_true")
    q = sub.add_parser("make-paper-tables")
    q.add_argument("--evaluation", action="append", required=True)
    q.add_argument("--status", choices=["verified-original-checkpoint", "new-training-reproduction", "partial"], required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("aggregate-paper-table")
    q.add_argument("--evaluations", required=True, help="JSON list of evaluation directories and optimizer_seed values")
    q.add_argument("--output", required=True)
    q.add_argument("--optimizer-seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="required optimizer seeds; use 0 for baselines fitted once per visible choice")
    q.add_argument("--averaging", choices=["visible-first", "run-means"], default="visible-first",
                   help="published visible-membership averaging or equal averaging of complete run means")
    q.add_argument("--status", choices=["verified-original-checkpoint", "new-training-reproduction", "partial"], required=True)
    for name in ("smoke", "prepare-demo"):
        q = sub.add_parser(name)
        q.add_argument("--output", required=True)
        if name == "smoke":
            q.add_argument("--device", default="cpu")
    args = vars(p.parse_args(argv))
    command = args.pop("command")
    if command == "doctor":
        import torch
        from . import __version__
        result = dict(python=platform.python_version(), torch=str(torch.__version__), ssrb=__version__,
                      cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    elif command == "check-profile":
        from .profiles import check_profile
        result = check_profile(args["profile"])
        print(json.dumps(result, sort_keys=True, default=str))
        if result["differences"]:
            raise SystemExit(1)
        return
    elif command == "prepare-bsds":
        from .prepare_bsds import prepare_bsds
        result = prepare_bsds(**args)
    elif command in {"prepare-ima", "prepare-lidc", "prepare-riga"}:
        from importlib import import_module
        dataset = command.split("-")[1]
        result = getattr(import_module("ssrb.prepare_"+dataset), "prepare_"+dataset)(**args)
    elif command == "analyze-representation":
        from .representation_statistics import run
        result = run(**args)
    elif command == "analyze-assignment":
        from .assignment_statistics import run
        result = run(**args)
    elif command == "bootstrap-representation":
        import pandas as pd
        from .legacy_statistics import bootstrap_means
        from .data import write_json
        value, record = bootstrap_means(pd.read_csv(args["units"]), replicates=10000, seed=20260814)
        result = {name: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in info.items()} for name, info in value.items()}
        if Path(args["output"]).exists():
            raise FileExistsError(args["output"])
        write_json(args["output"], result)
        result = record
    elif command == "make-visible-manifests":
        from .data import freeze_manifests
        result = freeze_manifests(**args)
    elif command == "extract-features":
        from .features import extract
        result = extract(**args)
    elif command == "import-features":
        from .features import import_cache
        result = import_cache(**args)
    elif command == "seal-inputs":
        from .data import seal_inputs
        result = seal_inputs(**args)
    elif command == "verify-features":
        from .features import verify
        result = verify(**args)
    elif command in {"align-heads", "infer-posthoc"}:
        from .posthoc import align, infer
        value = (align if command == "align-heads" else infer)(**args)
        result = value if command == "align-heads" else {"images": len(value)}
    elif command == "infer-comparison":
        from .comparisons import infer
        result = {"images": len(infer(**args))}
    elif command in {"train-comparison", "train-sharing"}:
        isolate = args.pop("isolate")
        if isolate:
            if args.get("cimd_root"):
                raise ValueError("CIMD requires its external package; use a container with only the visible root mounted")
            from .isolation import isolate_trainer
            abi = isolate_trainer(**{k: v for k, v in args.items() if k in {"root", "manifest", "output", "config", "anchor"}})
            print(json.dumps({"read_isolation": "Landlock", "abi": abi}), flush=True)
            args["_reserved_output"] = True
        if command == "train-sharing":
            from .sharing import train
        else:
            from .comparisons import train
        ck, history = train(**args)
        result = dict(best_epoch=ck["best_epoch"], epochs=len(history), best_objective=ck["best_objective"])
    elif command in {"train-partial", "infer-partial"}:
        from .partial import train, infer
        result = (train if command == "train-partial" else infer)(**args)
    elif command == "analyze-sharing":
        from .sharing_statistics import analyze
        result = analyze(**args)
    elif command == "plot-paper":
        from .publication_plots import run
        result = run(**args)
    elif command == "analyze-residuals":
        from .diagnostic_statistics import analyze
        result = analyze(**args)
    elif command == "diagnose-residuals":
        from .diagnostics import run
        result = run(**args)
    elif command == "analyze-paper":
        from .analysis import run
        result = run(**args)
    elif command == "plot-results":
        from .figures import results
        result = results(**args)
    elif command == "plot-banks":
        from .figures import banks
        result = banks(**args)
    elif command.startswith("train-"):
        isolate = args.pop("isolate")
        if isolate:
            from .isolation import isolate_trainer
            abi = isolate_trainer(**{k: v for k, v in args.items() if k in {"root", "manifest", "output", "config", "anchor"}})
            print(json.dumps({"read_isolation": "Landlock", "abi": abi}), flush=True)
            args["_reserved_output"] = True
        from .training import train
        ck, history = train("semantic" if command == "train-semantic" else "residual", **args)
        result = dict(best_epoch=ck["best_epoch"], epochs=len(history), best_objective=ck["best_objective"])
    elif command == "infer-sam2":
        from .proposals import infer
        result = {"images": len(infer(**args))}
    elif command == "evaluate-sam2-pool":
        from .proposals import evaluate_pool
        result = evaluate_pool(**args)
    elif command == "benchmark":
        from .runtime import benchmark
        result = benchmark(**args)
    elif command == "infer":
        from .inference import infer
        result = {"images": len(infer(**args))}
    elif command in {"make-extent-banks", "combine-ensemble"}:
        from .extent_banks import extents, ensemble
        result = (extents if command=="make-extent-banks" else ensemble)(**args)
    elif command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(**args)
    elif command == "make-paper-tables":
        rows = []
        for name in args["evaluation"]:
            summary = json.loads((Path(name)/"summary.json").read_text())
            for endpoint, metrics in summary.items():
                rows.append(dict(endpoint=endpoint, status=args["status"], evaluation=str(name), **metrics))
        output = Path(args["output"])
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        result = str(output)
    elif command == "aggregate-paper-table":
        from .tables import aggregate
        result = aggregate(**args)
    elif command == "prepare-demo":
        from .synthetic import prepare_demo
        result = str(prepare_demo(**args))
    else:
        from .synthetic import smoke
        result = smoke(**args)
    if result is not None:
        print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
