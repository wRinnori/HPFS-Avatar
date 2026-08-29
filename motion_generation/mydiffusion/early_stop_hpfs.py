#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

METRIC_NAMES = ["fgd", "fid", "diversity", "l1div", "srgr", "srgr_paper", "beat_align", "bc"]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def ensure_model_name_prefix(name: str):
    return name if name.startswith(("hpfs", "pfs")) else f"hpfs_{name}"


def parse_step_from_checkpoint(path: Path):
    m = re.search(r"model(\d+)\.pt$", path.name)
    if not m:
        return 0
    return int(m.group(1))


def discover_latest_train_dir(outputs_root: Path, exp_name: str, body: str, since_ts: float = None):
    pattern = f"*_{exp_name}_{body}"
    candidates = [p for p in outputs_root.glob(pattern) if p.is_dir()]
    if since_ts is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= (since_ts - 3.0)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def latest_checkpoint(run_dir: Path):
    ckpts = list(run_dir.glob("model*.pt"))
    if not ckpts:
        return None

    def step_of(path: Path):
        m = re.search(r"model(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=step_of, reverse=True)
    return ckpts[0]


def run_shell(cmd, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_resume_checkpoint(path_str: str):
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    if p.exists():
        return p
    return None


def metric_mode_default(metric_name: str):
    if metric_name in ["diversity", "l1div", "srgr", "srgr_paper", "beat_align", "bc"]:
        return "max"
    return "min"


def read_metric_value(metrics: dict, metric_name: str):
    value = metrics.get(metric_name)
    if value is None and metric_name == "fgd":
        value = metrics.get("fid")
    if value is None and metric_name == "beat_align":
        value = metrics.get("align")
    if value is None and metric_name == "bc":
        value = metrics.get("beat_align", metrics.get("align"))
    if value is None:
        raise RuntimeError(f"Metric '{metric_name}' not found in metrics json.")
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Chunked resume training + early stop for HPFS based on validation metric.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", type=str, required=True, help="Base training config yml")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="BEAT")
    parser.add_argument("--val-path", type=str, required=True, help="Validation split path used as tst_path during early-stop evaluation")
    parser.add_argument("--chunk-steps", type=int, default=5000, help="Train step increment per round")
    parser.add_argument("--max-steps", type=int, default=120000, help="Max total steps (global step) for this run")
    parser.add_argument("--save-iters", type=int, default=5000, help="Checkpoint save interval used in chunk training")
    parser.add_argument("--patience", type=int, default=3, help="Number of eval rounds without improvement before stop")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement amount")
    parser.add_argument("--metric", type=str, default="fgd", choices=METRIC_NAMES)
    parser.add_argument("--mode", type=str, default="min", choices=["min", "max"], help="Optimization direction for metric")
    parser.add_argument("--aux-metric", type=str, default="", help="Optional secondary metric that can also reset patience, e.g. diversity")
    parser.add_argument("--aux-mode", type=str, default="", choices=["", "min", "max"], help="Optimization direction for auxiliary metric")
    parser.add_argument("--aux-min-delta", type=float, default=0.0, help="Minimum improvement amount for auxiliary metric")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Existing checkpoint path for resume")
    parser.add_argument("--sample-max-len", type=int, default=60, help="Evaluate only first N files per round (speed/variance tradeoff)")
    parser.add_argument("--randm-num", type=int, default=120, help="Diversity random subset size in sample_linear")
    parser.add_argument("--eval-name-suffix", type=str, default="esval", help="Suffix for temporary eval config name")
    parser.add_argument("--eval-initial", dest="eval_initial", action="store_true", help="Evaluate initial resume checkpoint as round 0")
    parser.add_argument("--no-eval-initial", dest="eval_initial", action="store_false")
    parser.add_argument("--work-dir", type=str, default="outputs/auto_runs/early_stop")
    parser.add_argument("--results-csv", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.set_defaults(eval_initial=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    mydiff = root / "mydiffusion"
    outputs_root = root / "outputs"

    cfg_path = Path(args.config).resolve() if Path(args.config).is_absolute() else (mydiff / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    base_cfg = load_yaml(cfg_path)
    exp_name = ensure_model_name_prefix(str(base_cfg.get("name", "hpfs_run")))
    body = str(base_cfg.get("body", "whole"))

    work_dir = (mydiff / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    results_csv = Path(args.results_csv).resolve() if args.results_csv else work_dir / f"{exp_name}.early_stop.csv"
    summary_json = Path(args.summary_json).resolve() if args.summary_json else work_dir / f"{exp_name}.early_stop.summary.json"

    aux_metric = str(args.aux_metric).strip()
    if aux_metric:
        if aux_metric not in METRIC_NAMES:
            raise ValueError(f"Unsupported aux metric: {aux_metric}. Expected one of: {METRIC_NAMES}")
        aux_mode = args.aux_mode if args.aux_mode else metric_mode_default(aux_metric)
    else:
        aux_mode = ""

    resume_ckpt = resolve_resume_checkpoint(args.resume_checkpoint)
    if resume_ckpt is None:
        resume_ckpt = resolve_resume_checkpoint(str(base_cfg.get("resume_checkpoint", "")))

    current_step = parse_step_from_checkpoint(resume_ckpt) if resume_ckpt else 0
    if current_step >= args.max_steps:
        print(f"[WARN] resume step {current_step} >= max_steps {args.max_steps}; nothing to train.")

    best_metric = float("inf") if args.mode == "min" else float("-inf")
    best_ckpt = None
    best_step = -1
    best_aux_metric = float("inf") if aux_mode == "min" else float("-inf")
    best_aux_ckpt = None
    best_aux_step = -1
    no_improve = 0

    history = []

    if resume_ckpt is not None and args.eval_initial:
        eval_cfg = dict(base_cfg)
        eval_cfg["name"] = ensure_model_name_prefix(f"{exp_name}_{args.eval_name_suffix}")
        eval_cfg["model_path"] = str(resume_ckpt)
        eval_cfg["tst_path"] = str(Path(args.val_path).resolve())
        eval_cfg["randm_num"] = int(args.randm_num)
        eval_cfg_path = work_dir / f"{exp_name}.round00.eval.yml"
        dump_yaml(eval_cfg_path, eval_cfg)

        eval_cmd = [
            sys.executable,
            "sample_linear.py",
            "--config",
            str(eval_cfg_path),
            "--gpu",
            str(args.gpu),
            "--dataset",
            str(args.dataset),
            "--tst_path",
            str(Path(args.val_path).resolve()),
        ]
        if int(args.sample_max_len) > 0:
            eval_cmd.extend(["--max_len", str(int(args.sample_max_len))])
        eval_log = work_dir / f"{exp_name}.round00.eval.log"
        eval_rc = run_shell(eval_cmd, cwd=mydiff, log_path=eval_log)
        if eval_rc != 0:
            raise RuntimeError(f"Initial validation eval failed. See {eval_log}")

        metrics_path = mydiff / "outputs" / f"sample_dir_{eval_cfg['name']}_{body}" / "metrics_latest.json"
        if not metrics_path.exists():
            raise RuntimeError(f"metrics_latest.json not found after initial eval: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        metric_value = read_metric_value(metrics, args.metric)
        aux_metric_value = read_metric_value(metrics, aux_metric) if aux_metric else None

        best_metric = metric_value
        best_ckpt = resume_ckpt
        best_step = current_step
        if aux_metric:
            best_aux_metric = aux_metric_value
            best_aux_ckpt = resume_ckpt
            best_aux_step = current_step
        no_improve = 0

        row = {
            "round": 0,
            "target_step": current_step,
            "checkpoint_step": current_step,
            "checkpoint_path": str(resume_ckpt),
            "run_dir": "",
            "metric": args.metric,
            "metric_value": metric_value,
            "best_metric": best_metric,
            "best_step": best_step,
            "best_ckpt": str(best_ckpt),
            "aux_metric": aux_metric,
            "aux_metric_value": aux_metric_value if aux_metric else "",
            "best_aux_metric": best_aux_metric if aux_metric else "",
            "best_aux_step": best_aux_step if aux_metric else "",
            "best_aux_ckpt": str(best_aux_ckpt) if best_aux_ckpt else "",
            "improve_reason": "initial",
            "no_improve": no_improve,
            "patience": args.patience,
            "fgd": metrics.get("fgd", ""),
            "fid": metrics.get("fid", ""),
            "diversity": metrics.get("diversity", ""),
            "l1div": metrics.get("l1div", ""),
            "srgr": metrics.get("srgr", ""),
            "srgr_paper": metrics.get("srgr_paper", ""),
            "beat_align": metrics.get("beat_align", metrics.get("align", "")),
            "bc": metrics.get("bc", metrics.get("beat_align", metrics.get("align", ""))),
            "eval_metrics_path": str(metrics_path),
            "train_log": "",
            "eval_log": str(eval_log),
        }
        history.append(row)
        write_csv(results_csv, history)
        if aux_metric:
            print(
                f"[ROUND 0] step={current_step} {args.metric}={metric_value:.6f}, "
                f"{aux_metric}={aux_metric_value:.6f} (initial checkpoint)"
            )
        else:
            print(f"[ROUND 0] step={current_step} {args.metric}={metric_value:.6f} (initial checkpoint)")

    round_idx = 0
    while current_step < args.max_steps:
        round_idx += 1
        target_step = min(current_step + args.chunk_steps, args.max_steps)

        train_cfg = dict(base_cfg)
        train_cfg["name"] = exp_name
        train_cfg["max_num_steps"] = int(target_step)
        train_cfg["save_iters"] = int(args.save_iters)
        train_cfg["resume_checkpoint"] = str(resume_ckpt) if resume_ckpt else ""

        train_cfg_path = work_dir / f"{exp_name}.round{round_idx:02d}.train.yml"
        dump_yaml(train_cfg_path, train_cfg)

        train_cmd = [
            sys.executable,
            "train.py",
            "--config",
            str(train_cfg_path),
            "--gpu",
            str(args.gpu),
            "--dataset",
            str(args.dataset),
        ]
        train_log = work_dir / f"{exp_name}.round{round_idx:02d}.train.log"
        train_start = time.time()
        train_rc = run_shell(train_cmd, cwd=mydiff, log_path=train_log)
        if train_rc != 0:
            raise RuntimeError(f"Train failed at round {round_idx}. See {train_log}")

        run_dir = discover_latest_train_dir(outputs_root, exp_name, body, since_ts=train_start)
        if run_dir is None:
            run_dir = discover_latest_train_dir(outputs_root, exp_name, body, since_ts=None)
        if run_dir is None:
            raise RuntimeError(f"Cannot discover run dir for {exp_name} ({body}).")

        detected_ckpt = latest_checkpoint(run_dir)
        if detected_ckpt is None:
            raise RuntimeError(f"No checkpoint found under run dir: {run_dir}")

        prev_step = current_step
        resume_ckpt = detected_ckpt
        current_step = parse_step_from_checkpoint(detected_ckpt)
        if current_step < prev_step:
            raise RuntimeError(
                f"Checkpoint step moved backwards: prev={prev_step}, current={current_step}, ckpt={detected_ckpt}"
            )

        eval_cfg = dict(base_cfg)
        eval_cfg["name"] = ensure_model_name_prefix(f"{exp_name}_{args.eval_name_suffix}")
        eval_cfg["model_path"] = str(detected_ckpt)
        eval_cfg["tst_path"] = str(Path(args.val_path).resolve())
        eval_cfg["randm_num"] = int(args.randm_num)

        eval_cfg_path = work_dir / f"{exp_name}.round{round_idx:02d}.eval.yml"
        dump_yaml(eval_cfg_path, eval_cfg)

        eval_cmd = [
            sys.executable,
            "sample_linear.py",
            "--config",
            str(eval_cfg_path),
            "--gpu",
            str(args.gpu),
            "--dataset",
            str(args.dataset),
            "--tst_path",
            str(Path(args.val_path).resolve()),
        ]
        if int(args.sample_max_len) > 0:
            eval_cmd.extend(["--max_len", str(int(args.sample_max_len))])

        eval_log = work_dir / f"{exp_name}.round{round_idx:02d}.eval.log"
        eval_rc = run_shell(eval_cmd, cwd=mydiff, log_path=eval_log)
        if eval_rc != 0:
            raise RuntimeError(f"Validation eval failed at round {round_idx}. See {eval_log}")

        metrics_path = mydiff / "outputs" / f"sample_dir_{eval_cfg['name']}_{body}" / "metrics_latest.json"
        if not metrics_path.exists():
            raise RuntimeError(f"metrics_latest.json not found after eval: {metrics_path}")

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        metric_value = read_metric_value(metrics, args.metric)
        aux_metric_value = read_metric_value(metrics, aux_metric) if aux_metric else None

        if args.mode == "min":
            primary_improved = metric_value < (best_metric - args.min_delta)
        else:
            primary_improved = metric_value > (best_metric + args.min_delta)

        aux_improved = False
        if aux_metric:
            if aux_mode == "min":
                aux_improved = aux_metric_value < (best_aux_metric - args.aux_min_delta)
            else:
                aux_improved = aux_metric_value > (best_aux_metric + args.aux_min_delta)

        if primary_improved:
            best_metric = metric_value
            best_ckpt = detected_ckpt
            best_step = current_step
        if aux_improved:
            best_aux_metric = aux_metric_value
            best_aux_ckpt = detected_ckpt
            best_aux_step = current_step

        improved = primary_improved or aux_improved
        if improved:
            no_improve = 0
            if primary_improved and aux_improved:
                improve_reason = "both"
            elif primary_improved:
                improve_reason = args.metric
            else:
                improve_reason = aux_metric
        else:
            no_improve += 1
            improve_reason = "none"

        row = {
            "round": round_idx,
            "target_step": target_step,
            "checkpoint_step": current_step,
            "checkpoint_path": str(detected_ckpt),
            "run_dir": str(run_dir),
            "metric": args.metric,
            "metric_value": metric_value,
            "best_metric": best_metric,
            "best_step": best_step,
            "best_ckpt": str(best_ckpt) if best_ckpt else "",
            "aux_metric": aux_metric,
            "aux_metric_value": aux_metric_value if aux_metric else "",
            "best_aux_metric": best_aux_metric if aux_metric else "",
            "best_aux_step": best_aux_step if aux_metric else "",
            "best_aux_ckpt": str(best_aux_ckpt) if best_aux_ckpt else "",
            "improve_reason": improve_reason,
            "no_improve": no_improve,
            "patience": args.patience,
            "fgd": metrics.get("fgd", ""),
            "fid": metrics.get("fid", ""),
            "diversity": metrics.get("diversity", ""),
            "l1div": metrics.get("l1div", ""),
            "srgr": metrics.get("srgr", ""),
            "srgr_paper": metrics.get("srgr_paper", ""),
            "beat_align": metrics.get("beat_align", metrics.get("align", "")),
            "bc": metrics.get("bc", metrics.get("beat_align", metrics.get("align", ""))),
            "eval_metrics_path": str(metrics_path),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
        }
        history.append(row)
        write_csv(results_csv, history)

        if aux_metric:
            print(
                f"[ROUND {round_idx}] step={current_step} {args.metric}={metric_value:.6f} "
                f"(best={best_metric:.6f}), {aux_metric}={aux_metric_value:.6f} "
                f"(best={best_aux_metric:.6f}) reason={improve_reason} "
                f"no_improve={no_improve}/{args.patience}"
            )
        else:
            print(
                f"[ROUND {round_idx}] step={current_step} {args.metric}={metric_value:.6f} "
                f"best={best_metric:.6f} no_improve={no_improve}/{args.patience}"
            )

        if no_improve >= args.patience:
            print("[EARLY STOP] patience reached.")
            break

    summary = {
        "config": str(cfg_path),
        "exp_name": exp_name,
        "body": body,
        "metric": args.metric,
        "mode": args.mode,
        "aux_metric": aux_metric,
        "aux_mode": aux_mode,
        "aux_min_delta": args.aux_min_delta,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "chunk_steps": args.chunk_steps,
        "max_steps": args.max_steps,
        "best_metric": best_metric,
        "best_step": best_step,
        "best_ckpt": str(best_ckpt) if best_ckpt else "",
        "best_aux_metric": best_aux_metric if aux_metric else "",
        "best_aux_step": best_aux_step if aux_metric else "",
        "best_aux_ckpt": str(best_aux_ckpt) if best_aux_ckpt else "",
        "history_csv": str(results_csv),
        "rounds": len(history),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved history CSV: {results_csv}")
    print(f"Saved summary JSON: {summary_json}")
    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}")
    if aux_metric and best_aux_ckpt:
        print(f"Best {aux_metric} checkpoint: {best_aux_ckpt}")


if __name__ == "__main__":
    main()
