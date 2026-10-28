"""
Assignment 2: training pipeline for hard-hat compliance.

    uv run python main.py train

Start to finish, no manual steps:
  1. Download joseph-nelson/hard-hat-workers v5 (head, helmet) from Roboflow.
  2. Describe the dataset: instances per class, split sizes, object sizes.
  3. Finetune yolo11n from the COCO-pretrained checkpoint; copy best.pt to weights/.
  4. Score the finetuned model, COCO yolo11n zero-shot and a predict-nothing
     baseline on the same test split, per class and per frame.
  5. Sweep the confidence threshold against the cost model in costs.py,
     pick the cheapest single threshold and the cheapest review band.
  6. Write everything app.py needs into artifacts/.

Environment overrides, for smoke tests only (the submitted run uses defaults):
  EPOCHS, FRACTION     shorten training
  REUSE_WEIGHTS=1      skip training and re-evaluate the committed weights
"""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from costs import DEFAULT_COSTS, CostModel, alternatives, best_band, evaluate_band, single_threshold_sweep
from evaluation import detection_confusion, detection_metrics, frame_confusion, frame_scores, read_yolo_labels

ROOT = Path(__file__).resolve().parent
WORKSPACE, PROJECT, VERSION = "joseph-nelson", "hard-hat-workers", 5
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-{VERSION}"
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

BASE_MODEL = "yolo11n.pt"
TRAIN_ARGS = dict(
    epochs=int(os.environ.get("EPOCHS", 40)),
    fraction=float(os.environ.get("FRACTION", 1.0)),
    imgsz=416,        # source images are ~300 px; 416 upsamples small heads a little
    batch=32,
    patience=15,
    seed=0,
    deterministic=True,
    workers=4,
    save_period=5,    # extra epochN.pt snapshot every 5 epochs; last.pt is saved every epoch
    plots=True,
)
POSITIVE_CLASS = 0    # `head`: a visible head with no helmet
EVAL_CONF = 0.001     # keep almost every box so the threshold sweep sees them all
COCO_PERSON = 0


def pick_device() -> str:
    # Apple MPS is deliberately skipped: on torch 2.11 / ultralytics 8.4 the
    # training losses climb from a pretrained start on MPS while the same
    # batches on CPU fall (see REPORT.md), so MPS gradients are not trusted.
    return "0" if torch.cuda.is_available() else "cpu"


def download_dataset() -> Path:
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already at {DATA_DIR}")
    else:
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR))
    # Roboflow writes relative paths that Ultralytics resolves against its own
    # datasets_dir; rewrite them as absolute so training finds the images.
    yaml_path = DATA_DIR / "data_abs.yaml"
    yaml_path.write_text(
        f"path: {DATA_DIR}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: 2\nnames: ['head', 'helmet']\n")
    return yaml_path


def describe_dataset(names: dict[int, str]) -> dict:
    stats = {}
    for split in ["train", "valid", "test"]:
        gt = read_yolo_labels(DATA_DIR / split / "labels")
        images = sorted((DATA_DIR / split / "images").iterdir())
        sizes = np.array([Image.open(p).size for p in images])
        per_class = {}
        for cls, name in names.items():
            boxes = [b for v in gt.values() for b in v if b[0] == cls]
            side = np.array([np.sqrt((b[3] - b[1]) * (b[4] - b[2])) for b in boxes])
            per_class[name] = dict(
                instances=len(boxes),
                images_containing=sum(any(b[0] == cls for b in v) for v in gt.values()),
                median_rel_side=float(np.median(side)),
                p10_rel_side=float(np.percentile(side, 10)),
                share_small_lt_5pct=float(np.mean(side < 0.05)),
            )
        stats[split] = dict(
            images=len(images),
            frames_with_bare_head=float(np.mean([any(b[0] == POSITIVE_CLASS for b in v)
                                                 for v in gt.values()])),
            boxes_per_image=float(np.mean([len(v) for v in gt.values()])),
            median_width=int(np.median(sizes[:, 0])), median_height=int(np.median(sizes[:, 1])),
            per_class=per_class,
        )
    return stats


def train(yaml_path: Path) -> None:
    if os.environ.get("REUSE_WEIGHTS") == "1" and WEIGHTS.exists():
        print(f"REUSE_WEIGHTS=1: skipping training, evaluating {WEIGHTS}")
        return
    # last.pt is written after every epoch. If a previous run was interrupted
    # (it still carries its epoch counter), pick up where it stopped instead of
    # starting over; a finished run has epoch -1 and is not resumed.
    last = ROOT / "runs" / "hardhat" / "weights" / "last.pt"
    if last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1) >= 0:
        print(f"Resuming interrupted training from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(BASE_MODEL)
        model.train(data=str(yaml_path), device=pick_device(), project=str(ROOT / "runs"),
                    name="hardhat", exist_ok=True, **TRAIN_ARGS)
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    shutil.copy(best.parent.parent / "results.csv", ARTIFACTS / "training_results.csv")
    print(f"Saved {WEIGHTS}")


def predict(model: YOLO, image_dir: Path, keep=None) -> dict:
    """Run a model over a folder; return normalised boxes per image stem.

    One image per call, as a camera feed would be scored. Batching mixed-size
    images makes Ultralytics pad each to a full square, which shifts scores
    (0.04 vs 0.38 on one test frame) away from what the dashboard sees.
    """
    preds = {}
    for path in sorted(image_dir.iterdir()):
        r = model.predict(str(path), conf=EVAL_CONF, imgsz=TRAIN_ARGS["imgsz"],
                          device=pick_device(), verbose=False)[0]
        b = r.boxes
        rows = [(int(c), float(s), *map(float, xyxy))
                for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]
        preds[path.stem] = keep(rows) if keep else rows
    return preds


def coco_as_head(rows: list[tuple]) -> list[tuple]:
    """Zero-shot COCO baseline. COCO has `person` but no notion of a helmet, so
    every person is treated as a possible bare head and its box is shrunk to
    the top of the person, where a head would be. This is the most generous
    reading of a model that cannot see helmets."""
    out = []
    for c, s, x1, y1, x2, y2 in rows:
        if c != COCO_PERSON:
            continue
        w, h = x2 - x1, y2 - y1
        hw = min(w * 0.5, h * 0.25) if h > w * 1.5 else w * 0.6
        cx = (x1 + x2) / 2
        out.append((POSITIVE_CLASS, s, cx - hw / 2, y1, cx + hw / 2, y1 + hw * 1.2))
    return out


def sensitivity(scores, labels) -> list[dict]:
    """How the cheapest single threshold moves when the shakiest inputs move."""
    rows = []
    for fn in [30, 60, 120, 240, 500]:
        for base in [0.01, 0.05, 0.20]:
            c = CostModel(fn=fn, base_rate=base)
            sweep = single_threshold_sweep(scores, labels, c)
            best = min(sweep, key=lambda r: r["cost_per_1000"])
            band = best_band(scores, labels, c)
            rows.append(dict(fn=fn, base_rate=base, best_threshold=best["t_low"],
                             model_alone_per_1000=best["cost_per_1000"],
                             band_low=band["t_low"], band_high=band["t_high"],
                             band_per_1000=band["cost_per_1000"],
                             band_review_share=band["review_share"],
                             **{k: v for k, v in alternatives(c).items() if k != "human_checks_everything_staff"}))
    return rows


def plot_cost_curve(sweep: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["t_low"], sweep["cost_per_1000"], color="#1f4e79")
    ax.axvline(chosen, color="#c0392b", ls="--", label=f"chosen t = {chosen:.2f}")
    ax.set_xlabel("confidence threshold for `head`")
    ax.set_ylabel("expected cost per 1,000 frames (EUR)")
    ax.set_title("Model decides alone: cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    yaml_path = download_dataset()
    names = {0: "head", 1: "helmet"}

    stats = describe_dataset(names)
    (ARTIFACTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: {"images": v["images"], **{n: c["instances"] for n, c in v["per_class"].items()}}
                      for k, v in stats.items()}))

    train(yaml_path)

    test_images, test_labels = DATA_DIR / "test" / "images", DATA_DIR / "test" / "labels"
    gt = read_yolo_labels(test_labels)
    finetuned = predict(YOLO(str(WEIGHTS)), test_images)
    coco = predict(YOLO(BASE_MODEL), test_images, keep=coco_as_head)
    nothing = {k: [] for k in gt}

    # Frame-level decision: cost sweep and operating point.
    images, scores, labels = frame_scores(gt, finetuned, POSITIVE_CLASS)
    _, coco_scores, _ = frame_scores(gt, coco, POSITIVE_CLASS)
    sweep = pd.DataFrame(single_threshold_sweep(scores, labels))
    chosen = sweep.loc[sweep["cost_per_1000"].idxmin()]
    band = best_band(scores, labels)
    staff = DEFAULT_COSTS.reviewer_hours_per_day * 3600 / DEFAULT_COSTS.review_seconds
    band_one_person = best_band(scores, labels, max_review_share=staff / DEFAULT_COSTS.frames_per_day)
    # The unconstrained band reviews ~95% of frames, i.e. a person watches
    # almost everything. Reviewer accuracy does not survive that volume of
    # mostly empty frames, so price both bands, and checking everything, as
    # reviewer accuracy falls. The shipped band is the one-reviewer band.
    fatigue = []
    for h in [0.97, 0.95, 0.93, 0.90]:
        c = replace(DEFAULT_COSTS, human_accuracy=h)
        fatigue.append(dict(
            human_accuracy=h,
            shipped_band=evaluate_band(scores, labels, band_one_person["t_low"],
                                       band_one_person["t_high"], c)["cost_per_1000"],
            unconstrained_band=evaluate_band(scores, labels, band["t_low"], band["t_high"], c)["cost_per_1000"],
            human_checks_everything=alternatives(c)["human_checks_everything"]))
    coco_sweep = pd.DataFrame(single_threshold_sweep(coco_scores, labels))
    coco_best = coco_sweep.loc[coco_sweep["cost_per_1000"].idxmin()]

    pd.DataFrame(dict(image=images, score=scores, label=labels, coco_score=coco_scores)).to_csv(
        ARTIFACTS / "test_frames.csv", index=False)
    sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)
    pd.DataFrame(sensitivity(scores, labels)).to_csv(ARTIFACTS / "sensitivity.csv", index=False)
    plot_cost_curve(sweep, float(chosen["t_low"]), ARTIFACTS / "cost_curve.png")
    (ARTIFACTS / "test_predictions.json").write_text(json.dumps(
        {k: [[c, round(s, 4), *(round(x, 5) for x in xyxy)] for c, s, *xyxy in v if s >= 0.01]
         for k, v in finetuned.items()}))
    # Test labels too, so the dashboard's box-level confusion matrix works on a
    # machine that has the committed artifacts but never downloaded the data.
    (ARTIFACTS / "test_ground_truth.json").write_text(json.dumps(
        {k: [[c, *(round(x, 5) for x in xyxy)] for c, *xyxy in v] for k, v in gt.items()}))

    # Detection-level, per class, at the chosen threshold.
    conf = float(chosen["t_low"])
    models = {"finetuned_yolo11n": finetuned, "coco_yolo11n_zero_shot": coco, "predict_nothing": nothing}
    detection = {name: detection_metrics(gt, p, names, conf) for name, p in models.items()}
    # COCO and predict-nothing only know one class; their helmet rows are
    # zero by construction and say so rather than pretending to be measured.
    confusion = detection_confusion(gt, finetuned, len(names), conf)

    frame = {
        "finetuned_yolo11n": frame_confusion(scores, labels, conf),
        "coco_yolo11n_zero_shot": frame_confusion(coco_scores, labels, float(coco_best["t_low"])),
        "majority_class_all_pass": frame_confusion(np.zeros_like(scores), labels, 0.5),
    }
    frame["coco_yolo11n_zero_shot"]["threshold"] = float(coco_best["t_low"])
    frame["coco_yolo11n_zero_shot"]["cost_per_1000"] = float(coco_best["cost_per_1000"])

    ultra = YOLO(str(WEIGHTS)).val(data=str(yaml_path), split="test", imgsz=TRAIN_ARGS["imgsz"],
                                   device=pick_device(), project=str(ROOT / "runs"), name="test_val",
                                   exist_ok=True, plots=False, verbose=False)
    metrics = dict(
        dataset=dict(workspace=WORKSPACE, project=PROJECT, version=VERSION),
        names=names, positive_class=names[POSITIVE_CLASS],
        train_args=dict(base_model=BASE_MODEL, device=pick_device(), **TRAIN_ARGS),
        detection=detection,
        detection_confusion=dict(conf=conf, labels=[*names.values(), "background"],
                                 matrix=confusion.tolist(), rows="predicted", cols="true"),
        frame=frame,
        ultralytics_test_check=dict(map50=float(ultra.box.map50), map50_95=float(ultra.box.map),
                                    per_class_ap50={names[i]: float(a) for i, a in
                                                    zip(ultra.box.ap_class_index, ultra.box.ap50)}),
        costs=DEFAULT_COSTS.to_dict(),
        operating_point=dict(model_alone=chosen.to_dict(), band=band,
                             band_capped_one_reviewer=band_one_person,
                             shipped=band_one_person,
                             coco_model_alone=coco_best.to_dict()),
        reviewer_fatigue=fatigue,
        alternatives=alternatives(),
    )
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nChosen single threshold {conf:.2f}: EUR {chosen['cost_per_1000']:.0f} / 1,000 frames")
    print(f"Best band [{band['t_low']:.2f}, {band['t_high']:.2f}): EUR {band['cost_per_1000']:.0f} / 1,000, "
          f"review share {band['review_share']:.1%}, {band['reviewers_needed']:.2f} reviewers")
    print(f"Shipped band [{band_one_person['t_low']:.2f}, {band_one_person['t_high']:.2f}): "
          f"EUR {band_one_person['cost_per_1000']:.0f} / 1,000, review share "
          f"{band_one_person['review_share']:.1%}, {band_one_person['reviewers_needed']:.2f} reviewers")
    print(pd.DataFrame(fatigue).round(0).to_string(index=False))
    print(f"Alternatives: {json.dumps(alternatives(), default=float)}")
    for name, d in detection.items():
        print(name, {k: round(v["ap50"], 3) for k, v in d["per_class"].items()}, d["overall"])


if __name__ == "__main__":
    main()
