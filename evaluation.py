"""
Evaluation shared by train.py and app.py.

One implementation scores every model on the test split, so the finetuned
model, the COCO zero-shot model and the predict-nothing baseline are compared
by exactly the same rules. Boxes are (x1, y1, x2, y2) normalised to [0, 1];
IoU does not change under per-axis scaling, so pixel sizes are not needed.

    gt[image]    = [(cls, x1, y1, x2, y2), ...]
    preds[image] = [(cls, conf, x1, y1, x2, y2), ...]
"""

from pathlib import Path

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def read_yolo_labels(label_dir: Path) -> dict[str, list[tuple]]:
    gt = {}
    for path in sorted(Path(label_dir).glob("*.txt")):
        boxes = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            c, xc, yc, w, h = line.split()[:5]
            xc, yc, w, h = map(float, (xc, yc, w, h))
            boxes.append((int(c), xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        gt[path.stem] = boxes
    return gt


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def _match_class(gt, preds, cls, iou_t, conf_min=0.0):
    """Greedy COCO-style matching for one class: returns (tp flags, confs, n_gt)."""
    records, n_gt = [], 0
    for image, boxes in gt.items():
        g = np.array([b[1:] for b in boxes if b[0] == cls]).reshape(-1, 4)
        n_gt += len(g)
        p = sorted((d for d in preds.get(image, []) if d[0] == cls and d[1] >= conf_min),
                   key=lambda d: -d[1])
        if not p:
            continue
        pb = np.array([d[2:] for d in p]).reshape(-1, 4)
        ious = iou_matrix(pb, g)
        taken = np.zeros(len(g), dtype=bool)
        for i, d in enumerate(p):
            hit = False
            if len(g):
                cand = np.where(~taken, ious[i], -1)
                j = int(np.argmax(cand))
                if cand[j] >= iou_t:
                    taken[j], hit = True, True
            records.append((d[1], hit))
    records.sort(key=lambda r: -r[0])
    confs = np.array([r[0] for r in records])
    tp = np.array([r[1] for r in records], dtype=bool)
    return tp, confs, n_gt


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP, as COCO and Ultralytics compute it."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def detection_metrics(gt, preds, names: dict[int, str], conf: float) -> dict:
    """Per-class AP50, AP50-95, and precision/recall at a confidence threshold."""
    per_class = {}
    for cls, name in names.items():
        aps = []
        for t in IOU_THRESHOLDS:
            tp, _, n_gt = _match_class(gt, preds, cls, t)
            aps.append(average_precision(tp, n_gt))
        tp, confs, n_gt = _match_class(gt, preds, cls, 0.5)
        keep = confs >= conf
        n_tp = int(tp[keep].sum())
        n_pred = int(keep.sum())
        per_class[name] = dict(
            instances=n_gt, ap50=aps[0], ap50_95=float(np.mean(aps)),
            precision=n_tp / n_pred if n_pred else 0.0,
            recall=n_tp / n_gt if n_gt else 0.0,
            tp=n_tp, fp=n_pred - n_tp, fn=n_gt - n_tp,
        )
    overall = dict(
        ap50=float(np.mean([v["ap50"] for v in per_class.values()])),
        ap50_95=float(np.mean([v["ap50_95"] for v in per_class.values()])),
    )
    return dict(per_class=per_class, overall=overall, conf=conf)


def detection_confusion(gt, preds, n_classes: int, conf: float, iou_t: float = 0.5) -> np.ndarray:
    """Rows = predicted (classes..., background), cols = true (classes..., background).

    Class-agnostic matching, so a helmet predicted as a head shows up as a
    confusion rather than as a miss plus a false positive. Same layout as the
    Ultralytics confusion matrix.
    """
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    bg = n_classes
    for image, boxes in gt.items():
        g = np.array([b[1:] for b in boxes]).reshape(-1, 4)
        gc = [b[0] for b in boxes]
        p = sorted((d for d in preds.get(image, []) if d[1] >= conf), key=lambda d: -d[1])
        pb = np.array([d[2:] for d in p]).reshape(-1, 4)
        ious = iou_matrix(pb, g)
        taken = np.zeros(len(g), dtype=bool)
        for i, d in enumerate(p):
            if len(g):
                cand = np.where(~taken, ious[i], -1)
                j = int(np.argmax(cand))
                if cand[j] >= iou_t:
                    taken[j] = True
                    m[d[0], gc[j]] += 1
                    continue
            m[d[0], bg] += 1
        for j in np.where(~taken)[0]:
            m[bg, gc[j]] += 1
    return m


def frame_scores(gt, preds, positive_cls: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Frame-level decision data: (image ids, max positive-class conf, label)."""
    images = sorted(gt)
    scores = np.array([max((d[1] for d in preds.get(i, []) if d[0] == positive_cls), default=0.0)
                       for i in images])
    labels = np.array([int(any(b[0] == positive_cls for b in gt[i])) for i in images])
    return images, scores, labels


def frame_confusion(scores, labels, threshold: float) -> dict:
    flag = scores >= threshold
    tp = int(np.sum(flag & (labels == 1)))
    fp = int(np.sum(flag & (labels == 0)))
    fn = int(np.sum(~flag & (labels == 1)))
    tn = int(np.sum(~flag & (labels == 0)))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                accuracy=(tp + tn) / len(labels))
