"""
Assignment 2: hard-hat compliance dashboard.

    uv run python main.py app

Loads only what train.py produced: weights/best.pt and artifacts/. Nothing is
trained here. Every number on the operating-point tab is recomputed from the
per-frame test scores train.py saved, using the same costs.py the pipeline
used to pick the thresholds.
"""

import json
from dataclasses import replace
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from ultralytics import YOLO

from costs import DEFAULT_COSTS, alternatives, evaluate_band, single_threshold_sweep
from evaluation import detection_confusion

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
METRICS = json.loads((ARTIFACTS / "metrics.json").read_text())
STATS = json.loads((ARTIFACTS / "dataset_stats.json").read_text())
FRAMES = pd.read_csv(ARTIFACTS / "test_frames.csv")
PREDS = {k: [tuple(d) for d in v]
         for k, v in json.loads((ARTIFACTS / "test_predictions.json").read_text()).items()}
GT = {k: [tuple(b) for b in v]
      for k, v in json.loads((ARTIFACTS / "test_ground_truth.json").read_text()).items()}

NAMES = {int(k): v for k, v in METRICS["names"].items()}
POSITIVE = [k for k, v in NAMES.items() if v == METRICS["positive_class"]][0]
BAND = METRICS["operating_point"]["shipped"]
CHOSEN = METRICS["operating_point"]["model_alone"]["t_low"]
SCORES, LABELS = FRAMES["score"].to_numpy(), FRAMES["label"].to_numpy()
COLOURS = {0: "#d62728", 1: "#2ca02c"}
EUR = "€"


def decide(score: float, t_low: float, t_high: float) -> str:
    if score >= t_high:
        return "FAIL: bare head, radio the supervisor"
    if score >= t_low:
        return "REVIEW: send to a person"
    return "PASS: everyone in view wears a helmet"


def inspect(image, t_low, t_high):
    if image is None:
        return None, "Upload a frame.", pd.DataFrame()
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    # Gradio gives RGB; Ultralytics reads a numpy array as BGR but a PIL image
    # as RGB, so pass PIL or red helmets reach the model as blue ones.
    canvas = Image.fromarray(image).convert("RGB")
    result = MODEL.predict(canvas.copy(), conf=0.01, imgsz=416, verbose=False)[0]
    draw = ImageDraw.Draw(canvas)
    rows = []
    for c, s, (x1, y1, x2, y2) in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                     result.boxes.xyxy.tolist()):
        c = int(c)
        # Every head at or above t_low drives the decision, so all are shown;
        # faint helmet boxes change nothing and are hidden to keep it readable.
        if s < (t_low if c == POSITIVE else max(t_low, 0.25)):
            continue
        draw.rectangle([x1, y1, x2, y2], outline=COLOURS[c], width=3)
        draw.text((x1 + 2, y1 + 2), f"{NAMES[c]} {s:.2f}", fill=COLOURS[c])
        rows.append(dict(cls=NAMES[c], confidence=round(s, 3)))
    score = max((r["confidence"] for r in rows if r["cls"] == NAMES[POSITIVE]), default=0.0)
    verdict = f"**{decide(score, t_low, t_high)}**  \nhighest `head` confidence: {score:.3f}"
    return canvas, verdict, pd.DataFrame(rows)


def cost_model(base_rate, fn, fp, review):
    return replace(DEFAULT_COSTS, base_rate=base_rate, fn=fn, fp=fp, tp=fp, review=review)


def cost_curve(c, t_low, t_high):
    sweep = pd.DataFrame(single_threshold_sweep(SCORES, LABELS, c))
    best = sweep.loc[sweep["cost_per_1000"].idxmin()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["t_low"], y=sweep["cost_per_1000"], mode="lines",
                             name="model decides alone", line=dict(color="#1f4e79")))
    fig.add_trace(go.Scatter(x=[CHOSEN], y=[sweep.set_index("t_low").loc[round(CHOSEN, 2), "cost_per_1000"]],
                             mode="markers", marker=dict(size=12, color="#c0392b", symbol="x"),
                             name=f"chosen in pipeline (t = {CHOSEN:.2f})"))
    fig.add_trace(go.Scatter(x=[best["t_low"]], y=[best["cost_per_1000"]], mode="markers",
                             marker=dict(size=10, color="#f39c12"),
                             name=f"minimum at these costs (t = {best['t_low']:.2f})"))
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor="#f1c40f", opacity=0.2, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(xaxis_title="confidence threshold for `head`",
                      yaxis_title=f"expected cost per 1,000 frames ({EUR})",
                      height=380, margin=dict(l=40, r=20, t=30, b=40),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def operating_point(t_low, t_high, base_rate, fn, fp, review):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    c = cost_model(base_rate, fn, fp, review)
    row = evaluate_band(SCORES, LABELS, t_low, t_high, c)

    outcome = np.where(SCORES >= t_high, "fail", np.where(SCORES >= t_low, "review", "pass"))
    matrix = pd.DataFrame(
        {o: [int(np.sum((outcome == o) & (LABELS == 1))), int(np.sum((outcome == o) & (LABELS == 0)))]
         for o in ["pass", "review", "fail"]},
        index=["bare head in frame (positive)", "all helmeted (negative)"]).reset_index(names="true \\ outcome")

    alt = alternatives(c)
    model_alone = min(single_threshold_sweep(SCORES, LABELS, c), key=lambda r: r["cost_per_1000"])
    compare = pd.DataFrame([
        dict(policy="Do nothing (no monitoring)", cost_per_1000=alt["do_nothing"], review_share=0.0,
             reviewers=0.0),
        dict(policy="Person checks every frame", cost_per_1000=alt["human_checks_everything"],
             review_share=1.0, reviewers=alt["human_checks_everything_staff"]["reviewers_needed"]),
        dict(policy=f"Model alone (t = {model_alone['t_low']:.2f})",
             cost_per_1000=model_alone["cost_per_1000"], review_share=0.0, reviewers=0.0),
        dict(policy=f"This band [{t_low:.2f}, {t_high:.2f})", cost_per_1000=row["cost_per_1000"],
             review_share=row["review_share"], reviewers=row["reviewers_needed"]),
    ]).round(3)

    summary = (
        f"### {EUR}{row['cost_per_1000']:,.0f} per 1,000 frames\n"
        f"- Sent for review: **{row['review_share']:.1%}** of production frames "
        f"(~{row['reviews_per_day']:,.0f}/day of {c.frames_per_day:,}, "
        f"{row['review_hours_per_day']:.1f} h, **{row['reviewers_needed']:.2f} reviewers**)\n"
        f"- Bare-head frames missed outright (pass): **{row['pos_pass']:.1%}**; "
        f"helmeted frames wrongly alerted (fail): **{row['neg_fail']:.1%}**\n"
        f"- Test split counts below are raw; costs reweight them to a {base_rate:.0%} base rate."
    )
    m = detection_confusion(GT, PREDS, len(NAMES), t_low)
    labels = [*NAMES.values(), "background"]
    det = pd.DataFrame(m, index=[f"pred {l}" for l in labels],
                       columns=[f"true {l}" for l in labels]).reset_index(names="")
    return summary, matrix, det, compare, cost_curve(c, t_low, t_high)


def per_class_tables():
    rows = []
    for model, d in METRICS["detection"].items():
        for cls, v in d["per_class"].items():
            rows.append(dict(model=model, cls=cls, instances=v["instances"], AP50=v["ap50"],
                             AP50_95=v["ap50_95"], precision=v["precision"], recall=v["recall"]))
        rows.append(dict(model=model, cls="ALL (mean)", instances=None, AP50=d["overall"]["ap50"],
                         AP50_95=d["overall"]["ap50_95"], precision=None, recall=None))
    detection = pd.DataFrame(rows).round(3)
    frame = pd.DataFrame([dict(model=k, **{m: v[m] for m in ["tp", "fp", "fn", "tn", "precision",
                                                              "recall", "accuracy"]})
                          for k, v in METRICS["frame"].items()]).round(3)
    data = pd.DataFrame([dict(split=s, images=v["images"],
                              frames_with_bare_head=round(v["frames_with_bare_head"], 3),
                              **{f"{c} instances": p["instances"] for c, p in v["per_class"].items()},
                              **{f"{c} median size": round(p["median_rel_side"], 3)
                                 for c, p in v["per_class"].items()})
                         for s, v in STATS.items()])
    return detection, frame, data


with gr.Blocks(title="Hard-hat compliance") as demo:
    gr.Markdown(
        "# Hard-hat compliance: pass, review or fail\n"
        f"Finetuned YOLO11n on `{METRICS['dataset']['workspace']}/{METRICS['dataset']['project']}` "
        f"v{METRICS['dataset']['version']}. Positive class: **`head`** (a bare head). "
        f"Frame score = highest `head` confidence. Shipped band (cheapest one reviewer can handle): "
        f"pass < {BAND['t_low']:.2f} <= review < {BAND['t_high']:.2f} <= fail.")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=BAND["t_low"], step=0.01, label="t_low: below this, PASS")
        t_high = gr.Slider(0, 1, value=BAND["t_high"], step=0.01, label="t_high: at or above this, FAIL")

    with gr.Tab("Inspect a frame"):
        with gr.Row():
            upload = gr.Image(type="numpy", label="CCTV frame")
            annotated = gr.Image(type="pil", label="Detections at or above t_low")
        verdict = gr.Markdown()
        boxes = gr.Dataframe(label="Detections")

    with gr.Tab("Operating point and cost"):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.3, value=DEFAULT_COSTS.base_rate, step=0.005,
                                  label="Base rate: share of frames with a bare head")
            fn_cost = gr.Number(value=DEFAULT_COSTS.fn, label=f"Missed bare head ({EUR})")
            fp_cost = gr.Number(value=DEFAULT_COSTS.fp, label=f"Supervisor intervention ({EUR})")
            review_cost = gr.Number(value=DEFAULT_COSTS.review, label=f"One human review ({EUR})")
        summary = gr.Markdown()
        with gr.Row():
            frame_matrix = gr.Dataframe(label="Frame-level outcomes on the test split (counts)")
            det_matrix = gr.Dataframe(label="Box-level confusion matrix at t_low (IoU 0.5)")
        compare = gr.Dataframe(label="Against the alternatives, same costs and base rate")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Per-class results"):
        det_table, frame_table, data_table = per_class_tables()
        gr.Markdown("Box-level metrics on the test split, per class. Precision/recall at the "
                    f"pipeline-chosen threshold ({CHOSEN:.2f}). COCO knows no helmets, so its "
                    "`helmet` row is zero by construction.")
        gr.Dataframe(det_table)
        gr.Markdown("Frame-level decision (bare head present?) for each model and the trivial baseline.")
        gr.Dataframe(frame_table)
        gr.Markdown("Dataset composition (median size = box side relative to image side).")
        gr.Dataframe(data_table)
        gr.Markdown(f"Ultralytics cross-check on the test split: "
                    f"mAP50 {METRICS['ultralytics_test_check']['map50']:.3f}, "
                    f"mAP50-95 {METRICS['ultralytics_test_check']['map50_95']:.3f}.")

    op_inputs = [t_low, t_high, base_rate, fn_cost, fp_cost, review_cost]
    op_outputs = [summary, frame_matrix, det_matrix, compare, curve]
    for control in op_inputs:
        control.change(operating_point, op_inputs, op_outputs)
    for control in [upload, t_low, t_high]:
        control.change(inspect, [upload, t_low, t_high], [annotated, verdict, boxes])
    demo.load(operating_point, op_inputs, op_outputs)


if __name__ == "__main__":
    demo.launch()
