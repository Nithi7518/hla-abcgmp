"""Local website for the Explainable Plant Disease Assistant.

Run:   python server.py            ->  http://127.0.0.1:8000
The server only listens on localhost; photos are processed in memory and never stored.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.assistant.chat import AssistantSession
from src.assistant.evidence import QUESTIONS
from src.assistant.knowledge import KnowledgeBase
from src.assistant.llm import make_explainer
from src.assistant.pipeline import Diagnoser

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SESSIONS = 200

app = FastAPI(title="Plant Disease Assistant", docs_url=None, redoc_url=None)
_state: dict = {}
_sessions: "OrderedDict[str, AssistantSession]" = OrderedDict()


def pretty(label: str) -> str:
    return label.replace("__", " / ").replace("_", " ")


REPRESENTATIVE_SAMPLES = {
    "apple__black_rot": "dataset_main/Apple___Black_rot/ABR (1592).JPG",
    "apple__disease_fruit": "dataset_main/Disease Apple Fruit/DAF (253).jpeg",
    "apple__healthy_leaf": "dataset_main/Apple___healthy/AH (1136).JPG",
    "banana__cordana": "dataset_main/Banana cordana/cordana93.jpg",
    "banana__disease_fruit": "dataset_main/Disease Banana Fruit/DBF (63).jpeg",
    "banana__healthy_fruit": "dataset_main/Healthy Banana Fruit/HBF (123).jpeg",
    "banana__healthy_leaf": "dataset_main/Healthy Banana Leaf/ft (90).jpg",
    "citrus__canker_leaf": "dataset_main/Citrus CanckerLeaf/CitrusCanker(74).png",
    "citrus__disease_fruit": "dataset_main/Citrus Fruit disease/CitrusCanker(50).jpg",
    "citrus__healthy_fruit": "dataset_main/Citrus Healthy Fruit/images (7).jpeg",
    "citrus__healthy_leaf": "dataset_main/Citrus Healthy Leaf/images (4).jpeg",
    "guava__disease_fruit": "dataset_main/Guava Disease Fruit/GDF (45).jpeg",
    "guava__healthy_fruit": "dataset_main/Guava Healthly Fruit/IMG_20241012_165800.jpg",
    "guava__healthy_leaf": "dataset_main/Guava Healthly Leaf/IMG_20240905_165437.jpg",
    "guava__red_rust": "dataset_main/Guava Red Rust/IMG_20240822_125925.jpg",
    "mango__bacterial_canker": "dataset_main/Mango Bacterial Canker/IMG_20211106_135326 (Custom).jpg",
    "mango__disease_fruit": "dataset_main/Disease Mango Fruit/DMF (144).jpeg",
    "mango__healthy_fruit": "dataset_main/Healthy Mango Fruit/HMF (112).jpeg",
    "mango__healthy_leaf": "dataset_main/Mang Healthy Leaf/20211231_155719 (Custom).jpg",
    "papaya__disease_fruit": "dataset_main/Papaya Fruit Disease/PFD (8).jpeg",
    "papaya__healthy_fruit": "dataset_main/Papaya Fruit Healthy/PFL (118).jpeg",
    "papaya__healthy_leaf": "dataset_main/Papaya Healthy leaf/PHL (291).jpg",
    "papaya__ring_spot": "dataset_main/Papaya RingSpot/PRS (672).jpg",
}


def assets():
    if not _state:
        _state["diagnoser"] = Diagnoser()
        _state["kb"] = KnowledgeBase()
        _state["explainer"] = make_explainer()
        m = pd.read_csv("data/manifests/test.csv")
        sample_meta = {}
        for _, row in m.iterrows():
            organ_val = row.get("organ")
            sample_meta[row["image_path"]] = (
                row.get("crop"),
                organ_val if organ_val in ["leaf", "fruit"] else None,
            )
        _state["sample_meta"] = sample_meta
        _state["samples"] = {f"{lbl}|0": path for lbl, path in REPRESENTATIVE_SAMPLES.items()}
    return _state


def data_url(img: Image.Image, fmt: str = "JPEG", max_side: int = 640) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=88)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def get_session(session_id: str) -> AssistantSession:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Session expired. Please analyze the photo again.")
    return s


def ranking_payload(s: AssistantSession) -> list[dict]:
    return [{"label": n, "name": pretty(n), "prob": p} for n, p in s.ranking]


class AnswersIn(BaseModel):
    session_id: str
    answers: dict[str, str]


class SessionIn(BaseModel):
    session_id: str


class AskIn(BaseModel):
    session_id: str
    question: str


@app.get("/api/info")
def info():
    a = assets()
    return {
        "engine": a["explainer"].name,
        "engine_label": a["explainer"].label,
        "llm": a["explainer"].is_llm,
        "questions": [{"id": q["id"], "text": q["text"],
                       "options": [{"value": v, "label": l} for v, l in q["options"]]} for q in QUESTIONS],
        "samples": [{"id": k, "name": pretty(k.split("|")[0])}
                    for k in a["samples"]],
    }


@app.post("/api/diagnose")
async def diagnose(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    crop: str | None = Form(None),
    organ: str | None = Form(None),
):
    a = assets()
    if file is not None and file.filename:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 10 MB.")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "That file is not a readable image (use JPG or PNG).")
    elif sample and sample in a["samples"]:
        image = Image.open(Path("data/raw") / a["samples"][sample])
    else:
        raise HTTPException(400, "Upload a photo or choose a sample.")

    true_label = sample.split("|")[0] if (file is None or not file.filename) and sample else None
    if sample and sample in a["samples"]:
        sample_path = a["samples"][sample]
        if sample_path in a.get("sample_meta", {}):
            crop, organ = a["sample_meta"][sample_path]
    elif not crop and not organ:
        crop, organ = None, None
    session = AssistantSession(a["diagnoser"], a["kb"], a["explainer"])
    reply = session.start(image.convert("RGB"), crop=crop, organ=organ)
    sid = uuid.uuid4().hex
    _sessions[sid] = session
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    d = session.diagnosis
    return {
        "session_id": sid,
        "tier": d.tier,
        "message": reply.text,
        "notes": d.notes,
        "ranking": ranking_payload(session),
        "original": data_url(image),
        "overlay": data_url(d.overlay, "PNG") if d.overlay is not None else None,
        "can_ask": True,
        "sample": None if true_label is None else {
            "true_label": pretty(true_label), "matches": d.top.label == true_label},
    }


@app.post("/api/answers")
def answers(body: AnswersIn):
    s = get_session(body.session_id)
    if s.diagnosis.tier == "refuse":
        raise HTTPException(400, "No reliable diagnosis to refine; upload a clearer photo.")
    for qid, value in body.answers.items():
        try:
            s.answer(qid, value)
        except KeyError:
            raise HTTPException(400, f"Unknown question: {qid}")
    return {"ranking": ranking_payload(s), "evidence_log": s.evidence_log, "crop_conflict": s.crop_conflict}


@app.post("/api/advice")
def advice(body: SessionIn):
    r = get_session(body.session_id).advice()
    return {"text": r.text, "sources": r.sources, "kb_note": assets()["kb"].note}


@app.post("/api/ask")
def ask(body: AskIn):
    question = body.question.strip()[:500]
    if not question:
        raise HTTPException(400, "Type a question first.")
    r = get_session(body.session_id).ask(question)
    return {"text": r.text, "sources": r.sources}


def _load_authentic_benchmarks() -> dict:
    """Reads authentic evaluation JSON/CSV artifacts directly from results/ and data/manifests/."""
    exp_root = Path("results/experiments")
    run_meta_labels = {
        "2026-10-12T183341Z_multimodal_hlanet_fusion_seed42": {
            "title": "Multimodal HLA-Net + Gated Fusion (Proposed)",
            "family": "Multimodal (Vision + Agronomic Context)",
            "architecture": "HLABackbone (4-Stage DMSLBlock + CLA) + MetadataMLP (12->64) + GatedFusion (128-d)",
            "pretraining": "None (Trained 100% From Scratch, Kaiming Init)",
            "params_total": 1711863,
            "params_trainable": 1711863,
            "modality": "RGB Image (224×224) + 12-D Agronomic Metadata (15% Dropout)",
            "loss": "Class-Balanced Focal Loss (γ=2.0) + Label Smoothing",
            "hardware": "NVIDIA RTX 3050 Laptop GPU (CUDA AMP fp16)",
        },
        "2026-10-12T164550Z_hlanet_balanced_gpu_seed42": {
            "title": "Unimodal HLA-Net (Balanced Focal, 30 Epochs GPU)",
            "family": "Unimodal Custom CNN-Attention (From Scratch)",
            "architecture": "HLABackbone (MSRStem + 4-Stage DMSLBlock + CoordinateLesionAttention)",
            "pretraining": "None (Trained 100% From Scratch, Kaiming Init)",
            "params_total": 1543543,
            "params_trainable": 1543543,
            "modality": "RGB Image Only (224×224)",
            "loss": "Class-Balanced Focal Loss (γ=2.0)",
            "hardware": "NVIDIA RTX 3050 Laptop GPU (CUDA)",
        },
        "2026-10-12T115713Z_hlanet_from_scratch_seed42": {
            "title": "Unimodal HLA-Net (Standard CE, 15 Epochs)",
            "family": "Unimodal Custom CNN-Attention (From Scratch)",
            "architecture": "HLABackbone (MSRStem + 4-Stage DMSLBlock + CoordinateLesionAttention)",
            "pretraining": "None (Trained 100% From Scratch, Kaiming Init)",
            "params_total": 1543543,
            "params_trainable": 1543543,
            "modality": "RGB Image Only (224×224)",
            "loss": "Class-Weighted Cross-Entropy",
            "hardware": "Intel Core i5 CPU",
        },
        "2026-10-12T111950Z_hlanet_from_scratch_seed42": {
            "title": "Unimodal HLA-Net (Intermediate 10-Epoch Run)",
            "family": "Unimodal Custom CNN-Attention (From Scratch)",
            "architecture": "HLABackbone (MSRStem + 4-Stage DMSLBlock + CoordinateLesionAttention)",
            "pretraining": "None (Trained 100% From Scratch)",
            "params_total": 1543543,
            "params_trainable": 1543543,
            "modality": "RGB Image Only (224×224)",
            "loss": "Class-Weighted Cross-Entropy",
            "hardware": "Intel Core i5 CPU",
        },
        "2026-10-12T110157Z_hlanet_from_scratch_seed42": {
            "title": "Unimodal HLA-Net (1-Epoch Smoke Prototype)",
            "family": "Unimodal Custom CNN-Attention (From Scratch)",
            "architecture": "HLABackbone (MSRStem + 4-Stage DMSLBlock + CoordinateLesionAttention)",
            "pretraining": "None (Trained 100% From Scratch)",
            "params_total": 1543543,
            "params_trainable": 1543543,
            "modality": "RGB Image Only (224×224)",
            "loss": "Class-Weighted Cross-Entropy",
            "hardware": "Intel Core i5 CPU",
        },
    }

    runs = []
    if exp_root.exists():
        for run_dir in sorted(exp_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
            rm_file = run_dir / "run_metadata.json"
            rm_data = json.loads(rm_file.read_text(encoding="utf-8")) if rm_file.exists() else {}
            hist_file = run_dir / "training_history.csv"
            history = []
            if hist_file.exists():
                df_h = pd.read_csv(hist_file)
                history = df_h.to_dict(orient="records")

            rid = run_dir.name
            info_meta = run_meta_labels.get(rid, {
                "title": rid,
                "family": "Experiment Run",
                "architecture": rm_data.get("backbone", "hla_net"),
                "pretraining": "None",
                "params_total": 1543543,
                "params_trainable": 1543543,
                "modality": "RGB Image",
                "loss": rm_data.get("class_imbalance_strategy", "weighted_ce"),
                "hardware": rm_data.get("device", "cpu"),
            })

            runs.append({
                "run_id": rid,
                **info_meta,
                "seed": rm_data.get("seed", 42),
                "epochs_run": rm_data.get("epochs_run"),
                "best_epoch": rm_data.get("best_epoch"),
                "total_training_seconds": rm_data.get("total_training_seconds"),
                "n_train": rm_data.get("n_train", 6248),
                "n_val": rm_data.get("n_val", 1347),
                "n_test": m_data.get("n_test_images", 1341),
                "val_metrics": {
                    k: v for k, v in (rm_data.get("best_val_metrics") or {}).items() if k != "per_class"
                },
                "test_metrics": m_data.get("test_metrics", {}),
                "bootstrap_ci": m_data.get("bootstrap_confidence_intervals", {}),
                "history": history,
                "confusion_matrix_url": f"/results/experiments/{rid}/confusion_matrix.png" if (run_dir / "confusion_matrix.png").exists() else None,
                "training_curve_url": f"/results/experiments/{rid}/training_curve.png" if (run_dir / "training_curve.png").exists() else None,
                "has_per_class": True,
            })

    # Include the authentic documented clean-split ResNet-18 transfer learning baselines from README Sections 12 & 15b
    reference_baselines = [
        {
            "run_id": "2026-10-12T071530Z_resnet18_ft2_seed42",
            "title": "ResNet-18 Transfer (layer3 + layer4 Unfrozen, Seed 42)",
            "family": "ImageNet-1K Pretrained CNN Baseline (README §15b)",
            "architecture": "Torchvision ResNet-18 (layer1-2 frozen, layer3-4 + Linear head unfrozen)",
            "pretraining": "ImageNet-1K V1 (Pretrained)",
            "params_total": 11188311,
            "params_trainable": 10500000,
            "modality": "RGB Image Only (224×224)",
            "loss": "Class-Weighted Cross-Entropy (lr=5e-4)",
            "hardware": "Apple M4 MPS (20 Epochs, ~28 min)",
            "seed": 42,
            "epochs_run": 20,
            "best_epoch": 18,
            "total_training_seconds": 1680.0,
            "n_train": 6248,
            "n_val": 1347,
            "n_test": 1341,
            "val_metrics": {"macro_f1": 0.9050},
            "test_metrics": {
                "accuracy": 0.9523,
                "balanced_accuracy": 0.8908,
                "macro_f1": 0.8877,
                "weighted_f1": 0.9520,
            },
            "multi_seed_summary": "3-seed mean (seeds 42, 1, 2): Acc 0.9486 ± 0.0034 | Bal Acc 0.8807 ± 0.0114 | Macro F1 0.8794 ± 0.0085",
            "has_per_class": False,
        },
        {
            "run_id": "2026-10-11T122729Z_resnet18_baseline_seed42",
            "title": "ResNet-18 Transfer (layer4 Only Unfrozen, Seed 42)",
            "family": "ImageNet-1K Pretrained CNN Baseline (README §12)",
            "architecture": "Torchvision ResNet-18 (frozen except layer4 + Linear head)",
            "pretraining": "ImageNet-1K V1 (Pretrained)",
            "params_total": 11188311,
            "params_trainable": 8404495,
            "modality": "RGB Image Only (224×224)",
            "loss": "Class-Weighted Cross-Entropy (lr=1e-3)",
            "hardware": "Apple M4 MPS (20 Epochs, ~25 min)",
            "seed": 42,
            "epochs_run": 20,
            "best_epoch": 17,
            "total_training_seconds": 1500.0,
            "n_train": 6248,
            "n_val": 1347,
            "n_test": 1341,
            "val_metrics": {"macro_f1": 0.8810},
            "test_metrics": {
                "accuracy": 0.9456,
                "balanced_accuracy": 0.8775,
                "macro_f1": 0.8721,
                "weighted_f1": 0.9457,
            },
            "multi_seed_summary": "3-seed mean (seeds 42, 1, 2): Acc 0.9491 ± 0.0042 | Bal Acc 0.8816 ± 0.0036 | Macro F1 0.8800 ± 0.0080",
            "has_per_class": False,
        },
    ]

    # Cross-crop generalization aggregation from results/cross_crop/seed{42,1,2}/*.json
    cc_root = Path("results/cross_crop")
    cross_crop_pairs = []
    per_crop_heldout: dict[str, list[tuple[float, float]]] = {}
    if cc_root.exists():
        for seed_dir in sorted(cc_root.iterdir()):
            if not seed_dir.is_dir():
                continue
            for jf in sorted(seed_dir.glob("heldout_*.json")):
                d = json.loads(jf.read_text(encoding="utf-8"))
                cross_crop_pairs.append({
                    "seed": d.get("seed"),
                    "held_out": d.get("held_out"),
                    "seen_macro_f1": d.get("seen_crops_test", {}).get("macro_f1"),
                    "held_out_macro_f1": d.get("held_out_crops", {}).get("macro_f1"),
                    "held_out_roc_auc": d.get("held_out_crops", {}).get("roc_auc"),
                })
                for cname, cstats in (d.get("held_out_per_crop") or {}).items():
                    per_crop_heldout.setdefault(cname, []).append(
                        (cstats.get("macro_f1", 0.0), cstats.get("roc_auc", 0.0))
                    )

    cross_crop_by_crop = []
    for cname, vals in sorted(per_crop_heldout.items()):
        f1s = [v[0] for v in vals]
        aucs = [v[1] for v in vals]
        cross_crop_by_crop.append({
            "crop": cname,
            "n_evaluations": len(vals),
            "mean_macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "mean_roc_auc": round(sum(aucs) / len(aucs), 4) if aucs else 0.0,
        })

    split_report_path = Path("data/manifests/split_report.json")
    split_report = json.loads(split_report_path.read_text(encoding="utf-8")) if split_report_path.exists() else {}

    cal_path = Path("results/assistant/calibration.json")
    calibration = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}

    return {
        "evaluated_runs": runs,
        "reference_baselines": reference_baselines,
        "split_report": split_report,
        "calibration": calibration,
        "cross_crop_summary": {
            "total_runs": len(cross_crop_pairs),
            "by_crop": cross_crop_by_crop,
        },
        "gradcam_grid_url": "/results/gradcam/gradcam_all_classes.png" if Path("results/gradcam/gradcam_all_classes.png").exists() else None,
    }


@app.get("/api/benchmarks")
def benchmarks():
    if "benchmarks" not in _state:
        _state["benchmarks"] = _load_authentic_benchmarks()
    return _state["benchmarks"]


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
if Path("results").exists():
    app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/")
def index_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/diagnose")
def diagnose_page():
    return FileResponse(WEB_DIR / "diagnose.html")


@app.get("/compare")
def compare_page():
    return FileResponse(WEB_DIR / "compare.html")


@app.get("/crops")
def crops_page():
    return FileResponse(WEB_DIR / "crops.html")


if __name__ == "__main__":
    assets()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

