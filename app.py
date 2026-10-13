"""Explainable Diagnosis Assistant (Streamlit).  Run:  streamlit run app.py"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.assistant.chat import AssistantSession
from src.assistant.evidence import QUESTIONS
from src.assistant.knowledge import KnowledgeBase
from src.assistant.llm import DISCLAIMER, make_explainer
from src.assistant.pipeline import Diagnoser

st.set_page_config(page_title="Plant Disease Assistant", page_icon="🌿", layout="wide")

TIER_BADGE = {"confident": ("🟢", "Confident"), "uncertain": ("🟡", "Uncertain"), "refuse": ("🔴", "Cannot diagnose")}


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    return Diagnoser(), KnowledgeBase(), make_explainer()


@st.cache_data
def sample_images() -> dict[str, str]:
    m = pd.read_csv("data/manifests/test.csv")
    pick = m.groupby("label").head(1).sort_values("label")
    return {r.label.replace("__", " / ").replace("_", " "): r.image_path for r in pick.itertuples()}


def pretty(label: str) -> str:
    return label.replace("__", " / ").replace("_", " ")


diagnoser, kb, explainer = load_assets()

with st.sidebar:
    st.header("About")
    st.write("Upload a leaf or fruit photo (apple, banana, citrus, guava, mango, papaya). The assistant "
             "shows what the image model sees, how sure it is, asks a few questions, and gives "
             "**sourced** advice — or says it can't.")
    st.caption(f"Text engine: **{explainer.name}**" + ("" if explainer.name == "claude" else " (no API key; rule-based)"))
    st.warning(DISCLAIMER)
    st.caption("Limits: image model only (23 classes), validated on dataset photos; weak on some fruit "
               "classes and on unseen crops; studio/stock photos can be misread. Answers you give only "
               "nudge the ranking with fixed, unvalidated rules.")

st.title("🌿 Explainable Plant Disease Assistant")

col_up, col_sample = st.columns(2)
upload = col_up.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
samples = sample_images()
sample = col_sample.selectbox("...or try a dataset sample", ["—"] + list(samples))

image, key = None, None
if upload is not None:
    data = upload.getvalue()
    image, key = Image.open(upload).convert("RGB"), hashlib.md5(data).hexdigest()
elif sample != "—":
    path = Path("data/raw") / samples[sample]
    image, key = Image.open(path).convert("RGB"), f"sample:{samples[sample]}"

if image is None:
    st.info("Add a photo to begin.")
    st.stop()

if st.session_state.get("key") != key:
    session = AssistantSession(diagnoser, kb, explainer)
    first = session.start(image)
    st.session_state.update(key=key, session=session, first=first, chat=[], advice=None)
session: AssistantSession = st.session_state.session
d = session.diagnosis

left, right = st.columns(2)
left.image(image, caption="Your photo", width="stretch")
if d.overlay is not None:
    right.image(d.overlay, caption="Where the model looked (Grad-CAM)", width="stretch")
else:
    right.info("No heatmap: the model's confidence is too low to name a diagnosis.")

icon, name = TIER_BADGE[d.tier]
st.subheader(f"{icon} {name}")
st.markdown(st.session_state.first.text)
for note_ in d.notes:
    st.warning(note_)
for name_, prob_ in session.ranking:
    st.progress(min(max(prob_, 0.0), 1.0), text=f"{pretty(name_)} — {100 * prob_:.0f}%")

if session.pending_questions() or session.answers:
    st.divider()
    st.subheader("Tell me more (optional)")
    st.caption("Your answers re-rank the possibilities using fixed rules taken from the cited sources. "
               "They are user-reported and not validated.")
    with st.form("questions"):
        picked = {}
        for q in QUESTIONS:
            opts = {label: value for value, label in q["options"]}
            current = session.answers.get(q["id"])
            idx = list(opts.values()).index(current) if current in opts.values() else None
            choice = st.radio(q["text"], list(opts), index=idx, horizontal=True, key=f"q_{q['id']}_{key}")
            if choice is not None:
                picked[q["id"]] = opts[choice]
        if st.form_submit_button("Update diagnosis") and d.tier != "refuse":
            for qid, value in picked.items():
                session.answer(qid, value)
            st.rerun()
    if session.evidence_log:
        with st.expander("How your answers changed the ranking", expanded=True):
            for e in session.evidence_log:
                st.write(f"- **{e['question']}** → {e['answer']}: {e['effect']}")

st.divider()
st.subheader("Advice")
if st.button("Show sourced advice for the top diagnosis"):
    st.session_state.advice = session.advice()
if st.session_state.advice:
    adv = st.session_state.advice
    st.markdown(adv.text)
    for i, s in enumerate(adv.sources, 1):
        st.markdown(f"[{i}] [{s['source_title']}]({s['source_url']}) — {s['publisher_type']}")
    st.caption(kb.note)

st.subheader("Ask a follow-up")
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)
if prompt := st.chat_input("e.g. How does it spread? What should I do about it?"):
    reply = session.ask(prompt)
    refs = "".join(f"\n\n[{i}] [{s['source_title']}]({s['source_url']})" for i, s in enumerate(reply.sources, 1))
    st.session_state.chat += [("user", prompt), ("assistant", reply.text + refs)]
    st.rerun()
