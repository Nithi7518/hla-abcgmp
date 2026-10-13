"""End-to-end evaluation of the assistant (no LLM involved).

Usage:  python -m src.assistant.evaluate_assistant

1. Retrieval: do intent-style questions return the right passage topic, and are off-topic
   questions declined?
2. Over the whole test set: how many photos get a diagnosis tier of each kind, how often
   disease-specific cited advice is produced, and how often that advice is for the TRUE
   disease (advice precision) -- the number that matters for harm.
Output: results/assistant/assistant_eval.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from src.assistant.chat import AssistantSession, label_kind
from src.assistant.knowledge import KnowledgeBase
from src.assistant.pipeline import Diagnoser

INTENT_QUESTIONS = {
    "symptoms": ["what are the symptoms", "how can I recognise it", "what does it look like"],
    "spread": ["how does it spread", "what causes it to spread", "does rain make it worse"],
    "management": ["how do I treat this", "what should I do about it", "how can I prevent it"],
}
OFF_TOPIC = ["what is the price of tomatoes in Delhi", "tell me a joke", "who won the cricket match",
             "can you write me a poem", "what is the capital of France", "how do I fix my laptop"]


def retrieval_eval(kb: KnowledgeBase) -> dict:
    labels = sorted({l for p in kb.passages for l in p.labels})
    hits = total = 0
    for label in labels:
        for topic, qs in INTENT_QUESTIONS.items():
            for q in qs:
                total += 1
                r = kb.search(q, label=label)
                hits += bool(r) and r[0].topic == topic
    declined = sum(not kb.search(q, label=labels[0]) for q in OFF_TOPIC)
    return {"intent_questions": total, "top1_topic_correct": hits / total,
            "off_topic_questions": len(OFF_TOPIC), "off_topic_declined": declined / len(OFF_TOPIC)}


def test_set_eval(diagnoser: Diagnoser, kb: KnowledgeBase) -> dict:
    df = pd.read_csv("data/manifests/test.csv")
    tiers, kinds = Counter(), Counter()
    advised = advised_correct = cited = 0
    for r in df.itertuples():
        s = AssistantSession(diagnoser, kb)
        s.start(Image.open(Path("data/raw") / r.image_path))
        tiers[s.diagnosis.tier] += 1
        if s.diagnosis.tier == "refuse":
            continue
        label = s.ranking[0][0]
        kind = label_kind(label, kb)
        kinds[kind] += 1
        if kind == "specific":
            advised += 1
            advised_correct += label == r.label
            cited += bool(s.advice().sources)
    n = len(df)
    return {
        "n_test_images": n,
        "tier_share": {k: v / n for k, v in tiers.items()},
        "answer_kind_share_of_non_refused": {k: v / sum(kinds.values()) for k, v in kinds.items()},
        "disease_specific_advice_given": advised,
        "advice_precision_true_disease": advised_correct / advised if advised else None,
        "advice_with_citations": cited / advised if advised else None,
    }


def main() -> None:
    kb, diagnoser = KnowledgeBase(), Diagnoser()
    result = {"retrieval": retrieval_eval(kb), "test_set": test_set_eval(diagnoser, kb)}
    Path("results/assistant/assistant_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
