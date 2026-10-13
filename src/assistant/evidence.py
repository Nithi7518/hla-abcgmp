"""Conversational metadata (Phase C): fixed questions + transparent evidence reweighting.

IMPORTANT: this is NOT the Phase 9 fusion model and is NOT validated. Answers are
user-reported, and they only nudge the image model's calibrated probabilities with fixed,
documented likelihood ratios taken from the symptom descriptions in the knowledge base.
Every adjustment is returned as a log entry the user can see.
"""

from __future__ import annotations

import numpy as np

SUPPORT_RATIO = 3.0   # a symptom description matching a candidate's documented symptoms
WET_RATIO = 1.3       # wet/humid weather, for diseases the sources say it favours

CROPS = ["apple", "banana", "citrus", "guava", "mango", "papaya"]

# value -> (labels supported, user-facing description, knowledge-base passage id)
LESION_RULES = {
    "frogeye_rings": (["apple__black_rot", "apple__disease_fruit"],
                      "purple-edged tan 'frogeye' leaf spots, or brown/black rings on fruit", "apple_br_sym"),
    "angular_black": (["mango__bacterial_canker"],
                      "black, angular, greasy-looking spots between the veins, often with yellow halos", "mango_bc_sym"),
    "pustule_halo": (["citrus__canker_leaf", "citrus__disease_fruit"],
                     "raised corky pustules with a yellow halo", "citrus_canker_sym"),
    "orange_tufts": (["guava__red_rust"], "round, furry, orange or rust-coloured spots on the leaf", "guava_rr_sym"),
    "diamond_yellow": (["banana__cordana"], "large oval/diamond brown spots with a yellow margin", "banana_cordana_sym"),
    "mottle_shoestring": (["papaya__ring_spot"],
                          "yellow-green mottled leaves, small distorted leaves, ring spots on fruit", "papaya_prsv_sym"),
    "none": ([], "none of these / not sure", None),
}
WET_FAVOURED = {"banana__cordana": "banana_cordana_spread",
                "citrus__canker_leaf": "citrus_canker_spread", "guava__red_rust": "guava_rr_spread",
                "mango__bacterial_canker": "mango_bc_spread"}

QUESTIONS = [
    {"id": "crop", "text": "Which crop is this?",
     "options": [(c, c.capitalize()) for c in CROPS] + [("unsure", "Not sure")]},
    {"id": "lesion", "text": "Which best describes the damage you see?",
     "options": [(k, v[1]) for k, v in LESION_RULES.items()]},
    {"id": "weather", "text": "Has the weather been wet or humid recently?",
     "options": [("wet", "Yes, wet/humid"), ("dry", "No, mostly dry"), ("unsure", "Not sure")]},
]


def apply_evidence(probs: np.ndarray, labels: list[str], answers: dict[str, str]) -> tuple[np.ndarray, list[dict]]:
    """Returns (adjusted probabilities summing to 1, log of every adjustment made)."""
    p = probs.astype(np.float64).copy()
    log: list[dict] = []

    crop = answers.get("crop")
    if crop and crop != "unsure":
        mask = np.array([lab.split("__")[0] == crop for lab in labels])
        if p[mask].sum() > 0:
            p = np.where(mask, p, 0.0)
            log.append({"question": "crop", "answer": crop, "passage": None,
                        "effect": f"Restricted the diagnosis to {crop} classes (you stated the crop)."})

    lesion = answers.get("lesion")
    if lesion and lesion != "none":
        supported, desc, passage = LESION_RULES[lesion]
        hit = [i for i, lab in enumerate(labels) if lab in supported and p[i] > 0]
        for i in hit:
            p[i] *= SUPPORT_RATIO
        names = ", ".join(labels[i] for i in hit) or "no remaining candidate"
        log.append({"question": "lesion", "answer": desc, "passage": passage,
                    "effect": f"Boosted {names} x{SUPPORT_RATIO:g}: matches the documented symptoms."})

    if answers.get("weather") == "wet":
        hit = [i for i, lab in enumerate(labels) if lab in WET_FAVOURED and p[i] > 0]
        for i in hit:
            p[i] *= WET_RATIO
        if hit:
            log.append({"question": "weather", "answer": "wet/humid", "passage": None,
                        "effect": f"Slightly boosted (x{WET_RATIO:g}) diseases that the cited sources say wet conditions favour."})

    total = p.sum()
    return (p / total if total > 0 else probs), log
