"""Chat orchestrator: one AssistantSession per uploaded photo.

Flow: diagnose -> (optional) metadata questions -> evidence-adjusted ranking -> sourced advice.
Guardrails: refuse tier gives no diagnosis or advice; advice only comes from retrieved
knowledge-base passages with citations; no passages -> explicit "I don't have a sourced answer".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image

from src.assistant.evidence import QUESTIONS, apply_evidence
from src.assistant.knowledge import KnowledgeBase, Passage, citations, intent_topics
from src.assistant.llm import DISCLAIMER, NO_ANSWER_TOKEN, SYSTEM_PROMPT, TemplateExplainer
from src.assistant.pipeline import Diagnoser, Diagnosis


@dataclass
class Reply:
    text: str
    sources: list[dict] = field(default_factory=list)


# Disease names a user may type; lets them ask about a disease that is not in their photo.
DISEASE_PATTERNS = [
    ("mango__bacterial_canker", ("bacterial canker", "bacterial black spot", "mango canker")),
    ("citrus__canker_leaf", ("citrus canker", "canker")),
    ("apple__black_rot", ("black rot", "frogeye", "frog-eye", "frog eye")),
    ("banana__cordana", ("cordana", "diamond leaf spot")),
    ("guava__red_rust", ("red rust", "algal leaf spot")),
    ("papaya__ring_spot", ("ringspot", "ring spot", "ring-spot", "prsv")),
]
DISEASE_NAMES = {"mango__bacterial_canker": "mango bacterial canker", "citrus__canker_leaf": "citrus canker",
                 "apple__black_rot": "apple black rot", "banana__cordana": "banana Cordana leaf spot",
                 "guava__red_rust": "guava red rust", "papaya__ring_spot": "papaya ringspot virus"}


def disease_in_text(text: str) -> str | None:
    t = text.lower()
    for label, words in DISEASE_PATTERNS:
        if any(w in t for w in words):
            return label
    return None


PHOTO_WORDS = ("why", "reason", "reasons", "reaso", "sure", "confident", "confidence", "certain", "heatmap",
               "explain", "how did", "result", "mean", "means", "healthy", "wrong", "correct", "accurate", "trust",
               "crop", "what is this", "what is it", "what plant", "which plant", "what fruit", "what leaf",
               "fruit or leaf", "leaf or fruit", "organ", "identify", "what am i", "uncertain", "lowered")


IDENTIFY_RE = re.compile(r"\b(what|which)\b.*\b(disease|problem|issue|illness|infection)\b|what is (this|it)\b|\bidentify\b",
                         re.I)
DOSE_RE = re.compile(r"\b(dose|dosage|how much|per lit(re|er)|litre|liter|gram|grams|ml|millilit|concentration|"
                     r"rate|how often|how many times|ppm|schedule)\b", re.I)


def asks_about_photo(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in PHOTO_WORDS)


def label_kind(label: str, kb: KnowledgeBase) -> str:
    cls = label.split("__", 1)[1]
    if cls.startswith("healthy"):
        return "healthy"
    return "specific" if kb.covers(label) else "generic"


class AssistantSession:
    def __init__(self, diagnoser: Diagnoser, kb: KnowledgeBase, explainer: TemplateExplainer | None = None):
        self.diagnoser, self.kb = diagnoser, kb
        self.explainer = explainer or TemplateExplainer()
        self.diagnosis: Diagnosis | None = None
        self.answers: dict[str, str] = {}
        self.evidence_log: list[dict] = []
        self.ranking: list[tuple[str, float]] = []
        self.history: list[tuple[str, str]] = []
        self.last_llm_status = "not used"

    # --- step 1: photo ------------------------------------------------------------
    def start(self, image: Image.Image, crop: str | None = None, organ: str | None = None) -> Reply:
        self.diagnosis = self.diagnoser.diagnose(image, crop=crop, organ=organ)
        self.answers, self.evidence_log, self.history = {}, [], []
        if crop:
            self.answers["crop"] = crop
        if organ:
            self.answers["organ"] = organ
        self._rerank()
        # notes (e.g. the out-of-scope warning) are shown separately by the UI via diagnosis.notes
        return Reply(self.explainer.diagnosis_text(
            {"tier": self.diagnosis.tier, "ranking": self.ranking[:3], "ood": self.diagnosis.ood_flag}))

    # --- step 2: metadata questions -----------------------------------------------
    def pending_questions(self) -> list[dict]:
        if self.diagnosis is None or self.diagnosis.tier == "refuse":
            return []
        return [q for q in QUESTIONS if q["id"] not in self.answers]

    def answer(self, question_id: str, value: str) -> None:
        if question_id not in {q["id"] for q in QUESTIONS}:
            raise KeyError(question_id)
        self.answers[question_id] = value
        self._rerank()

    def _rerank(self) -> None:
        d = self.diagnosis
        probs, self.evidence_log = apply_evidence(d.probs, self.diagnoser.labels, self.answers)
        order = probs.argsort()[::-1][:3]
        self.ranking = [(self.diagnoser.labels[i], float(probs[i])) for i in order]

    @property
    def crop_conflict(self) -> bool:
        stated = self.answers.get("crop")
        return bool(stated and stated != "unsure" and stated != self.diagnosis.top.crop)

    # --- step 3: advice -----------------------------------------------------------
    def advice(self, topics: tuple[str, ...] = ("symptoms", "management")) -> Reply:
        if self.diagnosis is None or self.diagnosis.tier == "refuse":
            return Reply("I won't give treatment advice without a reliable diagnosis. " + DISCLAIMER)
        label = self.ranking[0][0]
        kind = label_kind(label, self.kb)
        passages = self.kb.for_label(label, topics) if kind == "specific" else []
        return self._compose(label, kind, passages)

    def ask(self, question: str) -> Reply:
        """Free-form follow-up. The page footer already carries the standing disclaimer, so chat replies
        do not repeat it."""
        r = self._ask(question)
        return Reply(r.text.replace("\n\n" + DISCLAIMER, "").strip(), r.sources)

    def _ask(self, question: str) -> Reply:
        """Answer a follow-up. Answers come only from knowledge-base passages; with an LLM engine the
        model phrases the answer from those passages and every answer is validated (see _llm_answer)."""
        mentioned = disease_in_text(question)
        # A question that names a disease is a knowledge question, not a question about the photo, so it
        # can be answered even when the photo itself could not be diagnosed reliably.
        if self.diagnosis is None or (self.diagnosis.tier == "refuse" and not mentioned):
            return Reply("Please upload a clearer photo first; I can't answer questions without a reliable diagnosis.")
        label = mentioned or self.ranking[0][0]
        kind = label_kind(label, self.kb)
        if not mentioned and len(question.split()) <= 2 and not asks_about_photo(question) \
                and not any(w in question.lower() for w in ("symptom", "spread", "treat", "prevent", "control")):
            reply = Reply(self._help_text())
        elif not mentioned and IDENTIFY_RE.search(question) and kind == "specific":
            reply = self._identify(label)
        elif kind != "specific":
            reply = Reply(self._photo_explanation(label) if asks_about_photo(question)
                          else self._no_disease_text(label, kind, question))
        else:
            reply = None
            if self.explainer.is_llm:
                reply = self._llm_answer(question, label)
            if reply is None:
                reply = self._compose(label, kind, self.kb.search(question, label=label), ask_label=label)
        self.history += [("user", question), ("assistant", reply.text.split("\n\n" + DISCLAIMER)[0])]
        return reply

    def _photo_explanation(self, label: str) -> str:
        """Explain the image model's result for THIS photo (no disease knowledge needed)."""
        d, (_, p) = self.diagnosis, self.ranking[0]
        crop, _, cls = label.partition("__")
        name = label.replace("__", " / ").replace("_", " ")
        level = {"confident": "confident", "uncertain": "only moderately sure"}.get(d.tier, "unsure")
        text = (f"The image model reads this photo as **{crop}**, class \"{cls.replace('_', ' ')}\". It compared your "
                f"photo with the ~9,000 labelled leaf and fruit photos it was trained on; its best match is {name}, "
                f"with a model score of {100 * p:.0f}% and my overall rating: {level}. "
                "The heatmap beside your photo shows the areas that influenced that decision most. ")
        if d.ood_flag:
            text += ("The score is high but I rate it lower because this photo looks unlike its training images "
                     "(for example a studio or stock photo, or an unusual background). ")
        if label_kind(label, self.kb) == "generic":
            text += ("The dataset only labels this class as a general 'fruit disease' without naming it, so I "
                     "cannot say which disease it is. ")
        text += ("It only knows 23 classes of apple, banana, citrus, guava, mango and papaya, so it can be wrong on "
                 "unusual photos. It is not a substitute for an expert.")
        return text + "\n\n" + DISCLAIMER

    def _identify(self, label: str) -> Reply:
        """'What is this disease?': name it with the model's confidence, then its sourced symptoms."""
        p = self.ranking[0][1]
        level = {"confident": "confident", "uncertain": "only moderately sure"}.get(self.diagnosis.tier, "unsure")
        head = (f"The image model's best match is **{label.replace('__', ' / ').replace('_', ' ')}** "
                f"(model score {100 * p:.0f}%, my rating: {level}). From the sources, this is how it looks:")
        body = self._compose(label, "specific", self.kb.for_label(label, ("symptoms",)))
        return Reply(head + "\n\n" + body.text, body.sources)

    def _help_text(self) -> str:
        return ("I can help with: (1) this photo — for example \"What crop is this?\" or \"Why are you unsure?\"; "
                "(2) a disease — for example \"What are the symptoms of black rot?\", \"How does citrus canker "
                "spread?\" or \"How do I prevent mango bacterial canker?\". What would you like to know?\n\n" + DISCLAIMER)

    def _no_disease_text(self, label: str, kind: str, question: str) -> str:
        if any(t == "assistant" and "I can still explain these diseases" in msg for t, msg in self.history):
            return ("As I mentioned, there is no specific disease to discuss for this photo. Name a disease "
                    "(for example \"citrus canker\" or \"black rot\") and I will explain it from my sources, "
                    "or upload another photo.\n\n" + DISCLAIMER)
        name = label.replace("__", " / ").replace("_", " ")
        topics = ", ".join(DISEASE_NAMES.values())
        if kind == "healthy":
            head = (f"Your photo looks healthy ({name}), so there is no disease here to treat, "
                    "and nothing for it to spread from.")
        else:
            head = (f"The training data only labels this as a general 'fruit disease' ({name}) without naming the "
                    "disease, so I can't give disease-specific answers about your photo.")
        return (f"{head} I can still explain these diseases from my sources: {topics}. "
                "Ask for example \"What are the symptoms of citrus canker?\" or \"How does banana Cordana spread?\". "
                "If you suspect a problem, upload a close-up of the affected leaf or fruit.\n\n" + DISCLAIMER)

    def _llm_prompt(self, question: str, label: str, passages: list[Passage], index: dict) -> str:
        d = self.diagnosis
        top = ", ".join(f"{n.replace('__', ' / ').replace('_', ' ')} {100 * p:.0f}%" for n, p in self.ranking[:3])
        said = "; ".join(f"{e['question']}: {e['answer']}" for e in self.evidence_log) or "nothing yet"
        if d.tier == "refuse":
            return (f"PHOTO ANALYSIS: the photo could not be analysed reliably; ignore it. The question is about: "
                    f"{DISEASE_NAMES.get(label, label)}. Likely topic: {', '.join(intent_topics(question)) or 'unclear'}.\n\n"
                    f"SOURCE PASSAGES\n" + "\n".join(f"[{','.join(str(index[x['source_url']]) for x in p.sources)}] "
                    f"({p.topic}) {p.text}" for p in passages) + f"\n\nQUESTION: {question}")
        blocks = "\n".join(f"[{','.join(str(index[s['source_url']]) for s in p.sources)}] ({p.topic}) {p.text}" for p in passages)
        hist = "\n".join(f"{'User' if r == 'user' else 'Assistant'}: {t}" for r, t in self.history[-6:]) or "(none)"
        return (f"PHOTO ANALYSIS (context only): the image model's top guesses are {top}; confidence level: "
                f"{d.tier}. User-reported details: {said}. The question is about: {DISEASE_NAMES.get(label, label)}. "
                f"Likely topic: {', '.join(intent_topics(question)) or 'unclear'}.\n\n"
                f"SOURCE PASSAGES\n{blocks}\n\nCONVERSATION SO FAR\n{hist}\n\nQUESTION: {question}")

    def _llm_answer(self, question: str, label: str) -> Reply | None:
        """LLM answer grounded in the passages for `label`; None means 'fall back to plain sourced text'."""
        passages = self.kb.for_label(label)
        cites = citations(passages)
        index = {c["source_url"]: i + 1 for i, c in enumerate(cites)}
        try:
            out = self.explainer.generate(SYSTEM_PROMPT, self._llm_prompt(question, label, passages, index))
        except Exception as e:  # network/model problems must never break the chat
            self.last_llm_status = f"llm error: {type(e).__name__}"
            return None
        if NO_ANSWER_TOKEN in out:
            self.last_llm_status = "model said no sourced answer"
            if DOSE_RE.search(question):
                return Reply("My sources do not give specific doses, concentrations or spray schedules, so I won't "
                             "guess one. Follow the product label and ask your local extension service."
                             "\n\n" + DISCLAIMER)
            if intent_topics(question):
                # The question is clearly about symptoms / spread / management and the knowledge base has
                # passages on that topic: show them rather than trusting an over-strict refusal.
                self.last_llm_status += " -> showing topic passages"
                return None
            return Reply("I don't have a sourced answer for that, so I won't guess. Please ask a local "
                         "extension service.\n\n" + DISCLAIMER)
        problem = validate_llm_answer(out, passages, question, n_sources=len(cites))
        if problem:
            self.last_llm_status = f"rejected: {problem}"
            return None
        self.last_llm_status = "accepted"
        used = sorted({int(n) for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", out) for n in re.split(r"\s*,\s*", grp)})
        notes = self._notes()
        return Reply("\n\n".join([out, *notes, DISCLAIMER]), [cites[n - 1] for n in used])

    def _notes(self) -> list[str]:
        notes = []
        if self.diagnosis.tier == "uncertain":
            notes.append("The model is only moderately sure about your photo, so treat this as general information "
                         "about the disease, not a confirmed diagnosis.")
        if self.crop_conflict:
            notes.append(f"You said this is {self.answers['crop']} but the image model's top guess was "
                         f"{self.diagnosis.top.crop}; please double-check.")
        return notes

    def _compose(self, label: str, kind: str, passages: list[Passage], ask_label: str | None = None) -> Reply:
        cites = citations(passages)
        index = {c["source_url"]: i + 1 for i, c in enumerate(cites)}
        ctx = {"label": label, "kind": kind, "passages": [
            {"topic": p.topic, "text": p.text, "refs": [index[s["source_url"]] for s in p.sources]} for p in passages]}
        text = self.explainer.advice_text(ctx)
        notes = []
        if self.diagnosis.tier == "uncertain" and ask_label is None:
            notes.append("The model is only moderately sure, so treat this as 'if it is "
                         f"{label.replace('__', ' / ').replace('_', ' ')}'.")
        if self.crop_conflict and ask_label is None:
            notes.append(f"You said this is {self.answers['crop']} but the image model's top guess was "
                         f"{self.diagnosis.top.crop}; I followed your answer, but please double-check.")
        return Reply("\n\n".join([text, *notes, DISCLAIMER]), cites)


def validate_llm_answer(text: str, passages: list[Passage], question: str, n_sources: int) -> str | None:
    """Return None if the LLM answer is acceptable, otherwise the reason it was rejected.

    - must cite at least one source and only valid source numbers
    - every number in the answer must appear in the source passages or the user's question
      (stops invented doses / schedules / percentages)
    - must be reasonably short"""
    if len(text) > 1500:
        return "too long"
    groups = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
    if not groups:
        return "no citation"
    nums = {int(n) for g in groups for n in re.split(r"\s*,\s*", g)}
    if min(nums) < 1 or max(nums) > n_sources:
        return "invalid citation number"
    body = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    allowed = " ".join(p.text for p in passages) + " " + question
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", allowed))
    for n in re.findall(r"\d+(?:\.\d+)?", body):
        if n not in allowed_numbers:
            return f"number {n} not in sources"
    return None
