import numpy as np
import pytest
from PIL import Image

from src.assistant.chat import AssistantSession, label_kind
from src.assistant.evidence import apply_evidence
from src.assistant.knowledge import KnowledgeBase
from src.assistant.pipeline import Candidate, Diagnoser, Diagnosis

LABELS = ["apple__black_rot", "apple__healthy_leaf", "citrus__canker_leaf", "mango__bacterial_canker",
          "mango__disease_fruit", "papaya__ring_spot"]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


class StubDiagnoser:
    labels = LABELS

    def __init__(self, probs, tier):
        self.probs, self.tier = np.array(probs, dtype=float), tier

    def diagnose(self, image):
        order = self.probs.argsort()[::-1][:3]
        cands = [Candidate(LABELS[i], *LABELS[i].split("__"), float(self.probs[i])) for i in order]
        return Diagnosis(self.tier, cands, 0.0, None, self.probs)


def session(kb, probs, tier):
    return AssistantSession(StubDiagnoser(probs, tier), kb)


IMG = Image.new("RGB", (10, 10))


# --- evidence ---------------------------------------------------------------------
def test_evidence_noop_without_answers():
    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    out, log = apply_evidence(p, LABELS, {})
    assert np.allclose(out, p) and log == []


def test_lesion_answer_boosts_matching_label_and_logs_it():
    p = np.array([0.1, 0.1, 0.1, 0.3, 0.2, 0.2])
    out, log = apply_evidence(p, LABELS, {"lesion": "pustule_halo"})
    assert out[2] > p[2] and np.isclose(out.sum(), 1)
    assert log and log[0]["passage"] == "citrus_canker_sym"


def test_crop_answer_restricts_to_that_crop():
    p = np.array([0.6, 0.2, 0.05, 0.05, 0.05, 0.05])
    out, _ = apply_evidence(p, LABELS, {"crop": "mango"})
    assert out[0] == 0 and out[1] == 0 and np.isclose(out.sum(), 1)
    assert out.argmax() in (3, 4)


def test_unsure_answers_change_nothing():
    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    out, log = apply_evidence(p, LABELS, {"crop": "unsure", "lesion": "none", "weather": "unsure"})
    assert np.allclose(out, p) and log == []


# --- knowledge --------------------------------------------------------------------
def test_every_passage_has_a_source(kb):
    assert all(p.sources and all(s["source_url"].startswith("http") for s in p.sources) for p in kb.passages)


def test_search_declines_off_topic(kb):
    assert kb.search("what is the price of tomatoes in Delhi", label="mango__bacterial_canker") == []
    assert kb.search("how do I fix my laptop", label="mango__bacterial_canker") == []


def test_short_intent_questions_are_answered(kb):
    spread = kb.search("how does it spread?", label="mango__bacterial_canker")
    assert spread and spread[0].topic == "spread"
    assert kb.search("what should I do about it?", label="apple__black_rot")[0].topic == "management"
    assert kb.search("what are the symptoms", label="papaya__ring_spot")[0].topic == "symptoms"


def test_search_respects_label(kb):
    hits = kb.search("pruning tools spread bacteria", label="mango__bacterial_canker")
    assert hits and all("mango__bacterial_canker" in p.labels for p in hits)


# --- chat guardrails ----------------------------------------------------------------
def test_refuse_tier_gives_no_diagnosis_or_advice(kb):
    s = session(kb, [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], "refuse")
    assert "can't give a reliable diagnosis" in s.start(IMG).text
    assert s.pending_questions() == []
    assert "won't give treatment advice" in s.advice().text
    assert s.ask("how to treat it").sources == []


def test_confident_specific_label_gets_cited_advice(kb):
    s = session(kb, [0.02, 0.01, 0.01, 0.94, 0.01, 0.01], "confident")
    s.start(IMG)
    r = s.advice()
    assert r.sources and "[1]" in r.text and "extension" in r.text.lower()


def test_generic_fruit_disease_gives_no_specific_treatment(kb):
    s = session(kb, [0.01, 0.01, 0.01, 0.01, 0.95, 0.01], "confident")
    s.start(IMG)
    r = s.advice()
    assert label_kind("mango__disease_fruit", kb) == "generic"
    assert r.sources == [] and "can't give disease-specific" in r.text


def test_healthy_label_says_no_disease(kb):
    s = session(kb, [0.05, 0.9, 0.01, 0.01, 0.02, 0.01], "confident")
    s.start(IMG)
    assert "No disease detected" in s.advice().text


def test_uncertain_tier_is_flagged_in_advice(kb):
    s = session(kb, [0.02, 0.01, 0.01, 0.7, 0.2, 0.06], "uncertain")
    s.start(IMG)
    assert "only moderately sure" in s.advice().text


def test_crop_conflict_is_disclosed(kb):
    s = session(kb, [0.02, 0.01, 0.01, 0.9, 0.05, 0.01], "confident")
    s.start(IMG)
    s.answer("crop", "papaya")
    assert s.crop_conflict and "double-check" in s.advice().text


def test_unknown_question_id_rejected(kb):
    s = session(kb, [0.9, 0.02, 0.02, 0.02, 0.02, 0.02], "confident")
    s.start(IMG)
    with pytest.raises(KeyError):
        s.answer("favourite_colour", "blue")


# --- tiers -------------------------------------------------------------------------
def test_ood_flag_downgrades_one_tier_and_explains(kb):
    from src.assistant.pipeline import TIER_DOWN
    assert (TIER_DOWN["confident"], TIER_DOWN["uncertain"], TIER_DOWN["refuse"]) == ("uncertain", "refuse", "refuse")
    probs = np.array([0.02, 0.01, 0.01, 0.94, 0.01, 0.01])
    s = session(kb, probs, "uncertain")
    s.diagnoser.diagnose = lambda img: Diagnosis(
        "uncertain", [Candidate(LABELS[3], "mango", "bacterial_canker", 0.94)], 0.0, None, probs,
        99.0, True, ["This photo looks unlike the images the model was trained on"])
    first = s.start(IMG).text
    assert "looks unlike the images" in " ".join(s.diagnosis.notes) and "unusual" in first


def test_tier_thresholds():
    d = object.__new__(Diagnoser)
    d.t_confident, d.t_refuse = 0.8, 0.6
    assert (d.tier_for(0.9), d.tier_for(0.7), d.tier_for(0.5)) == ("confident", "uncertain", "refuse")


# --- LLM answers: grounding guard + follow-up behaviour ---------------------------------
class ScriptedLLM:
    """Stands in for a real model: returns a fixed string and records the prompt it was given."""
    name, label, is_llm = "scripted", "scripted", True

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, system, user):
        self.prompts.append(user)
        return self.reply



def llm_session(kb, reply, probs=(0.02, 0.01, 0.01, 0.94, 0.01, 0.01), tier="confident"):
    from src.assistant.llm import TemplateExplainer

    class LLM(ScriptedLLM, TemplateExplainer):
        pass

    llm = LLM(reply)
    s = AssistantSession(StubDiagnoser(probs, tier), kb, llm)
    s.start(IMG)
    return s, llm


def test_llm_answer_with_valid_citation_is_accepted_and_sources_limited(kb):
    s, llm = llm_session(kb, "The bacteria spread by wind-blown rain and on nursery trees. [1]")
    r = s.ask("how does it spread?")
    assert s.last_llm_status == "accepted" and "wind-blown rain" in r.text
    assert len(r.sources) == 1 and r.sources[0]["source_url"].startswith("http")
    assert "SOURCE PASSAGES" in llm.prompts[0] and "how does it spread?" in llm.prompts[0]


def test_llm_answer_without_citation_is_rejected_and_falls_back_to_sourced_text(kb):
    s, _ = llm_session(kb, "It spreads by wind and rain.")
    r = s.ask("how does it spread?")
    assert s.last_llm_status == "rejected: no citation"
    assert r.sources and "[1]" in r.text          # plain sourced passage instead


def test_llm_answer_with_invented_number_is_rejected(kb):
    s, _ = llm_session(kb, "Spray copper at 7 grams per litre every 3 days. [1]")
    s.ask("what dose of copper?")
    assert s.last_llm_status.startswith("rejected: number 7")


def test_llm_answer_with_bad_citation_number_is_rejected(kb):
    s, _ = llm_session(kb, "They spread by rain. [9]")
    s.ask("how does it spread?")
    assert s.last_llm_status == "rejected: invalid citation number"


def test_llm_no_sourced_answer_token_gives_standard_decline(kb):
    from src.assistant.llm import NO_ANSWER_TOKEN
    s, _ = llm_session(kb, NO_ANSWER_TOKEN)
    r = s.ask("what is the capital of France?")
    assert "sourced answer" in r.text and r.sources == []


def test_llm_exception_never_breaks_the_chat(kb):
    class Boom:
        name = label = "boom"
        is_llm = True

        def generate(self, *a):
            raise TimeoutError("model down")

        from src.assistant.llm import TemplateExplainer as _T
        diagnosis_text = _T.diagnosis_text
        advice_text = _T.advice_text

    s = AssistantSession(StubDiagnoser([0.02, 0.01, 0.01, 0.94, 0.01, 0.01], "confident"), kb, Boom())
    s.start(IMG)
    r = s.ask("how does it spread?")
    assert s.last_llm_status == "llm error: TimeoutError" and r.sources


def test_healthy_photo_followups_explain_instead_of_repeating_a_stock_line(kb):
    s = session(kb, [0.05, 0.9, 0.01, 0.01, 0.02, 0.01], "confident")
    s.start(IMG)
    r = s.ask("how does it spread?")
    assert "looks healthy" in r.text and "citrus canker" in r.text.lower()


def test_user_can_ask_about_a_named_disease_even_when_photo_is_healthy(kb):
    s = session(kb, [0.05, 0.9, 0.01, 0.01, 0.02, 0.01], "confident")
    s.start(IMG)
    r = s.ask("How does citrus canker spread?")
    assert r.sources and "wind-driven rain" in r.text
    from src.assistant.chat import disease_in_text
    assert disease_in_text("tell me about bacterial canker on mango") == "mango__bacterial_canker"
    assert disease_in_text("what is canker") == "citrus__canker_leaf"
    assert disease_in_text("hello") is None


def test_why_questions_explain_the_photo_result_and_repeats_are_shortened(kb):
    s = session(kb, [0.05, 0.9, 0.01, 0.01, 0.02, 0.01], "confident")
    s.start(IMG)
    why = s.ask("What is the reason?").text
    assert "model score" in why and "heatmap" in why
    first = s.ask("how does it spread?").text
    second = s.ask("and how do I treat it?").text
    assert "I can still explain these diseases" in first
    assert "As I mentioned" in second and second != first


def test_photo_questions_are_answered_from_the_result_even_for_generic_fruit_class(kb):
    s = session(kb, [0.01, 0.01, 0.01, 0.01, 0.95, 0.01], "confident")   # mango__disease_fruit (generic)
    s.start(IMG)
    r = s.ask("What crop is this?")
    assert "reads this photo as **mango**" in r.text and "cannot say which disease" in r.text


def test_vague_one_word_question_gets_a_help_prompt(kb):
    s = session(kb, [0.01, 0.01, 0.01, 0.01, 0.95, 0.01], "confident")
    s.start(IMG)
    r = s.ask("What?")
    assert "What would you like to know?" in r.text and r.sources == []


def test_refusal_on_a_treatment_question_falls_back_to_topic_passages(kb):
    from src.assistant.llm import NO_ANSWER_TOKEN
    s, _ = llm_session(kb, NO_ANSWER_TOKEN)
    r = s.ask("How to cure it?")
    assert r.sources and "Copper" in r.text or "Use disease-free" in r.text       # management passage shown
    r2 = s.ask("how do I get rid of it")
    assert r2.sources


def test_dose_questions_get_a_specific_decline_not_a_made_up_dose(kb):
    from src.assistant.llm import NO_ANSWER_TOKEN
    s, _ = llm_session(kb, NO_ANSWER_TOKEN)
    r = s.ask("how much copper per litre should I spray?")
    assert "do not give specific doses" in r.text and r.sources == []


def test_off_topic_still_declines_after_refusal_fallback_was_added(kb):
    from src.assistant.llm import NO_ANSWER_TOKEN
    s, _ = llm_session(kb, NO_ANSWER_TOKEN)
    for q in ("what is the price of tomatoes in Delhi?", "how do I fix my laptop", "tell me a joke"):
        r = s.ask(q)
        assert r.sources == [] and "sourced answer" in r.text


def test_what_is_this_disease_names_it_with_confidence_then_symptoms(kb):
    s = session(kb, [0.02, 0.01, 0.01, 0.94, 0.01, 0.01], "confident")
    s.start(IMG)
    r = s.ask("What is this disease?")
    assert "mango / bacterial canker" in r.text and "model score" in r.text and "angular" in r.text and r.sources


def test_followup_replies_do_not_repeat_the_disclaimer_but_advice_still_has_it(kb):
    from src.assistant.llm import DISCLAIMER
    s, _ = llm_session(kb, "The bacteria spread by wind-blown rain and on nursery trees. [1]")
    for q in ("how does it spread?", "what is the price of tomatoes?", "What crop is this?", "What?",
              "What is this disease?", "How to cure it?"):
        assert DISCLAIMER not in s.ask(q).text, q
    assert DISCLAIMER in s.advice().text


def test_named_disease_question_is_answered_even_when_the_photo_is_undiagnosable(kb):
    s = session(kb, [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], "refuse")
    s.start(IMG)
    r = s.ask("How does mango bacterial canker spread?")
    assert r.sources and "wind-blown rain" in r.text
    assert "upload a clearer photo" in s.ask("How does it spread?").text      # unnamed -> still needs the photo
