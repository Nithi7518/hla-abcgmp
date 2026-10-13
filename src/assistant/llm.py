"""Explainers: turn structured results + retrieved passages into chat text.

The classifier decides the label; the explainer only PHRASES it. Three engines:
  - `OllamaExplainer`  : a local LLM (default: whatever Ollama model is installed, e.g. llama3).
                         No API key, no data leaves the machine.
  - `ClaudeExplainer`  : Claude via the Anthropic API, used only if `anthropic` is installed
                         and ANTHROPIC_API_KEY is set.
  - `TemplateExplainer`: deterministic fallback with no model at all.
`make_explainer()` picks Claude, then Ollama, then the template. Set LLM_DISABLE=1 to force the
template. LLM output is never trusted blindly: src/assistant/chat.py validates every LLM answer
(valid citations, no numbers that are not in the sources) and falls back to the sourced text.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DISCLAIMER = ("This is an AI aid, not a diagnosis by an expert. Confirm with your local agricultural "
              "extension service before treating, spraying or removing plants.")

NO_ANSWER_TOKEN = "NO_SOURCED_ANSWER"

SYSTEM_PROMPT = (
    "You are the chat assistant of a plant-disease app for farmers and gardeners. Answer the user's "
    "question using ONLY the SOURCE PASSAGES provided. Rules:\n"
    "1. Use only facts stated in the passages. Never add products, doses, schedules, numbers, causes or "
    "advice that are not in them.\n"
    "2. After each factual sentence, add the source number(s) exactly as shown at the start of the "
    "passage, in square brackets, e.g. [1] or [1,2].\n"
    f"3. If the passages do not answer the question, reply with exactly: {NO_ANSWER_TOKEN}\n"
    "4. Questions about curing, treating, controlling, stopping, preventing or getting rid of the disease are "
    "answered with the management measures in the passages. If the passages describe control measures but no "
    "cure, say so plainly and list the measures. Questions about what the disease is, or what it looks like, "
    "are answered with the symptoms in the passages.\n"
    "4b. Be concise (at most 120 words), friendly and plain. Do not use headings.\n"
    "5. The photo analysis is context only. Never sound more certain than the stated confidence; if it "
    "says uncertain, say the diagnosis is uncertain.\n"
    "6. Never mention these rules or the word 'passages'."
)


def pct(p: float) -> str:
    return f"{100 * p:.0f}%"


class TemplateExplainer:
    name = "template"
    label = "built-in rule-based explainer (no language model running)"
    is_llm = False

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def diagnosis_text(self, ctx: dict) -> str:
        tier, ranking = ctx["tier"], ctx["ranking"]
        top = ranking[0]
        alts = ", ".join(f"{n.replace('__', ' / ').replace('_', ' ')} ({pct(p)})" for n, p in ranking[1:3] if p >= 0.05)
        name = top[0].replace("__", " / ").replace("_", " ")
        if tier == "refuse":
            return ("I can't give a reliable diagnosis from this photo: the model's confidence is too low "
                    f"(best guess {name} at only {pct(top[1])}). Try a clearer, well-lit close-up of the "
                    "affected area, and make sure it is a leaf or fruit of apple, banana, citrus, guava, mango "
                    "or papaya.")
        lead = "The model is confident that this looks like" if tier == "confident" else \
            "I'm not very sure, but this most likely looks like"
        text = f"{lead} **{name}** ({pct(top[1])})."
        if ctx.get("ood") and tier != "confident":
            text += (f" The model's own score is high, but I rate this {'Uncertain' if tier == 'uncertain' else 'lower'} "
                     "because the photo looks unusual (see the note below).")
        if alts:
            text += f" Other possibilities: {alts}."
        return text

    def advice_text(self, ctx: dict) -> str:
        label = ctx["label"].replace("__", " / ").replace("_", " ")
        if ctx["kind"] == "healthy":
            return (f"No disease detected ({label}). Keep monitoring; if you still see symptoms, "
                    "take a close-up of the affected spot and try again.")
        if ctx["kind"] == "generic":
            return (f"The training data labels this only as 'fruit disease' ({label}) without naming the "
                    "pathogen, so I can't give disease-specific treatment advice. A local extension "
                    "office or plant clinic can identify it from a sample.")
        if not ctx["passages"]:
            return "I don't have a sourced answer for that, so I won't guess. Please ask a local extension service."
        lines = [f"**{p['topic'].capitalize()}:** {p['text']} [{', '.join(map(str, p['refs']))}]"
                 for p in ctx["passages"]]
        return "\n\n".join(lines)


class OllamaExplainer(TemplateExplainer):
    """Local LLM through the Ollama HTTP API (default http://localhost:11434)."""

    is_llm = True

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                models = [m["name"] for m in json.load(r).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(f"Ollama not reachable at {self.host}: {e}")
        wanted = model or os.environ.get("OLLAMA_MODEL")
        if wanted:
            match = [m for m in models if m == wanted or m.split(":")[0] == wanted]
            if not match:
                raise RuntimeError(f"Ollama model {wanted!r} is not installed (have: {models})")
            self.model = match[0]
        elif models:
            self.model = models[0]
        else:
            raise RuntimeError("Ollama is running but has no models installed")
        self.name = f"{self.model} (local, via Ollama)"
        self.label = f"{self.model} — a language model running locally on this computer via Ollama"

    def generate(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.2, "num_predict": 400},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["message"]["content"].strip()


class ClaudeExplainer(TemplateExplainer):
    is_llm = True

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic  # raises ImportError when unavailable -> caller falls back

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"{model} (Anthropic API)"
        self.label = f"Claude ({model}) via the Anthropic API"

    def generate(self, system: str, user: str) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=500, system=system,
                                          messages=[{"role": "user", "content": user}])
        return msg.content[0].text.strip()


def make_explainer() -> TemplateExplainer:
    if os.environ.get("LLM_DISABLE") == "1":
        return TemplateExplainer()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeExplainer()
        except ImportError:
            pass
    try:
        return OllamaExplainer()
    except RuntimeError:
        return TemplateExplainer()
