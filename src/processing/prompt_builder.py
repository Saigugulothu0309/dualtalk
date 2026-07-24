from __future__ import annotations

from dataclasses import dataclass

from src.processing.gesture_profile import get_gesture_profile
from src.processing.conversation_context import ConversationContext


@dataclass
class PromptBuilder:
    profile_name: str = "hospital"

    def __post_init__(self):
        self.profile = get_gesture_profile(self.profile_name)

    def vocabulary(self) -> dict[str, str]:
        return getattr(self.profile, "vocabulary", {}) or {}

    def sentence_templates(self) -> dict[str, str]:
        return getattr(self.profile, "sentence_templates", {}) or {}

    def polite_responses(self) -> dict[str, str]:
        return getattr(self.profile, "polite_responses", {}) or {}

    def emergency_phrases(self) -> dict[str, str]:
        return getattr(self.profile, "emergency_phrases", {}) or {}

    def get_template(self, key: str) -> str | None:
        if not key:
            return None
        return self.sentence_templates().get(str(key).strip().lower())

    def get_polite_response(self, key: str) -> str | None:
        if not key:
            return None
        return self.polite_responses().get(str(key).strip().lower())

    def get_emergency_phrase(self, key: str) -> str | None:
        if not key:
            return None
        return self.emergency_phrases().get(str(key).strip().lower())

    def get_vocabulary_word(self, key: str) -> str | None:
        if not key:
            return None
        return self.vocabulary().get(str(key).strip().lower())


class NLPProcessor:
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder

    def process(
        self,
        intent: str | None,
        sentence: str | None,
        context: ConversationContext,
    ) -> str | None:
        if sentence is None and intent is None:
            return None

        cleaned = self._clean_sentence(sentence or "")
        if intent:
            cleaned = self._apply_intent_template(intent, cleaned)

        cleaned = self._merge_context(context.get_recent_intents(), cleaned)
        cleaned = self._complete_sentence(cleaned)
        cleaned = self._apply_corrections(cleaned)
        return cleaned

    def _clean_sentence(self, sentence: str) -> str:
        sentence = str(sentence or "").strip()
        if not sentence:
            return ""
        words = sentence.split()
        deduped = []
        last_word = None
        for word in words:
            normalized_word = word.strip('.,!?;:"\'').lower()
            if normalized_word == last_word:
                continue
            deduped.append(word)
            last_word = normalized_word
        return " ".join(deduped)

    def _apply_intent_template(self, intent: str, sentence: str) -> str:
        template = self.prompt_builder.get_template(intent)
        if template:
            return template

        if sentence:
            return sentence

        if intent in {"help", "need_help"}:
            return self.prompt_builder.get_template("help") or "I need help."
        if intent == "pause":
            return self.prompt_builder.get_template("pause") or "Please wait for a moment."
        if intent in {"affirmative", "negative", "gratitude"}:
            return self.prompt_builder.get_polite_response(intent) or sentence
        if intent == "emergency":
            return self.prompt_builder.get_emergency_phrase("emergency") or "This is an emergency."

        return sentence

    def _merge_context(self, intents: list[str], sentence: str) -> str:
        if len(intents) >= 2:
            combined_key = " ".join(intents[-2:])
            template = self.prompt_builder.get_template(combined_key)
            if template:
                return template

        if sentence:
            return sentence

        if intents:
            first_intent = intents[-1]
            template = self.prompt_builder.get_template(first_intent)
            if template:
                return template

        return sentence

    def _complete_sentence(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""

        lower_text = normalized.lower()
        if lower_text.startswith("need "):
            normalized = f"I {normalized}"
        elif lower_text == "bathroom":
            normalized = "I need to use the bathroom."
        elif lower_text.startswith("bathroom"):
            normalized = "I need to use the bathroom."
        elif lower_text.startswith("drink"):
            normalized = "I need some water."
        elif lower_text.startswith("eat"):
            normalized = "I need something to eat."
        elif lower_text.startswith("wait"):
            normalized = "Please wait for a moment."
        elif lower_text == "yes thanks" or lower_text == "thanks yes":
            normalized = "Yes, thank you."
        elif lower_text == "yes thank you" or lower_text == "thank you yes":
            normalized = "Yes, thank you."
        elif lower_text == "thanks" or lower_text == "thank you":
            normalized = "Thank you."

        return normalized

    def _apply_corrections(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""

        while "  " in normalized:
            normalized = normalized.replace("  ", " ")
        normalized = normalized.replace(" ,", ",")
        normalized = normalized.replace(" .", ".")
        if not normalized.endswith("."):
            normalized = f"{normalized}."
        return normalized
