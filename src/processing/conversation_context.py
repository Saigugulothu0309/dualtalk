from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationContext:
    room_code: str | None = None
    max_history: int = 8
    intent_history: list[str] = field(default_factory=list)
    sentence_history: list[str] = field(default_factory=list)

    def add_entry(self, intent: str | None, sentence: str | None):
        if intent:
            normalized_intent = str(intent).strip().lower()
            if normalized_intent:
                self.intent_history.append(normalized_intent)
                self.intent_history = self.intent_history[-self.max_history :]

        if sentence:
            normalized_sentence = str(sentence).strip()
            if normalized_sentence:
                self.sentence_history.append(normalized_sentence)
                self.sentence_history = self.sentence_history[-self.max_history :]

    def get_recent_intents(self, count: int = 2) -> list[str]:
        return self.intent_history[-count:]

    def get_recent_sentences(self, count: int = 2) -> list[str]:
        return self.sentence_history[-count:]

    def clear(self):
        self.intent_history.clear()
        self.sentence_history.clear()

    def __len__(self):
        return len(self.intent_history)
