"""Small, offline adaptive communication engine."""
from dataclasses import dataclass, field


PROFILES = {
    "hospital": {
        "vocabulary": {"medicine": "medicine", "doctor": "doctor", "help": "assistance"},
        "templates": ["I need my {word}.", "I missed my {word}.", "Can someone bring my {word}?", "I need help with my {word}."],
        "polite": ["Please", "Could you please"],
        "emergency": ["I need urgent medical help.", "Please call a doctor."],
    },
    "education": {"vocabulary": {"help": "help", "book": "book", "water": "water"}, "templates": ["I need {word}.", "Could you help me with {word}?", "Please bring me {word}."], "polite": ["Please", "Could you please"], "emergency": ["I need immediate help." ]},
    "government": {"vocabulary": {"help": "assistance", "document": "document", "water": "water"}, "templates": ["I need {word}.", "Could you help me with my {word}?", "Please provide my {word}."], "polite": ["Please", "Could you please"], "emergency": ["I need urgent assistance."]},
    "family": {"vocabulary": {"medicine": "medicine", "help": "help", "water": "water"}, "templates": ["I need my {word}.", "Can someone bring me {word}?", "Please help me with {word}."], "polite": ["Please", "Could you"], "emergency": ["I need help right now."]},
    "general": {"vocabulary": {}, "templates": ["I need {word}.", "Can someone help me with {word}?", "Please bring me {word}."], "polite": ["Please", "Could you please"], "emergency": ["I need urgent help."]},
}


@dataclass
class ContextEngine:
    room_code: str | None = None
    max_history: int = 8
    messages: list[str] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)

    def add(self, text: str):
        text = str(text or "").strip()
        if not text:
            return
        self.messages = (self.messages + [text])[-self.max_history:]
        for key in ("medicine", "medication", "doctor", "water", "help", "book", "document"):
            if key in text.lower():
                self.entities["object"] = key

    def resolve(self, text: str) -> str:
        text = str(text or "").strip()
        return text.replace(" it ", f" {self.entities.get('object', 'it')} ") if self.entities else text

    def clear(self):
        self.messages.clear(); self.entities.clear()


class AdaptiveEngine:
    def __init__(self, profile="general", room_code=None):
        self.context = ContextEngine(room_code=room_code)
        self.set_profile(profile)

    def set_profile(self, profile):
        self.profile_name = str(profile or "general").lower()
        if self.profile_name not in PROFILES:
            self.profile_name = "general"
        self.profile = PROFILES[self.profile_name]

    def suggestions(self, gesture=None, intent=None, sentence=None, count=4):
        raw = str(sentence or intent or gesture or "").strip()
        if not raw:
            return []
        word = self.profile["vocabulary"].get(raw.lower(), raw.lower().replace("_", " "))
        if raw.lower() in {"emergency", "urgent", "danger"}:
            result = list(self.profile["emergency"])
        else:
            result = [template.format(word=word) for template in self.profile["templates"]]
        if self.context.messages and word in {"it", "that"}:
            result = [self.context.resolve(item) for item in result]
        result = list(dict.fromkeys(result))[:max(3, min(int(count), 5))]
        return result

    def choose(self, suggestions):
        return suggestions[0] if suggestions else None

    def record(self, text):
        self.context.add(text)

    def clear(self):
        self.context.clear()
