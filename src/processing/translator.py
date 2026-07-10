from deep_translator import GoogleTranslator


SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Japanese": "ja",
    "Chinese": "zh-CN",
}


class Translator:
    """
    Translates text between languages using Google Translate.
    Falls back to the original text silently if translation fails.
    """

    def __init__(self, target_language: str = "en"):
        self.target = target_language

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if self.target == "en":
            return text
        try:
            return GoogleTranslator(source="auto", target=self.target).translate(text)
        except Exception:
            return text

    def set_language(self, language_name: str):
        if language_name in SUPPORTED_LANGUAGES:
            self.target = SUPPORTED_LANGUAGES[language_name]
        elif language_name in SUPPORTED_LANGUAGES.values():
            self.target = language_name

    @staticmethod
    def list_languages() -> dict:
        return SUPPORTED_LANGUAGES.copy()
