import unittest

from src.processing.conversation_context import ConversationContext, ConversationContextManager
from src.processing.prompt_builder import NLPProcessor, PromptBuilder


class NLPProcessorTests(unittest.TestCase):
    def test_nlp_processor_applies_templates_for_help_sequence(self):
        prompt = PromptBuilder(profile_name="hospital")
        processor = NLPProcessor(prompt)
        context = ConversationContext(room_code="TEST")

        sentence = processor.process("need_water", "I need water.", context)
        self.assertEqual(sentence, "I need some water.")

    def test_nlp_processor_merges_sequential_intents(self):
        prompt = PromptBuilder(profile_name="hospital")
        processor = NLPProcessor(prompt)
        context = ConversationContext(room_code="TEST")
        context.add_entry("help", "I need help.")

        sentence = processor.process("need_bathroom", "I need the bathroom.", context)
        self.assertEqual(sentence, "I need to use the bathroom.")

    def test_nlp_processor_removes_repeated_words(self):
        prompt = PromptBuilder(profile_name="hospital")
        processor = NLPProcessor(prompt)
        context = ConversationContext(room_code="TEST")

        sentence = processor.process(None, "Please please wait wait.", context)
        self.assertEqual(sentence, "Please wait.")

    def test_conversation_context_manager_tracks_room_history_and_context_aware_sentence(self):
        manager = ConversationContextManager(room_code="TEST")
        manager.addGesture("HELP")
        manager.addIntent("help")
        manager.addGesture("DRINK")
        manager.addIntent("need_water")

        recent = manager.getRecentContext()
        self.assertEqual(len(recent["gestures"]), 2)
        self.assertEqual(len(recent["intents"]), 2)
        self.assertEqual(len(recent["sentences"]), 0)
        self.assertEqual(manager.generateContextAwareSentence(), "I need some water.")

    def test_conversation_context_manager_clears_room_context_on_leave(self):
        manager = ConversationContextManager(room_code="TEST")
        manager.addGesture("HELP")
        manager.addIntent("help")

        manager.clearContext()
        recent = manager.getRecentContext()
        self.assertEqual(recent["gestures"], [])
        self.assertEqual(recent["intents"], [])
        self.assertEqual(recent["sentences"], [])


if __name__ == "__main__":
    unittest.main()
