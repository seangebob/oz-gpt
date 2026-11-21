import unittest
from unittest.mock import Mock

from model.book_lookup import AnswerSpan, BookIndex, BookSession, NO_ANSWER, book_passages
from model.chat_evaluation import generate_session_predictions, grounded_correctness


BOOK = '''DOROTHY AND THE WIZARD IN OZ
BY L. FRANK BAUM
LIST OF CHAPTERS
1 THE EARTHQUAKE
CHAPTER 1.
Dorothy traveled with a kitten named Eureka.
CHAPTER 2.
Jim is a cab-horse, and Zeb drives the buggy.
'''


class BookLookupTests(unittest.TestCase):
    def setUp(self):
        self.passages = list(book_passages(BOOK))
        self.index = BookIndex(self.passages)

    def test_index_preserves_source_lines_and_excludes_contents(self):
        self.assertEqual([passage.section for passage in self.passages], ['Title page', 'Chapter 1', 'Chapter 2'])
        self.assertEqual(self.passages[1].start_line, 5)
        self.assertEqual(self.passages[1].end_line, 6)
        self.assertNotIn('LIST OF CHAPTERS', ' '.join(passage.text for passage in self.passages))
        self.assertEqual(self.index.search('Who wrote the book by Baum?')[0][0].section, 'Title page')

    def test_session_abstains_without_a_supported_span(self):
        reader = Mock()
        reader.read.return_value = [AnswerSpan('Kansas', 100)]
        session = BookSession(self.index, reader)
        self.assertEqual(session.reply('Who was Napoleon?', remember=False).text, NO_ANSWER)
        reader.read.assert_not_called()
        self.assertIsNone(session.reply('Who drives the buggy?').source)
        reader.read.return_value = [AnswerSpan('Zeb', 0)]
        self.assertIsNone(session.reply('Who drives the buggy?').source)

    def test_followups_use_previous_subject_and_report_citations(self):
        def read(question, passages):
            if question == 'Who is Jim?':
                text = 'cab-horse'
            elif question == 'Who is with Jim?':
                text = 'Zeb'
            else:
                text = ''
            return [AnswerSpan(text, 8) for _ in passages]

        reader = Mock()
        reader.read.side_effect = read
        session = BookSession(self.index, reader)
        turns = [(1, {'messages': [
            {'role': 'user', 'content': 'Who is Jim?'},
            {'role': 'assistant', 'content': 'A cab-horse.'},
            {'role': 'user', 'content': 'Who is with him?'},
            {'role': 'assistant', 'content': 'Zeb.'},
        ]})]
        rows = generate_session_predictions(session, turns, 'generated')
        self.assertEqual(rows[1]['reply'], 'Zeb')
        self.assertTrue(rows[1]['history_used'])
        self.assertTrue(rows[1]['reference_match'])
        self.assertEqual(rows[1]['source']['section'], 'Chapter 2')
        self.assertIn('Zeb', rows[1]['source']['text'])
        self.assertIsNone(rows[1]['source_supports_reply'])

        rows[1]['source_supports_reply'] = False
        self.assertEqual(grounded_correctness(rows[1:]), '0/1')
        rows[1]['source_supports_reply'] = True
        self.assertEqual(grounded_correctness(rows[1:]), '1/1')
        session.reset()
        result = session.reply('Who is with him?')
        self.assertEqual(result.text, NO_ANSWER)
        self.assertFalse(result.history_used)


if __name__ == '__main__':
    unittest.main()
