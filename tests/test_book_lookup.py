import unittest
from unittest.mock import Mock, patch

from model.book_lookup import AnswerSpan, BookIndex, BookSession, NO_ANSWER, Passage, book_passages, followup_question
from model.chat_evaluation import generate_session_predictions, grounded_correctness, passage_review_counts


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

    def test_chapter_heading_helps_rank_passages_later_in_the_chapter(self):
        unrelated = Passage('Chapter 1', 10, 12, 'Zeb stopped on the road.', 'THE EARTHQUAKE')
        relevant = Passage('Chapter 2', 30, 32, 'Zeb stopped on the road.', 'ZEB RETURNS TO THE RANCH')
        index = BookIndex([unrelated, relevant])
        self.assertEqual(index.search('Where did Zeb return?')[0][0], relevant)

    @patch('model.book_lookup.PASSAGE_WORDS', 10)
    @patch('model.book_lookup.PASSAGE_STEP', 8)
    def test_retrieval_completes_a_matching_sentence_at_either_edge(self):
        book = '''CHAPTER 1.

The little piglet hid where nobody could find it
until it fell into a big vase.

CHAPTER 2.

The Wizard was born in Omaha.
'''
        plain_index = BookIndex(book_passages(book))
        index = BookIndex.from_text(book)

        for question, first_line in [('Where was the piglet?', 1), ('What was in the vase?', 3)]:
            with self.subTest(question=question):
                original, original_score = plain_index.search(question)[0]
                extended, score = index.search(question)[0]
                self.assertEqual(score, original_score)
                self.assertIn(original.text, extended.text)
                self.assertIn('The little piglet hid where nobody could find it until it fell into a big vase.', extended.text)
                self.assertEqual((extended.start_line, extended.end_line), (first_line, 4))
                self.assertEqual(extended.section, 'Chapter 1')

        unrelated = 'Which chapter?'
        self.assertEqual(index.search(unrelated), plain_index.search(unrelated))

    def test_session_abstains_without_a_supported_span(self):
        reader = Mock()
        reader.read.return_value = [AnswerSpan('Kansas', 100)]
        session = BookSession(self.index, reader)
        self.assertEqual(session.reply('Who was Napoleon?', remember=False).text, NO_ANSWER)
        reader.read.assert_not_called()
        self.assertIsNone(session.reply('Who drives the buggy?').source)
        reader.read.return_value = [AnswerSpan('Zeb', 0)]
        self.assertIsNone(session.reply('Who drives the buggy?').source)

    def test_trace_keeps_search_order_and_marks_the_selected_passage(self):
        index = Mock()
        index.search.return_value = [(self.passages[1], 3.0), (self.passages[2], 1.0)]
        reader = Mock()
        reader.read.return_value = [AnswerSpan('Dorothy', 3, 24), AnswerSpan('Zeb', 9)]
        session = BookSession(index, reader, trace_passages=True)
        turns = [(1, {'messages': [
            {'role': 'user', 'content': 'Who drives the buggy?'},
            {'role': 'assistant', 'content': 'Zeb.'},
        ]})]
        rows = generate_session_predictions(session, turns, 'generated')
        trace = rows[0]['passage_trace']
        self.assertEqual(trace['selected_rank'], 2)
        self.assertEqual([p['rank'] for p in trace['candidates']], [1, 2])
        self.assertEqual(trace['candidates'][0]['reader_text'], self.passages[1].text[:24])
        self.assertEqual(trace['candidates'][1]['reader_span'], 'Zeb')
        self.assertEqual(passage_review_counts(rows), (0, 0, 1))
        trace['evidence_ranks'] = [2]
        self.assertEqual(passage_review_counts(rows), (1, 0, 0))
        trace['evidence_ranks'] = []
        self.assertEqual(passage_review_counts(rows), (0, 1, 0))
        trace['evidence_ranks'] = [3]
        with self.assertRaisesRegex(ValueError, 'retrieved passage ranks'):
            passage_review_counts(rows)

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
        session = BookSession(self.index, reader, trace_passages=True)
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
        self.assertEqual(rows[1]['passage_trace']['query'], 'Who is with Jim?')
        self.assertEqual(rows[1]['passage_trace']['previous_question'], 'Who is Jim?')

        rows[1]['source_supports_reply'] = False
        self.assertEqual(grounded_correctness(rows[1:]), '0/1')
        rows[1]['source_supports_reply'] = True
        self.assertEqual(grounded_correctness(rows[1:]), '1/1')
        session.reset()
        result = session.reply('Who is with him?')
        self.assertEqual(result.text, NO_ANSWER)
        self.assertFalse(result.history_used)

    def test_followup_search_anchors_the_item_to_the_previous_action(self):
        index = Mock()
        index.search.return_value = []
        session = BookSession(index, Mock(), trace_passages=True)
        previous = 'What did Clara reveal at the fair?'
        session.remember(previous, 'exceedingly thoughtful')
        reply = session.reply('Where was it?')
        index.search.assert_called_once_with('Where did Clara produce the item from?', previous)
        self.assertEqual(reply.passage_trace.previous_answer, 'exceedingly thoughtful')

    def test_followup_rewrites_use_the_question_for_people_and_objects(self):
        cases = [
            ('What did he fix?', 'What did the Tin Woodman use at the palace?', 'a branch',
             'What did the Tin Woodman fix?'),
            ('Where was it?', 'Which little dog went missing?', 'a neighbor',
             'Where was the little dog that went missing?'),
            ('Where is it?', 'Which red dog was lost?', 'a neighbor', 'Where is the red dog that was lost?'),
            ('Why that name?', 'Who is in the stable?', 'Comet', 'Why was Comet given that name?'),
            ('Where did she lose them?', 'Which witch owned the silver shoes?', 'the Wicked Witch',
             'Where did she lose the silver shoes?'),
        ]
        for question, previous, answer, expected in cases:
            with self.subTest(question=question, previous=previous):
                self.assertEqual(followup_question(question, previous, answer), expected)


if __name__ == '__main__':
    unittest.main()
