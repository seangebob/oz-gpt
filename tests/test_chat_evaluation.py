import unittest
from unittest.mock import Mock

import torch

from model.book_lookup import AnswerSpan, BookSession, Passage
from model.chat_data import END_OF_TURN, ChatTokenizer, format_prompt, format_turn
from model.chat_evaluation import correctness, generate_predictions, reference_matches
from model.chat_session import ChatSession, DEFAULT_MAX_TOKENS, REPLY_RESERVE
from model.first_reply import FirstReplyClassifier, fit_first_replies
from model.model import block_size, chars


class ChatEvaluationTests(unittest.TestCase):
    def test_first_reply_classifier_selects_learned_answers_and_falls_back(self):
        classifier = FirstReplyClassifier(fit_first_replies([
            ('What does a sparrow eat?', 'Seeds.'), ('What is a sparrow fed?', 'Seeds.'),
            ('What does a panda eat?', 'Bamboo.'), ('What is a panda fed?', 'Bamboo.'),
        ]))
        self.assertEqual(classifier.predict('What do sparrows eat?'), 'Seeds.')
        self.assertEqual(classifier.predict('What do pandas eat?'), 'Bamboo.')
        self.assertIsNone(classifier.predict('12345'))

    def test_classifier_answers_only_the_first_turn_in_each_history_mode(self):
        tokenizer = ChatTokenizer(chars)
        conversations = [(1, {'messages': [
            {'role': 'user', 'content': 'Who is Jim?'},
            {'role': 'assistant', 'content': 'A horse.'},
            {'role': 'user', 'content': 'What kind?'},
            {'role': 'assistant', 'content': 'A cab-horse.'},
        ]})]
        for mode in ('generated', 'none', 'reference'):
            with self.subTest(mode=mode):
                model, classifier = Mock(), Mock()
                classifier.predict.return_value = 'A horse.'

                def generate(context, *_args, **_kwargs):
                    return torch.cat((context, torch.tensor([tokenizer.encode('A cab-horse.' + END_OF_TURN)])), dim=1)

                model.generate.side_effect = generate
                rows = generate_predictions(model, tokenizer, conversations, mode, DEFAULT_MAX_TOKENS,
                                            first_reply_classifier=classifier)
                classifier.predict.assert_called_once_with('Who is Jim?')
                self.assertEqual([row['reply_method'] for row in rows], ['classifier', 'generator'])
                self.assertEqual([row['clean_stop'] for row in rows], [None, True])
                self.assertEqual(rows[1]['history_used'], mode != 'none')
                self.assertTrue(all(row['reference_match'] for row in rows))

    def test_book_assistance_preserves_classifier_and_its_followup(self):
        tokenizer = ChatTokenizer(chars)
        book = Mock(spec=BookSession, previous_source=None)
        model, classifier = Mock(), Mock()
        classifier.predict.return_value = 'A horse.'

        def generate(context, *_args, **_kwargs):
            return torch.cat((context, torch.tensor([tokenizer.encode('A cab-horse.' + END_OF_TURN)])), dim=1)

        model.generate.side_effect = generate
        session = ChatSession(model, tokenizer, first_reply_classifier=classifier, book_session=book)

        first = session.reply('Who is Jim?')
        second = session.reply('What kind?')
        third = session.reply('What happened next?')

        self.assertEqual((first.reply_method, second.reply_method, third.reply_method),
                         ('classifier', 'generator', 'generator'))
        self.assertTrue(second.history_used)
        book.reply.assert_not_called()
        book.remember.assert_called_with('What happened next?', 'A cab-horse.')

    def test_cited_first_reply_can_override_an_unrelated_classifier_answer(self):
        tokenizer = ChatTokenizer(chars)
        passage = Passage('Chapter 1', 10, 14, 'Dorothy will carry a basket to the garden. Her basket was red.')
        book = BookSession(Mock(search=Mock(return_value=[(passage, 2.0)])),
                           Mock(read=Mock(return_value=[AnswerSpan('a basket', 7)])), trace_passages=True)
        classifier = Mock(predict=Mock(return_value='A lantern.'))
        session = ChatSession(Mock(), tokenizer, first_reply_classifier=classifier, book_session=book)

        reply = session.reply('What did Dorothy carry to the garden?', remember=False)
        self.assertEqual((reply.text, reply.reply_method, reply.source), ('a basket', 'book', passage))
        self.assertIsNone(book.previous_source)
        self.assertFalse(session.history)

        session.reset()
        reply = session.reply('What did Dorothy carry to the garden?')
        self.assertEqual(reply.passage_trace.query, 'What did Dorothy carry to the garden?')
        self.assertEqual(book.previous_question, 'What did Dorothy carry to the garden?')
        self.assertEqual(book.previous_source, passage)
        self.assertEqual(list(session.history), [('What did Dorothy carry to the garden?', 'a basket')])

    def test_weakly_related_or_equivalent_book_answers_keep_classifier_and_its_history(self):
        tokenizer = ChatTokenizer(chars)
        passages = [
            (Passage('Chapter 2', 30, 34, 'Dorothy met Zeb.'), 'Zeb'),
            (Passage('Chapter 1', 10, 14, 'Dorothy will carry a basket to the garden.'), 'basket'),
        ]
        classifier = Mock(predict=Mock(return_value='A basket.'))
        for passage, text in passages:
            with self.subTest(text=text):
                book = BookSession(Mock(search=Mock(return_value=[(passage, 2.0)])),
                                   Mock(read=Mock(return_value=[AnswerSpan(text, 7)])), trace_passages=True)
                session = ChatSession(Mock(), tokenizer, first_reply_classifier=classifier, book_session=book)

                reply = session.reply('What did Dorothy carry to the garden?')

                self.assertEqual((reply.text, reply.reply_method, reply.source), ('A basket.', 'classifier', None))
                self.assertIsNotNone(reply.passage_trace)
                self.assertIsNone(book.previous_source)
                self.assertEqual(book.previous_answer, 'A basket.')

    def test_book_assistance_uses_cited_history_and_keeps_generator_fallback(self):
        tokenizer = ChatTokenizer(chars)
        passage = Passage('Chapter 1', 1, 4, 'Jim pulled a buggy. The buggy was red.')
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))

        def read(question, passages):
            text = {'What did Jim pull?': 'a buggy', 'What color was the item Jim pulled?': 'red'}.get(question, '')
            return [AnswerSpan(text, 8) for _ in passages]

        book = BookSession(index, Mock(read=Mock(side_effect=read)), trace_passages=True)
        conversations = [(1, {'messages': [
            {'role': 'user', 'content': 'What did Jim pull?'},
            {'role': 'assistant', 'content': 'A buggy.'},
            {'role': 'user', 'content': 'What color was it?'},
            {'role': 'assistant', 'content': 'Red.'},
            {'role': 'user', 'content': 'Where is Zeb?'},
            {'role': 'assistant', 'content': 'At the ranch.'},
        ]})]
        model = Mock()

        def generate(context, *_args, **_kwargs):
            prompt = tokenizer.decode(context[0].tolist())
            self.assertIn('red', prompt)
            return torch.cat((context, torch.tensor([tokenizer.encode('At the ranch.' + END_OF_TURN)])), dim=1)

        model.generate.side_effect = generate
        rows = generate_predictions(model, tokenizer, conversations, 'generated', DEFAULT_MAX_TOKENS,
                                    book_session=book)

        self.assertEqual([row['reply_method'] for row in rows], ['book', 'book', 'generator'])
        self.assertEqual([row['clean_stop'] for row in rows], [None, None, True])
        self.assertTrue(rows[1]['history_used'])
        self.assertEqual(rows[1]['source']['section'], 'Chapter 1')
        self.assertEqual(rows[1]['passage_trace']['previous_answer'], 'a buggy')
        self.assertIsNone(rows[2]['source'])
        self.assertIsNone(rows[2]['passage_trace']['selected_rank'])
        self.assertEqual(book.previous_answer, 'At the ranch.')

    def test_history_modes_use_the_right_reply_and_reset_between_conversations(self):
        tokenizer = ChatTokenizer(chars)
        conversations = [
            (1, {'messages': [
                {'role': 'user', 'content': 'Who is Jim?'},
                {'role': 'assistant', 'content': 'A horse.'},
                {'role': 'user', 'content': 'What is his name?'},
                {'role': 'assistant', 'content': 'Jim.'},
            ]}),
            (3, {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi!'},
            ]}),
        ]
        histories = {
            'generated': format_turn('Who is Jim?', 'A kitten.'),
            'none': '',
            'reference': format_turn('Who is Jim?', 'A horse.'),
        }
        for mode, history in histories.items():
            with self.subTest(history=mode):
                contexts = []
                responses = iter(('A kitten.' + END_OF_TURN, 'JIM!' + END_OF_TURN, 'He'))

                def generate(context, _limit, **options):
                    self.assertFalse(options['sample'])
                    contexts.append(tokenizer.decode(context[0].tolist()))
                    return torch.cat((context, torch.tensor([tokenizer.encode(next(responses))])), dim=1)

                model = Mock()
                model.generate.side_effect = generate
                rows = generate_predictions(model, tokenizer, conversations, mode, DEFAULT_MAX_TOKENS)
                self.assertEqual(contexts, [
                    format_prompt('Who is Jim?'),
                    history + format_prompt('What is his name?'),
                    format_prompt('Hello'),
                ])
                self.assertEqual([(row['conversation'], row['turn']) for row in rows], [(1, 1), (1, 2), (3, 1)])
                self.assertEqual(rows[1]['history_used'], bool(history))
                self.assertFalse(any(row['history_compacted'] for row in rows))
                self.assertEqual([row['correct'] for row in rows], [None, True, None])
                self.assertEqual([row['clean_stop'] for row in rows], [True, True, False])

    def test_long_reply_is_shortened_when_including_the_latest_turn(self):
        tokenizer = ChatTokenizer(chars)
        first_prompt = 'Who is Jim?'
        first_reply = 'Jim is a horse. He pulls the buggy and travels with Dorothy and Zeb.'
        followup = 'What kind of animal?'
        contexts = []
        responses = iter((first_reply, 'A horse.'))

        def generate(context, max_tokens, **_options):
            contexts.append((tokenizer.decode(context[0].tolist()), max_tokens))
            return torch.cat((context, torch.tensor([tokenizer.encode(next(responses) + END_OF_TURN)])), dim=1)

        model = Mock()
        model.generate.side_effect = generate
        session = ChatSession(model, tokenizer)
        self.assertEqual(session.reply(first_prompt).text, first_reply)
        reply = session.reply(followup)

        self.assertTrue(reply.history_used)
        self.assertTrue(reply.history_compacted)
        self.assertEqual(reply.text, 'A horse.')
        prompt, remaining = contexts[1]
        self.assertEqual(prompt, format_turn(first_prompt, 'Jim is a horse. He') + format_prompt(followup))
        self.assertGreaterEqual(remaining, REPLY_RESERVE)
        self.assertEqual(len(prompt) + remaining, block_size + 1)

    def test_reviewed_correctness_is_distinct_from_reference_matches(self):
        self.assertTrue(reference_matches('  THE   Magic Belt! ', 'The Magic Belt.'))
        self.assertFalse(reference_matches('The enchanted belt.', 'The Magic Belt.'))
        self.assertEqual(correctness([{'correct': None}]), 'pending (1 to review)')
        self.assertEqual(correctness([{'correct': True}, {'correct': None}]), '1/1 reviewed (1 pending)')
        self.assertEqual(correctness([{'correct': True}, {'correct': False}]), '1/2')
        with self.assertRaisesRegex(ValueError, 'true, false, or null'):
            correctness([{'correct': 1}])


if __name__ == '__main__':
    unittest.main()
