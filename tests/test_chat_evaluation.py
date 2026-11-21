import unittest
from unittest.mock import Mock

import torch

from model.chat_data import END_OF_TURN, ChatTokenizer, format_prompt, format_turn
from model.chat_evaluation import correctness, generate_predictions, reference_matches
from model.chat_session import DEFAULT_MAX_TOKENS
from model.model import chars


class ChatEvaluationTests(unittest.TestCase):
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
                self.assertEqual([row['correct'] for row in rows], [None, True, None])
                self.assertEqual([row['clean_stop'] for row in rows], [True, True, False])

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
