from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from model.chat_data import (
    END_OF_TURN, IGNORE_TARGET, ChatTokenizer, collate_examples, format_prompt,
    format_turn, load_chat_checkpoint, load_examples, with_history,
)
from model.chat_training import save_checkpoint
from model.model import GPTLanguageModel, block_size, chars


class ChatTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ChatTokenizer(chars)

    def load_conversation(self, *turns):
        messages = []
        for prompt, reply in turns:
            messages.extend([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': reply},
            ])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jsonl'
            path.write_text(json.dumps({'messages': messages}), encoding='utf-8')
            return load_examples(path, self.tokenizer)

    def test_only_current_reply_and_stop_marker_are_targets(self):
        first, second = self.load_conversation(('Hi', 'Hello!'), ('Who is Jim?', 'A horse.'))
        inputs, targets = second
        prompt = format_prompt('Hi') + 'Hello!' + END_OF_TURN + '\n' + format_prompt('Who is Jim?')
        reply = 'A horse.' + END_OF_TURN
        self.assertEqual(self.tokenizer.decode(inputs), (prompt + reply)[:-1])
        self.assertEqual(targets[:len(prompt) - 1], [IGNORE_TARGET] * (len(prompt) - 1))
        self.assertEqual(self.tokenizer.decode(targets[len(prompt) - 1:]), reply)

        _, padded_targets = collate_examples([first, second])
        self.assertTrue(padded_targets[0, len(first[1]):].eq(IGNORE_TARGET).all())

    def test_context_trimming_keeps_current_turn_and_rejects_oversize_turns(self):
        _, second = self.load_conversation(('Hi', 'a' * 70), ('Hello', 'b' * 50))
        self.assertEqual(self.tokenizer.decode(second[0]), (format_prompt('Hello') + 'b' * 50 + END_OF_TURN)[:-1])
        with self.assertRaisesRegex(ValueError, r'chat.jsonl:1: Turn 1 exceeds'):
            self.load_conversation(('Hi', 'a' * block_size))

    def test_history_keeps_only_consecutive_recent_turns(self):
        first = format_turn('Hi', 'Hello!')
        second = format_turn('Who is Jim?', 'Jim is the cab-horse.')
        prompt = format_prompt('What kind?')
        self.assertEqual(with_history(prompt, [first, second], len(first + second + prompt)), first + second + prompt)
        self.assertEqual(with_history(prompt, [first, second], len(second + prompt)), second + prompt)
        self.assertEqual(with_history(prompt, [first, second], len(second + prompt) - 1), prompt)

    def test_cli_uses_history_and_resets_it(self):
        replies = ('Jim is the cab-horse.' + END_OF_TURN, 'A horse.' + END_OF_TURN, 'Ask me about Oz.' + END_OF_TURN)
        seen_prompts = []

        def generate(context, max_new_tokens, **kwargs):
            seen_prompts.append(self.tokenizer.decode(context[0].tolist()))
            reply = replies[len(seen_prompts) - 1]
            self.assertLessEqual(context.size(1) + max_new_tokens, block_size + 1)
            return torch.cat((context, torch.tensor([self.tokenizer.encode(reply)])), dim=1)

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.pth'
            path.touch()
            with (
                patch('sys.argv', ['chatbot', '--chat', str(path)]),
                patch('model.chat_data.load_chat_checkpoint', return_value=({}, self.tokenizer)),
                patch('model.model.get_device', return_value='cpu'),
                patch('model.model.GPTLanguageModel') as model_class,
                patch('builtins.input', side_effect=['Who is Jim?', 'What kind?', '/reset', 'What kind?', 'quit']),
                redirect_stdout(StringIO()) as stdout,
            ):
                model_class.return_value.generate.side_effect = generate
                runpy.run_module('model.chatbot', run_name='__main__')

        follow_up = format_prompt('What kind?')
        self.assertEqual(seen_prompts, [
            format_prompt('Who is Jim?'),
            format_turn('Who is Jim?', 'Jim is the cab-horse.') + follow_up,
            follow_up,
        ])
        self.assertIn('Assistant: A horse.', stdout.getvalue())
        self.assertNotIn(END_OF_TURN.strip(), stdout.getvalue())

    def test_checkpoint_restores_token_order(self):
        tokenizer = ChatTokenizer(list(reversed(chars)))
        model = GPTLanguageModel(len(chars))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.pth'
            save_checkpoint(path, model, tokenizer)
            weights, loaded_tokenizer = load_chat_checkpoint(path)
        self.assertEqual(loaded_tokenizer.characters, tokenizer.characters)
        self.assertEqual(loaded_tokenizer.encode('Hello'), tokenizer.encode('Hello'))
        self.assertTrue(torch.equal(weights['lm_head.weight'], model.lm_head.weight))

    def test_generation_only_stops_on_a_newly_generated_marker(self):
        model = GPTLanguageModel(4)
        prompt = torch.tensor([[1]])
        scores = [torch.nn.functional.one_hot(torch.tensor([[token]]), 4).float().log() for token in (2, 3, 1, 2)]
        with patch.object(model, 'forward', side_effect=[(logits, None) for logits in scores]):
            generated = model.generate(prompt, 6, stop_ids=(1, 2))
        self.assertEqual(generated.tolist(), [[1, 2, 3, 1, 2]])


if __name__ == '__main__':
    unittest.main()
