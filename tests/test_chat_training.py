from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from model.chat_data import (
    END_OF_TURN, IGNORE_TARGET, ChatTokenizer, collate_examples, format_prompt,
    format_turn, load_chat_checkpoint, load_examples, with_history,
)
from model.chat_training import DATA_DIR, save_checkpoint
from model.chat_session import ChatSession
from model.model import GPTLanguageModel, block_size, chars


def chat_split(split):
    path = DATA_DIR / f'oz_chat_{split}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def normalized_messages(conversations, role, *, first_only=False):
    return {
        ' '.join(re.findall(r'[a-z0-9]+', message['content'].casefold()))
        for conversation in conversations
        for message in (conversation['messages'][:1] if first_only else conversation['messages'])
        if message['role'] == role
    }


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

    def test_default_validation_replies_are_disjoint_from_training(self):
        self.assertFalse(normalized_messages(chat_split('train'), 'assistant') &
                         normalized_messages(chat_split('val'), 'assistant'))

    def test_fact_benchmarks_keep_novel_questions_and_unseen_answers_separate(self):
        training = chat_split('train')
        development = chat_split('paraphrase_dev')
        paraphrases = chat_split('paraphrase_test')
        unseen = chat_split('unseen_facts_test')
        earlier = training + chat_split('val') + chat_split('test')
        families = {conversation.get('fact_family') for conversation in training}
        self.assertTrue({row['fact_family'] for row in development + paraphrases} <= families)
        self.assertFalse({row['fact_family'] for row in unseen} & families)
        self.assertFalse(normalized_messages(earlier + development + paraphrases, 'assistant') &
                         normalized_messages(unseen, 'assistant'))

        prompts = normalized_messages(earlier, 'user', first_only=True)
        for split in (development, paraphrases, unseen):
            questions = normalized_messages(split, 'user', first_only=True)
            self.assertEqual(len(questions), len(split))
            self.assertFalse(questions & prompts)
            prompts.update(questions)

        for name in ('train', 'paraphrase_dev', 'paraphrase_test', 'unseen_facts_test'):
            load_examples(DATA_DIR / f'oz_chat_{name}.jsonl', self.tokenizer)

    def test_context_trimming_keeps_current_turn_and_rejects_oversize_turns(self):
        _, second = self.load_conversation(('Hi', 'a' * 70), ('Hello', 'b' * 50))
        self.assertLessEqual(len(second[0]), block_size)
        current_turn = (format_prompt('Hello') + 'b' * 50 + END_OF_TURN)[:-1]
        self.assertTrue(self.tokenizer.decode(second[0]).endswith(current_turn))
        with self.assertRaisesRegex(ValueError, r'chat.jsonl:1: Turn 1 exceeds'):
            self.load_conversation(('Hi', 'a' * block_size))

    def test_history_keeps_only_consecutive_recent_turns(self):
        first = format_turn('Hi', 'Hello!')
        second = format_turn('Who is Jim?', 'Jim is the cab-horse.')
        prompt = format_prompt('What kind?')
        self.assertEqual(with_history(prompt, [first, second], len(first + second + prompt)), first + second + prompt)
        self.assertEqual(with_history(prompt, [first, second], len(second + prompt)), second + prompt)
        self.assertEqual(with_history(prompt, [first, second], len(second + prompt) - 1), prompt)

    def test_training_and_chat_pack_the_same_shortened_history(self):
        first = 'Who is Jim?'
        answer = 'Jim is a horse. He pulls the buggy and travels with Dorothy and Zeb.'
        follow_up = 'What kind of animal?'
        reply = 'A horse.' + END_OF_TURN
        _, (inputs, targets) = self.load_conversation((first, answer), (follow_up, 'A horse.'))
        prompt = self.tokenizer.decode(inputs)[:-len(reply) + 1]
        self.assertEqual(prompt, format_turn(first, 'Jim is a horse. He') + format_prompt(follow_up))

        with patch.object(GPTLanguageModel, 'generate') as generate:
            generate.side_effect = lambda context, *_args, **_kw: torch.cat(
                (context, torch.tensor([self.tokenizer.encode(reply)])), dim=1,
            )
            session = ChatSession(GPTLanguageModel(len(chars)), self.tokenizer)
            session.remember(first, answer)
            self.assertTrue(session.reply(follow_up).history_compacted)
        self.assertEqual(self.tokenizer.decode(generate.call_args.args[0][0].tolist()), prompt)
        self.assertEqual(self.tokenizer.decode([token for token in targets if token != IGNORE_TARGET]), reply)

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
                patch('model.model.model_from_state') as model_from_state,
                patch('builtins.input', side_effect=['Who is Jim?', 'What kind?', '/reset', 'What kind?', 'quit']),
                redirect_stdout(StringIO()) as stdout,
            ):
                model_from_state.return_value.generate.side_effect = generate
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
        model = GPTLanguageModel(len(chars), heads=4, layers=2)
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
