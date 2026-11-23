from contextlib import redirect_stderr, redirect_stdout
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
    encode_conversations, format_turn, load_chat_checkpoint, load_examples, pack_history, with_history,
)
from model.chat_training import (
    DATA_DIR, DevelopmentMatches, development_score, generation_loss, main as train_chat, save_checkpoint,
)
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
    def test_reply_start_loss_only_upweights_answer_characters(self):
        marker_size = len(END_OF_TURN)
        targets = torch.tensor([
            [IGNORE_TARGET] + [1] * (14 + marker_size),
            [IGNORE_TARGET, 0] + [1] * marker_size + [IGNORE_TARGET] * 13,
        ])
        scores = torch.arange(targets.numel(), dtype=torch.float32).reshape(targets.shape) / 7
        logits = torch.stack((scores, torch.zeros_like(scores)), dim=-1)
        token_losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
        boosted = token_losses[0, 1:13].sum() + token_losses[1, 1]
        expected = (token_losses.sum() + 3 * boosted) / (targets.ne(IGNORE_TARGET).sum() + 3 * 13)

        torch.testing.assert_close(generation_loss(logits, targets, 4), expected)
        torch.testing.assert_close(generation_loss(logits, targets, 1),
                                   token_losses.sum() / targets.ne(IGNORE_TARGET).sum())

    def setUp(self):
        self.tokenizer = ChatTokenizer(chars)

    def load_conversation(self, *turns, tokenizer=None):
        messages = []
        for prompt, reply in turns:
            messages.extend([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': reply},
            ])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jsonl'
            path.write_text(json.dumps({'messages': messages}), encoding='utf-8')
            return load_examples(path, tokenizer or self.tokenizer)

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

    def test_followup_benchmarks_hold_out_first_questions_and_reuse_followup_wording(self):
        training = chat_split('train')
        earlier = training + sum((chat_split(name) for name in (
            'val', 'test', 'paraphrase_dev', 'paraphrase_test', 'unseen_facts_test',
        )), [])
        prompts = normalized_messages(earlier, 'user', first_only=True)
        pairs = {(row['fact_family'], row['messages'][2]['content']) for row in training if 'followup_group' in row}
        for name in ('memory_dev', 'memory_test', 'memory_unseen',
                     'coverage_dev', 'coverage_test', 'coverage_unseen', 'generalization_test'):
            split = chat_split(name)
            questions = normalized_messages(split, 'user', first_only=True)
            self.assertEqual(len(questions), len(split))
            self.assertFalse(prompts & questions)
            prompts.update(questions)
            load_examples(DATA_DIR / f'oz_chat_{name}.jsonl', ChatTokenizer(chars, compact_history=True))
            if name.endswith('_unseen') or name == 'generalization_test':
                self.assertFalse(normalized_messages(earlier, 'assistant') & normalized_messages(split, 'assistant'))
            else:
                by_question = {}
                for row in split:
                    if 'followup_group' not in row:
                        continue
                    followup, answer = (message['content'] for message in row['messages'][2:])
                    self.assertIn((row['fact_family'], followup), pairs)
                    by_question.setdefault(followup, set()).add(answer)
                self.assertTrue(by_question and all(len(answers) > 1 for answers in by_question.values()))

    def test_balanced_selection_prioritizes_the_weakest_first_reply_score(self):
        uneven = [DevelopmentMatches(8, 10, 14, 20), DevelopmentMatches(2, 10, 9, 20)]
        balanced = [DevelopmentMatches(4, 10, 8, 20), DevelopmentMatches(4, 10, 8, 20)]
        stronger = [DevelopmentMatches(5, 10, 8, 20), DevelopmentMatches(4, 10, 7, 20)]
        self.assertGreater(development_score(uneven, 'matches'), development_score(balanced, 'matches'))
        self.assertGreater(development_score(balanced, 'balanced-first'), development_score(uneven, 'balanced-first'))
        self.assertGreater(development_score(stronger, 'balanced-first'), development_score(balanced, 'balanced-first'))

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

    def test_compact_history_preserves_both_parts_when_the_legacy_turn_will_not_fit(self):
        first = 'What did Zeb use when he climbed down after the wooden wings?'
        answer = 'A strap from the harness on the buggy.'
        followup = 'What was it made of?'
        current = format_prompt(followup)
        tokenizer = ChatTokenizer(chars, compact_history=True)
        _, (inputs, targets) = self.load_conversation((first, answer), (followup, 'Leather.'), tokenizer=tokenizer)
        reply_start = next(index for index, target in enumerate(targets) if target != IGNORE_TARGET)
        prompt = tokenizer.decode(inputs[:reply_start + 1])
        self.assertIn('Q: What did Zeb', prompt)
        self.assertIn('\nA: A strap', prompt)
        self.assertTrue(prompt.endswith(current))
        self.assertEqual(pack_history(current, [(first, answer)], 97), (current, False))
        self.assertEqual(pack_history(current, [(first, answer)], len(current), compact_history=True), (current, False))

        with patch.object(GPTLanguageModel, 'generate') as generate:
            generate.side_effect = lambda context, *_args, **_kwargs: torch.cat(
                (context, torch.tensor([tokenizer.encode('Leather.' + END_OF_TURN)])), dim=1,
            )
            session = ChatSession(GPTLanguageModel(len(chars)), tokenizer)
            session.remember(first, answer)
            reply = session.reply(followup)
        self.assertTrue(reply.history_used)
        self.assertTrue(reply.history_compacted)
        self.assertEqual(tokenizer.decode(generate.call_args.args[0][0].tolist()), prompt)

    def test_hiding_history_replies_preserves_all_reply_targets(self):
        conversation = [(1, {'messages': [
            {'role': 'user', 'content': 'What did Zeb wear?'},
            {'role': 'assistant', 'content': 'A velvet costume.'},
            {'role': 'user', 'content': 'What color was it?'},
            {'role': 'assistant', 'content': 'Maroon.'},
        ]})]
        tokenizer = ChatTokenizer(chars, compact_history=True)
        unmasked = encode_conversations(conversation, tokenizer, 'memory')
        masked = encode_conversations(conversation, tokenizer, 'memory', mask_history_replies=True)
        self.assertEqual(unmasked[0], masked[0])
        inputs, targets = masked[1]
        self.assertIn('\nA: ?\n', tokenizer.decode(inputs))
        self.assertEqual(tokenizer.decode([target for target in targets if target != IGNORE_TARGET]),
                         'Maroon.' + END_OF_TURN)

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
        model = GPTLanguageModel(len(chars), heads=4, layers=2)
        for compact in (False, True):
            with self.subTest(compact=compact), TemporaryDirectory() as directory:
                tokenizer = ChatTokenizer(list(reversed(chars)), compact_history=compact)
                path = Path(directory) / 'chat.pth'
                save_checkpoint(path, model, tokenizer)
                weights, loaded_tokenizer = load_chat_checkpoint(path)
                self.assertEqual(loaded_tokenizer.characters, tokenizer.characters)
                self.assertEqual(loaded_tokenizer.encode('Hello'), tokenizer.encode('Hello'))
                self.assertEqual(loaded_tokenizer.compact_history, compact)
                self.assertTrue(torch.equal(weights['lm_head.weight'], model.lm_head.weight))

    def test_training_from_chat_preserves_source_checkpoint_and_history_format(self):
        tokenizer = ChatTokenizer(list(reversed(chars)), compact_history=True)
        model = GPTLanguageModel(len(chars))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source.pth', root / 'output.pth'
            training, validation = root / 'train.jsonl', root / 'val.jsonl'
            save_checkpoint(source, model, tokenizer)
            original = source.read_bytes()
            for file, question, reply in ((training, 'Who is Jim?', 'A horse.'),
                                          (validation, 'Who is Eureka?', 'A kitten.')):
                file.write_text(json.dumps({'messages': [
                    {'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply},
                ]}), encoding='utf-8')
            args = ['chat_training', '--init-chat-checkpoint', str(source), '--train-data', str(training),
                    '--val-data', str(validation), '--device', 'cpu', '--steps', '1', '--batch-size', '1']
            with patch('sys.argv', args + ['--output', str(source)]), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    train_chat()
            self.assertEqual(exit_status.exception.code, 2)
            with patch('sys.argv', args + ['--output', str(output)]), redirect_stdout(StringIO()):
                train_chat()
            self.assertEqual(source.read_bytes(), original)
            weights, saved = load_chat_checkpoint(output)
            self.assertTrue(saved.compact_history)
            self.assertEqual(saved.characters, tokenizer.characters)
            self.assertFalse(torch.equal(weights['lm_head.weight'], model.lm_head.weight))

    def test_generation_only_stops_on_a_newly_generated_marker(self):
        model = GPTLanguageModel(4)
        prompt = torch.tensor([[1]])
        scores = [torch.nn.functional.one_hot(torch.tensor([[token]]), 4).float().log() for token in (2, 3, 1, 2)]
        with patch.object(model, 'forward', side_effect=[(logits, None) for logits in scores]):
            generated = model.generate(prompt, 6, stop_ids=(1, 2))
        self.assertEqual(generated.tolist(), [[1, 2, 3, 1, 2]])


if __name__ == '__main__':
    unittest.main()
