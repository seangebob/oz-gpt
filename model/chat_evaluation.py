import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle

import torch

from model.chat_data import (
    CHAT_CHECKPOINT, ChatTokenizer, encode_conversations, load_chat_checkpoint, read_conversations,
)
from model.chat_session import ChatSession, DEFAULT_MAX_TOKENS, HISTORY_PACKING, REPLY_RESERVE
from model.chat_training import DATA_DIR, evaluate
from model.model import GPTLanguageModel, MODEL_PATH, block_size, chars


HISTORY_MODES = ('generated', 'none', 'reference')
REPORT_VERSION = 1


def reference_matches(reply, expected):
    reply_key = ' '.join(reply.casefold().split()).rstrip('.!?').rstrip()
    expected_key = ' '.join(expected.casefold().split()).rstrip('.!?').rstrip()
    return bool(expected_key) and reply_key == expected_key


def generate_predictions(model, tokenizer, conversations, history_mode, max_tokens):
    session = ChatSession(model, tokenizer, max_tokens=max_tokens)
    predictions = []
    for line_number, conversation in conversations:
        session.reset()
        messages = conversation['messages']
        for index in range(0, len(messages), 2):
            prompt = messages[index]['content'].strip()
            expected = messages[index + 1]['content'].strip()
            try:
                reply = session.reply(prompt, remember=history_mode == 'generated')
            except ValueError as error:
                raise ValueError(f'Conversation {line_number}, turn {index // 2 + 1}: {error}') from error
            if history_mode == 'reference':
                session.remember(prompt, expected)
            matched = reference_matches(reply.text, expected)
            predictions.append({
                'conversation': line_number,
                'turn': index // 2 + 1,
                'prompt': prompt,
                'expected': expected,
                'reply': reply.text,
                'history_used': reply.history_used,
                'history_compacted': reply.history_compacted,
                'ended_with_marker': reply.ended_with_marker,
                'clean_stop': reply.ended_with_marker and 'END]' not in reply.text,
                'reference_match': matched,
                'correct': True if matched else None,
            })
    return predictions


def correctness(predictions):
    judgments = [row['correct'] for row in predictions]
    if any(value is not None and type(value) is not bool for value in judgments):
        raise ValueError('Each correct field must be true, false, or null.')
    correct = sum(value is True for value in judgments)
    reviewed = sum(value is not None for value in judgments)
    total = len(judgments)
    if reviewed == total:
        return f'{correct}/{total}' if total else 'n/a'
    if not reviewed:
        return f'pending ({total} to review)'
    return f'{correct}/{reviewed} reviewed ({total - reviewed} pending)'


def print_summary(path, report):
    print(f'{path} | {report["conversations"]} conversations | test SHA-256 {report["test_sha256"][:12]}')
    print(f'  Context: {report["block_size"]} | max new tokens: {report["max_tokens"]} '
          f'| reply reserve: {report["reply_reserve"]}')
    packing = report.get('history_packing', 'complete_turns').replace('_', ' ')
    print(f'  History packing:      {packing}')
    for run in report['runs']:
        rows = run['predictions']
        first = [row for row in rows if row['turn'] == 1]
        followups = [row for row in rows if row['turn'] > 1]
        print(f'\n{run["checkpoint"]} ({run["checkpoint_sha256"][:12]}) | history: {run["history"]}')
        print(f'  Correct first replies: {correctness(first)}')
        print(f'  Correct follow-ups:    {correctness(followups)}')
        print(f'  Reference matches:     {sum(row["reference_match"] for row in first)}/{len(first)} first replies, '
              f'{sum(row["reference_match"] for row in followups)}/{len(followups)} follow-ups')
        print(f'  Follow-ups with history: {sum(row["history_used"] for row in followups)}/{len(followups)}')
        if followups and 'history_compacted' in followups[0]:
            print(f'  Compacted history:     {sum(row["history_compacted"] for row in followups)}/{len(followups)} follow-ups')
        print(f'  Clean stops:           {sum(row["clean_stop"] for row in rows)}/{len(rows)}')
        print(f'  Reference reply loss:  {run["reference_reply_loss"]:.4f} (lower is better)')


def run_evaluation(args):
    if args.max_tokens < 1:
        raise ValueError('--max-tokens must be positive.')
    if args.output.exists():
        raise FileExistsError(f'Report already exists: {args.output}')
    checkpoints = args.checkpoint or [CHAT_CHECKPOINT]
    history_modes = list(dict.fromkeys(args.history or ['generated']))
    test_bytes = args.test_data.read_bytes()
    with io.StringIO(test_bytes.decode('utf-8'), newline=None) as source:
        conversations = list(read_conversations(source, args.test_data))
    report = {
        'schema_version': REPORT_VERSION,
        'test_file': str(args.test_data),
        'test_sha256': hashlib.sha256(test_bytes).hexdigest(),
        'conversations': len(conversations),
        'device': 'cpu',
        'torch_version': str(torch.__version__),
        'cpu_threads': torch.get_num_threads(),
        'decoding': 'greedy',
        'block_size': block_size,
        'max_tokens': args.max_tokens,
        'reply_reserve': REPLY_RESERVE,
        'history_packing': HISTORY_PACKING,
        'runs': [],
    }
    for path in checkpoints:
        checkpoint_bytes = path.read_bytes()
        source = io.BytesIO(checkpoint_bytes)
        if path.resolve() == Path(MODEL_PATH).resolve():
            state = torch.load(source, map_location='cpu', weights_only=True)
            tokenizer = ChatTokenizer(chars)
        else:
            state, tokenizer = load_chat_checkpoint(source)
        model = GPTLanguageModel(len(tokenizer.characters))
        model.load_state_dict(state)
        examples = encode_conversations(conversations, tokenizer, args.test_data)
        loss = evaluate(model, examples, batch_size=8, device='cpu')
        for history in history_modes:
            report['runs'].append({
                'checkpoint': str(path),
                'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
                'history': history,
                'reference_reply_loss': round(loss, 6),
                'predictions': generate_predictions(model, tokenizer, conversations, history, args.max_tokens),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print_summary(args.output, report)


def summarize(paths):
    reports = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in paths]
    for path, report in reports:
        if not isinstance(report, dict) or report.get('schema_version') != REPORT_VERSION:
            raise ValueError(f'{path} is not an evaluation report from this command.')
    if len({(report['test_sha256'], report['max_tokens']) for _, report in reports}) > 1:
        raise ValueError('Compare reports with the same test dataset and maximum reply length.')
    for path, report in reports:
        print_summary(path, report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Oz chat checkpoints on held-out conversations.')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Generate replies and save an evaluation report.')
    run.add_argument('--checkpoint', action='append', type=Path, help='Checkpoint to evaluate. Repeat to compare checkpoints.')
    run.add_argument('--test-data', type=Path, default=DATA_DIR / 'oz_chat_test.jsonl', help='Held-out JSONL conversations.')
    run.add_argument('--history', action='append', choices=HISTORY_MODES, help='History mode. Repeat to compare modes (default: generated).')
    run.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS, help='Maximum new tokens per reply.')
    run.add_argument('--output', type=Path, required=True, help='New JSON report path.')
    review = commands.add_parser('summarize', help='Print scores from one or more saved reports.')
    review.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()

    try:
        if args.command == 'run':
            run_evaluation(args)
        else:
            summarize(args.reports)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        parser.exit(1, f'Evaluation error: {error}\n')


if __name__ == '__main__':
    main()
