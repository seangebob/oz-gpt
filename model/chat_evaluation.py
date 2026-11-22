import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle

import torch

from model.book_lookup import BOOK_PATH, READER_MODEL, READER_REVISION, BookReply, BookSession
from model.chat_data import (
    CHAT_CHECKPOINT, ChatTokenizer, conversation_turns, encode_conversations, load_chat_checkpoint, read_conversations,
    reference_matches,
)
from model.chat_session import ChatSession, DEFAULT_MAX_TOKENS, REPLY_RESERVE
from model.chat_training import DATA_DIR, evaluate
from model.first_reply import FIRST_REPLY_CHECKPOINT, load_first_reply_classifier
from model.model import MODEL_PATH, RETRAINED_BASE_PATH, block_size, chars, model_from_state


HISTORY_MODES = ('generated', 'none', 'reference')
REPORT_VERSION = 1


def generate_predictions(model, tokenizer, conversations, history_mode, max_tokens, *, first_reply_classifier=None):
    session = ChatSession(model, tokenizer, max_tokens=max_tokens, first_reply_classifier=first_reply_classifier)
    return generate_session_predictions(session, conversations, history_mode)


def passage_record(passage):
    return {
        'section': passage.section,
        'start_line': passage.start_line,
        'end_line': passage.end_line,
        'text': passage.text,
        'heading': passage.heading,
    }


def passage_trace_record(reply):
    trace = reply.passage_trace
    candidates = []
    selected_rank = None
    for rank, (passage, score) in enumerate(trace.candidates, start=1):
        row = {'rank': rank, 'search_score': round(score, 6), **passage_record(passage)}
        if passage == reply.source:
            selected_rank = rank
        if rank <= len(trace.spans):
            span = trace.spans[rank - 1]
            row.update(reader_span=span.text, reader_margin=round(span.margin, 6))
            if span.context_end is not None and span.context_end < len(passage.text):
                row['reader_text'] = passage.text[:span.context_end]
        candidates.append(row)
    return {
        'query': trace.query,
        'previous_question': trace.previous_question,
        'previous_answer': trace.previous_answer,
        'selected_rank': selected_rank,
        'evidence_ranks': None,
        'review_note': '',
        'candidates': candidates,
    }


def generate_session_predictions(session, conversations, history_mode):
    predictions = []
    for line_number, conversation in conversations:
        session.reset()
        try:
            for turn, (prompt, expected) in enumerate(conversation_turns(conversation), start=1):
                reply = session.reply(prompt, remember=history_mode == 'generated')
                if history_mode == 'reference':
                    session.remember(prompt, expected)
                matched = reference_matches(reply.text, expected)
                row = {
                    'conversation': line_number,
                    'turn': turn,
                    'prompt': prompt,
                    'expected': expected,
                    'reply': reply.text,
                    'history_used': reply.history_used,
                    'reference_match': matched,
                    'correct': True if matched else None,
                }
                if isinstance(reply, BookReply):
                    row['source'] = passage_record(reply.source) if reply.source else None
                    row['source_supports_reply'] = None if reply.source else False
                    if reply.passage_trace is not None:
                        row['passage_trace'] = passage_trace_record(reply)
                else:
                    row.update(
                        reply_method=reply.reply_method,
                        history_compacted=reply.history_compacted,
                        ended_with_marker=reply.ended_with_marker,
                        clean_stop=(reply.ended_with_marker and 'END]' not in reply.text
                                    if reply.reply_method == 'generator' else None),
                    )
                predictions.append(row)
        except ValueError as error:
            raise ValueError(f'Conversation {line_number}: {error}') from error
    return predictions


def reviewed_score(judgments):
    if any(value is not None and type(value) is not bool for value in judgments):
        raise ValueError('Each review field must be true, false, or null.')
    correct = sum(value is True for value in judgments)
    reviewed = sum(value is not None for value in judgments)
    total = len(judgments)
    if reviewed == total:
        return f'{correct}/{total}' if total else 'n/a'
    if not reviewed:
        return f'pending ({total} to review)'
    return f'{correct}/{reviewed} reviewed ({total - reviewed} pending)'


def correctness(predictions):
    return reviewed_score([row['correct'] for row in predictions])


def grounded_correctness(predictions):
    judgments = []
    for row in predictions:
        support = row.get('source_supports_reply')
        if support is not None and type(support) is not bool:
            raise ValueError('Each source_supports_reply field must be true, false, or null.')
        correct = row['correct']
        if correct is False or support is False:
            judgments.append(False)
        elif correct is True and support is True:
            judgments.append(True)
        else:
            judgments.append(None)
    return reviewed_score(judgments)


def passage_review_counts(predictions):
    found = missing = pending = 0
    for row in predictions:
        trace = row.get('passage_trace')
        if trace is None:
            continue
        ranks = trace['evidence_ranks']
        if ranks is None:
            pending += 1
            continue
        available = {candidate['rank'] for candidate in trace['candidates']}
        if not isinstance(ranks, list) or any(type(rank) is not int or rank not in available for rank in ranks):
            raise ValueError(f'Conversation {row["conversation"]}, turn {row["turn"]}: '
                             'evidence_ranks must contain retrieved passage ranks.')
        if ranks:
            found += 1
        else:
            missing += 1
    return found, missing, pending


def print_summary(path, report):
    print(f'{path} | {report["conversations"]} conversations | test SHA-256 {report["test_sha256"][:12]}')
    if any(run.get('mode') != 'book' for run in report['runs']):
        print(f'  Chat context: {report["block_size"]} | max new tokens: {report["max_tokens"]} '
              f'| reply reserve: {report["reply_reserve"]}')
    for run in report['runs']:
        rows = run['predictions']
        first = [row for row in rows if row['turn'] == 1]
        followups = [row for row in rows if row['turn'] > 1]
        book_mode = run.get('mode') == 'book'
        if book_mode:
            print(f'\nBook lookup ({run["reader_model"]}) | history: {run["history"]}')
        else:
            print(f'\n{run["checkpoint"]} ({run["checkpoint_sha256"][:12]}) | history: {run["history"]}')
            packing = run.get('history_packing', report.get('history_packing', 'complete_turns')).replace('_', ' ')
            print(f'  History packing:      {packing}')
        print(f'  Correct first replies: {correctness(first)}')
        print(f'  Correct follow-ups:    {correctness(followups)}')
        print(f'  Reference matches:     {sum(row["reference_match"] for row in first)}/{len(first)} first replies, '
              f'{sum(row["reference_match"] for row in followups)}/{len(followups)} follow-ups')
        print(f'  Follow-ups with history: {sum(row["history_used"] for row in followups)}/{len(followups)}')
        if book_mode:
            sourced = sum(row['source'] is not None for row in rows)
            print(f'  Correct first replies with support: {grounded_correctness(first)}')
            print(f'  Correct follow-ups with support:    {grounded_correctness(followups)}')
            print(f'  Replies with a source:  {sourced}/{len(rows)}')
            print(f'  Abstentions:           {len(rows) - sourced}/{len(rows)}')
            found, missing, pending = passage_review_counts(rows)
            if found or missing or pending:
                print(f'  Passage reviews:        {found} with evidence, {missing} without evidence, {pending} pending')
        else:
            if followups and 'history_compacted' in followups[0]:
                print(f'  Compacted history:     {sum(row["history_compacted"] for row in followups)}/{len(followups)} follow-ups')
            generated = [row for row in rows if row['clean_stop'] is not None]
            if run.get('first_reply_classifier'):
                print(f'  Classifier first replies: {len(rows) - len(generated)}/{len(first)}')
            print(f'  Clean stops:           {sum(row["clean_stop"] for row in generated)}/{len(generated)}')
            print(f'  Reference reply loss:  {run["reference_reply_loss"]:.4f} (lower is better)')


def run_evaluation(args):
    if args.trace_passages and not args.book:
        raise ValueError('--trace-passages requires --book.')
    if args.max_tokens < 1:
        raise ValueError('--max-tokens must be positive.')
    if args.output.exists():
        raise FileExistsError(f'Report already exists: {args.output}')
    checkpoints = args.checkpoint or ([] if args.book else [CHAT_CHECKPOINT])
    classifier = None
    classifier_metadata = {}
    if args.first_reply_classifier:
        if not checkpoints:
            raise ValueError('--first-reply-classifier requires a chat checkpoint when used with --book.')
        classifier_bytes = args.first_reply_classifier.read_bytes()
        classifier = load_first_reply_classifier(io.BytesIO(classifier_bytes))
        classifier_metadata = {
            'first_reply_classifier': str(args.first_reply_classifier),
            'first_reply_classifier_sha256': hashlib.sha256(classifier_bytes).hexdigest(),
        }
    history_modes = list(dict.fromkeys(args.history or ['generated']))
    test_bytes = args.test_data.read_bytes()
    with io.StringIO(test_bytes.decode('utf-8'), newline=None) as source:
        conversations = list(read_conversations(source, args.test_data))
    if not conversations:
        raise ValueError(f'{args.test_data} contains no usable conversations.')
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
        'runs': [],
    }
    for path in checkpoints:
        checkpoint_bytes = path.read_bytes()
        source = io.BytesIO(checkpoint_bytes)
        if path.resolve() in {Path(MODEL_PATH).resolve(), Path(RETRAINED_BASE_PATH).resolve()}:
            state = torch.load(source, map_location='cpu', weights_only=True)
            tokenizer = ChatTokenizer(chars)
        else:
            state, tokenizer = load_chat_checkpoint(source)
        model = model_from_state(state, len(tokenizer.characters))
        examples = encode_conversations(conversations, tokenizer, args.test_data)
        loss = evaluate(model, examples, batch_size=8, device='cpu')
        for history in history_modes:
            report['runs'].append({
                'checkpoint': str(path),
                'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
                'history': history,
                'history_packing': tokenizer.history_packing,
                'reference_reply_loss': round(loss, 6),
                **classifier_metadata,
                'predictions': generate_predictions(model, tokenizer, conversations, history, args.max_tokens,
                                                    first_reply_classifier=classifier),
            })
    if args.book:
        session = BookSession.from_book(trace_passages=args.trace_passages)
        for history in history_modes:
            report['runs'].append({
                'mode': 'book',
                'book': BOOK_PATH.name,
                'book_sha256': session.book_sha256,
                'reader_model': READER_MODEL,
                'reader_revision': READER_REVISION,
                'history': history,
                'predictions': generate_session_predictions(session, conversations, history),
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
    run.add_argument('--first-reply-classifier', nargs='?', const=FIRST_REPLY_CHECKPOINT, type=Path,
                     metavar='CHECKPOINT', help='Use a trained classifier for high-scoring first replies')
    run.add_argument('--book', action='store_true', help='Evaluate book lookup. Combine with --checkpoint for a comparison.')
    run.add_argument('--trace-passages', action='store_true', help='Save ranked passages and reader spans for book lookup.')
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
