import argparse
from collections import Counter
from math import isfinite
from pathlib import Path
from pickle import UnpicklingError
import random

import torch
from torch.nn import functional as F

from model.chat_data import (
    CHAT_TRAIN_OUTPUT, CHAT_FORMAT_VERSION, COMPACT_CHAT_FORMAT_VERSION, IGNORE_TARGET, MODEL_CONFIG,
    ChatTokenizer, collate_examples, conversation_turns, encode_conversations, load_chat_checkpoint, load_examples,
    read_conversations, reference_matches,
)
from model.chat_session import ChatSession
from model.model import (
    RETRAINED_BASE_PATH, block_size, chars, get_device, load_base_model, model_dimensions, model_from_state, n_embd,
)
from model.training import save_checkpoint as save_new_checkpoint


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
ANSWER_CLASS_WEIGHT = 0.5


@torch.no_grad()
def evaluate(model, examples, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(examples), batch_size):
        inputs, targets = collate_examples(examples[start:start + batch_size])
        num_tokens = targets.ne(IGNORE_TARGET).sum().item()
        _, loss = model(inputs.to(device), targets.to(device))
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    return total_loss / total_tokens


def save_checkpoint(path, model, tokenizer):
    weights = model.state_dict()
    heads, layers = model_dimensions(weights)
    save_new_checkpoint(path, {
        'chat_format': COMPACT_CHAT_FORMAT_VERSION if tokenizer.compact_history else CHAT_FORMAT_VERSION,
        'model_config': {**MODEL_CONFIG, 'n_head': heads, 'n_layer': layers},
        'characters': list(tokenizer.characters),
        'model_state_dict': weights,
    })


def reply_tokens(targets):
    return tuple(target for target in targets if target != IGNORE_TARGET)


def reply_sampling_weights(examples):
    replies = [reply_tokens(targets) for _, targets in examples]
    counts = Counter(replies)
    return [1 / counts[reply] for reply in replies]


def pad_prompt(example, newline_id, rng):
    inputs, targets = example
    padding = rng.randint(0, min(14, block_size - len(inputs))) if rng.random() < 0.65 else 0
    return [newline_id] * padding + inputs, [IGNORE_TARGET] * padding + targets


def answer_class_loss(model, classifier, inputs, targets, classes):
    hidden = model.hidden_states(inputs)
    logits = model.lm_head(hidden)
    reply_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    # The last prompt character can see the question and history, but no answer tokens.
    prompt_end = targets.eq(IGNORE_TARGET).cumprod(1).sum(1)
    question_states = hidden[torch.arange(inputs.size(0), device=inputs.device), prompt_end]
    class_loss = F.cross_entropy(classifier(question_states), classes)
    return reply_loss + ANSWER_CLASS_WEIGHT * class_loss, reply_loss


def development_matches(model, tokenizer, conversations, device):
    model.eval()
    session = ChatSession(model, tokenizer, device=device)
    matches = total = 0
    for _, conversation in conversations:
        session.reset()
        for prompt, expected in conversation_turns(conversation):
            matches += reference_matches(session.reply(prompt).text, expected)
            total += 1
    return matches, total


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the Oz model to answer user messages.')
    parser.add_argument('--train-data', type=Path, default=DATA_DIR / 'oz_chat_train.jsonl')
    parser.add_argument('--val-data', type=Path, default=DATA_DIR / 'oz_chat_val.jsonl')
    parser.add_argument('--dev-data', type=Path,
                        help='Select by generated reference matches on this development file, then by validation loss')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--base-checkpoint', type=Path, default=Path(RETRAINED_BASE_PATH))
    initialization.add_argument('--init-chat-checkpoint', type=Path,
                                help='Start from previously trained chat weights and inherit their history format')
    parser.add_argument('--output', type=Path, default=CHAT_TRAIN_OUTPUT)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sampling', choices=('balanced', 'uniform'), default='balanced',
                        help='Balance distinct replies or sample training examples uniformly')
    parser.add_argument('--answer-class', action='store_true',
                        help='Experiment with predicting the entire reply from the question during training')
    parser.add_argument('--compact-history', action='store_true',
                        help='Train with shorter labels for previous questions and replies')
    parser.add_argument('--history-answer-dropout', type=float, default=0,
                        help='Probability of hiding previous assistant replies during training')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default=get_device(allow_mps=True))
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.eval_every) < 1 or not 0 < args.lr < float('inf'):
        parser.error('Steps, batch size, evaluation interval, and learning rate must be positive and finite.')
    if not 0 <= args.history_answer_dropout <= 1:
        parser.error('--history-answer-dropout must be between 0 and 1.')
    datasets = [args.train_data, args.val_data]
    if args.dev_data is not None:
        datasets.append(args.dev_data)
    dataset_paths = {path.resolve() for path in datasets}
    if len(dataset_paths) != len(datasets):
        parser.error('Use separate training, validation, and development files.')
    start_checkpoint = args.init_chat_checkpoint or args.base_checkpoint
    if args.output.resolve() in dataset_paths | {start_checkpoint.resolve()}:
        parser.error('Choose an output path separate from the starting checkpoint and datasets.')
    if args.output.exists() or args.output.is_symlink():
        parser.error('The output path already exists. Choose a new path with --output.')

    try:
        torch.manual_seed(args.seed)
        if args.init_chat_checkpoint is not None:
            state, tokenizer = load_chat_checkpoint(start_checkpoint)
            if args.compact_history and not tokenizer.compact_history:
                tokenizer = ChatTokenizer(tokenizer.characters, compact_history=True)
            model = model_from_state(state, len(tokenizer.characters))
        else:
            tokenizer = ChatTokenizer(chars, compact_history=args.compact_history)
            model = load_base_model(start_checkpoint)
        train_examples = load_examples(args.train_data, tokenizer)
        masked_examples = (load_examples(args.train_data, tokenizer, mask_history_replies=True)
                           if args.history_answer_dropout else None)
        val_examples = load_examples(args.val_data, tokenizer)
        dev_conversations = []
        if args.dev_data is not None:
            with args.dev_data.open(encoding='utf-8') as source:
                dev_conversations = list(read_conversations(source, args.dev_data))
            encode_conversations(dev_conversations, tokenizer, args.dev_data)
        model.to(args.device)
    except (OSError, ValueError, RuntimeError, EOFError, UnpicklingError) as error:
        parser.exit(1, f'Error preparing chat training: {error}\n')

    rng = random.Random(args.seed)
    history_rng = random.Random(args.seed + 40)
    weights = reply_sampling_weights(train_examples) if args.sampling == 'balanced' else None
    parameters = list(model.parameters())
    if args.answer_class:
        distinct_replies = dict.fromkeys(reply_tokens(targets) for _, targets in train_examples)
        reply_classes = {reply: index for index, reply in enumerate(distinct_replies)}
        classifier = torch.nn.Linear(n_embd, len(reply_classes)).to(args.device)
        parameters.extend(classifier.parameters())
        offset_rng = random.Random(args.seed + 20)
        newline_id = tokenizer.encode('\n')[0]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    print(f'Starting checkpoint: {start_checkpoint} | history packing: {tokenizer.history_packing}')
    print(f'Device: {args.device} | train replies: {len(train_examples)} | validation replies: {len(val_examples)}')
    initial_loss = evaluate(model, val_examples, args.batch_size, args.device)
    print(f'Starting checkpoint | validation reply loss {initial_loss:.4f}')

    best_score = None
    best_step = 0
    best_state = None
    model.train()
    for step in range(1, args.steps + 1):
        indices = rng.choices(range(len(train_examples)), weights=weights, k=args.batch_size)
        batch = [masked_examples[index] if masked_examples and history_rng.random() < args.history_answer_dropout
                 else train_examples[index] for index in indices]
        if args.answer_class:
            labels = torch.tensor([reply_classes[reply_tokens(targets)] for _, targets in batch], device=args.device)
            batch = [pad_prompt(example, newline_id, offset_rng) for example in batch]
        inputs, targets = collate_examples(batch)
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        if args.answer_class:
            loss, reply_loss = answer_class_loss(model, classifier, inputs, targets, labels)
        else:
            _, reply_loss = model(inputs, targets)
            loss = reply_loss
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_examples, args.batch_size, args.device)
            matches = 0
            summary = f'Step {step:>4d} | train reply loss {reply_loss.item():.4f} | validation reply loss {val_loss:.4f}'
            if dev_conversations:
                matches, total = development_matches(model, tokenizer, dev_conversations, args.device)
                summary += f' | development reference matches {matches}/{total}'
            print(summary)
            score = (-matches, val_loss)
            if isfinite(val_loss) and (best_score is None or score < best_score):
                best_state = {key: tensor.detach().to('cpu', copy=True) for key, tensor in model.state_dict().items()}
                best_score = score
                best_step = step
            model.train()

    if best_state is None:
        parser.exit(1, 'Training did not produce a finite validation loss.\n')
    model.to('cpu').load_state_dict(best_state)
    try:
        save_checkpoint(args.output, model, tokenizer)
    except OSError as error:
        parser.exit(1, f'Could not save chat checkpoint at {args.output}: {error}\n')
    print(f'Saved best chat checkpoint to {args.output} at step {best_step} (validation reply loss {best_score[1]:.4f}).')


if __name__ == '__main__':
    main()
