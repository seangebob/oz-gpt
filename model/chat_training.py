import argparse
from collections import Counter
from pathlib import Path
import random

import torch

from model.chat_data import (
    CHAT_TRAIN_OUTPUT, CHAT_FORMAT_VERSION, IGNORE_TARGET, MODEL_CONFIG,
    ChatTokenizer, collate_examples, load_examples,
)
from model.model import RETRAINED_BASE_PATH, chars, get_device, load_base_model, model_dimensions
from model.training import save_checkpoint as save_new_checkpoint


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


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
        'chat_format': CHAT_FORMAT_VERSION,
        'model_config': {**MODEL_CONFIG, 'n_head': heads, 'n_layer': layers},
        'characters': list(tokenizer.characters),
        'model_state_dict': weights,
    })


def reply_sampling_weights(examples):
    replies = [tuple(target for target in targets if target != IGNORE_TARGET) for _, targets in examples]
    counts = Counter(replies)
    return [1 / counts[reply] for reply in replies]


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the Oz model to answer user messages.')
    parser.add_argument('--train-data', type=Path, default=DATA_DIR / 'oz_chat_train.jsonl')
    parser.add_argument('--val-data', type=Path, default=DATA_DIR / 'oz_chat_val.jsonl')
    parser.add_argument('--base-checkpoint', type=Path, default=Path(RETRAINED_BASE_PATH))
    parser.add_argument('--output', type=Path, default=CHAT_TRAIN_OUTPUT)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sampling', choices=('balanced', 'uniform'), default='balanced',
                        help='Balance distinct replies or sample training examples uniformly')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default=get_device(allow_mps=True))
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.eval_every) < 1 or not 0 < args.lr < float('inf'):
        parser.error('Steps, batch size, evaluation interval, and learning rate must be positive and finite.')
    if args.train_data.resolve() == args.val_data.resolve():
        parser.error('Use separate training and validation files.')
    if args.output.resolve() in {args.base_checkpoint.resolve(), args.train_data.resolve(), args.val_data.resolve()}:
        parser.error('Choose an output path separate from the base checkpoint and datasets.')
    if args.output.exists() or args.output.is_symlink():
        parser.error('The output path already exists. Choose a new path with --output.')

    try:
        tokenizer = ChatTokenizer(chars)
        train_examples = load_examples(args.train_data, tokenizer)
        val_examples = load_examples(args.val_data, tokenizer)
        torch.manual_seed(args.seed)
        model = load_base_model(args.base_checkpoint).to(args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error preparing chat training: {error}\n')

    rng = random.Random(args.seed)
    weights = reply_sampling_weights(train_examples) if args.sampling == 'balanced' else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f'Device: {args.device} | train replies: {len(train_examples)} | validation replies: {len(val_examples)}')
    initial_loss = evaluate(model, val_examples, args.batch_size, args.device)
    print(f'Base checkpoint | validation reply loss {initial_loss:.4f}')

    best_loss = float('inf')
    best_step = 0
    best_state = None
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = collate_examples(rng.choices(train_examples, weights=weights, k=args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs.to(args.device), targets.to(args.device))
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_examples, args.batch_size, args.device)
            print(f'Step {step:>4d} | train reply loss {loss.item():.4f} | validation reply loss {val_loss:.4f}')
            if val_loss < best_loss:
                best_state = {key: tensor.detach().to('cpu', copy=True) for key, tensor in model.state_dict().items()}
                best_loss = val_loss
                best_step = step
            model.train()

    if best_state is None:
        parser.exit(1, 'Training did not produce a finite validation loss.\n')
    model.to('cpu').load_state_dict(best_state)
    try:
        save_checkpoint(args.output, model, tokenizer)
    except OSError as error:
        parser.exit(1, f'Could not save chat checkpoint at {args.output}: {error}\n')
    print(f'Saved best chat checkpoint to {args.output} at step {best_step} (validation reply loss {best_loss:.4f}).')


if __name__ == '__main__':
    main()
