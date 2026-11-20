import argparse
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import torch

from model.chat_data import (
    CHAT_CHECKPOINT, CHAT_FORMAT_VERSION, IGNORE_TARGET, MODEL_CONFIG,
    ChatTokenizer, collate_examples, load_examples,
)
from model.model import GPTLanguageModel, MODEL_PATH, chars, get_device


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save({
                'chat_format': CHAT_FORMAT_VERSION,
                'model_config': MODEL_CONFIG,
                'characters': list(tokenizer.characters),
                'model_state_dict': model.state_dict(),
            }, temporary)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the Oz model to answer user messages.')
    parser.add_argument('--train-data', type=Path, default=DATA_DIR / 'oz_chat_train.jsonl')
    parser.add_argument('--val-data', type=Path, default=DATA_DIR / 'oz_chat_val.jsonl')
    parser.add_argument('--output', type=Path, default=CHAT_CHECKPOINT)
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default=get_device(allow_mps=True))
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.eval_every) < 1 or not 0 < args.lr < float('inf'):
        parser.error('Steps, batch size, evaluation interval, and learning rate must be positive and finite.')
    if args.train_data.resolve() == args.val_data.resolve():
        parser.error('Use separate training and validation files.')
    if args.output.resolve() in {Path(MODEL_PATH).resolve(), args.train_data.resolve(), args.val_data.resolve()}:
        parser.error('Choose an output path separate from the base checkpoint and datasets.')

    try:
        tokenizer = ChatTokenizer(chars)
        train_examples = load_examples(args.train_data, tokenizer)
        val_examples = load_examples(args.val_data, tokenizer)
        torch.manual_seed(args.seed)
        model = GPTLanguageModel(len(tokenizer.characters))
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        model.to(args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error preparing chat training: {error}\n')

    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f'Device: {args.device} | train replies: {len(train_examples)} | validation replies: {len(val_examples)}')
    initial_loss = evaluate(model, val_examples, args.batch_size, args.device)
    print(f'Base checkpoint | validation reply loss {initial_loss:.4f}')

    best_loss = float('inf')
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = collate_examples(rng.choices(train_examples, k=args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs.to(args.device), targets.to(args.device))
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_examples, args.batch_size, args.device)
            print(f'Step {step:>4d} | train reply loss {loss.item():.4f} | validation reply loss {val_loss:.4f}')
            if val_loss < best_loss:
                save_checkpoint(args.output, model, tokenizer)
                best_loss = val_loss
            model.train()

    if best_loss == float('inf'):
        parser.exit(1, 'Training did not produce a finite validation loss.\n')
    print(f'Saved best chat checkpoint to {args.output} (validation reply loss {best_loss:.4f}).')


if __name__ == '__main__':
    main()
