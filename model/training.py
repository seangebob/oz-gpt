import argparse
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from model.model import DATA_PATH, MODEL_PATH, GPTLanguageModel, block_size, encode, get_device, vocab_size


DEFAULT_OUTPUT = Path(MODEL_PATH).with_name('model-base-experiment.pth')


def load_splits(path):
    data = torch.tensor(encode(path.read_text(encoding='utf-8')), dtype=torch.long)
    boundary = len(data) * 9 // 10
    splits = {'train': data[:boundary], 'val': data[boundary:]}
    for name, split in splits.items():
        if len(split) <= block_size:
            raise ValueError(f'{path}: {name} split needs at least {block_size + 1} tokens.')
    return splits


def get_batch(data, batch_size, device, generator=None):
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(block_size + 1)
    windows = data[offsets].to(device)
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


@torch.no_grad()
def estimate_loss(model, splits, batch_size, eval_batches, device):
    # Use the same evaluation windows each time without consuming training randomness.
    generator = torch.Generator().manual_seed(0)
    was_training = model.training
    model.eval()
    try:
        losses = {}
        for name, split in splits.items():
            batch_losses = [
                model(*get_batch(split, batch_size, device, generator))[1]
                for _ in range(eval_batches)
            ]
            losses[name] = torch.stack(batch_losses).mean().item()
        return losses
    finally:
        model.train(was_training)


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(state, temporary)
        # Publish the complete file without replacing an output created during training.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Train a new GPT base checkpoint on wizard_of_oz.txt.')
    parser.add_argument('-batch_size', '--batch-size', type=int, required=True, help='Sequences per training step')
    parser.add_argument('--max_iters', type=int, default=200, help='Number of training steps (default: 200)')
    parser.add_argument('--eval-every', '--eval_iters', type=int, default=100,
                        help='Training steps between evaluations (default: 100)')
    parser.add_argument('--eval-batches', type=int, default=20,
                        help='Batches per split at each evaluation (default: 20)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='New checkpoint path (default: model/model-base-experiment.pth)')
    args = parser.parse_args()
    if min(args.batch_size, args.max_iters, args.eval_every, args.eval_batches) < 1:
        parser.error('Batch size, training steps, evaluation interval, and evaluation batches must be positive.')
    if not isfinite(args.lr) or args.lr <= 0:
        parser.error('Learning rate must be positive and finite.')
    if args.output.resolve() == Path(MODEL_PATH).resolve():
        parser.error('The bundled base checkpoint is protected. Choose a new path with --output.')
    if args.output.exists() or args.output.is_symlink():
        parser.error(f'Output already exists at {args.output}. Choose a new path with --output.')

    try:
        splits = load_splits(Path(DATA_PATH))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    device = get_device()
    model = GPTLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f'Using device: {device}')
    print(f'batch_size={args.batch_size}  max_iters={args.max_iters}  eval_every={args.eval_every}  '
          f'eval_batches={args.eval_batches}  lr={args.lr}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    for step in range(args.max_iters):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, splits, args.batch_size, args.eval_batches, device)
            print(f"step {step:>5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch(splits['train'], args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(model, splits, args.batch_size, args.eval_batches, device)
    print(f"Final    | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    try:
        save_checkpoint(args.output, model.state_dict())
    except OSError as error:
        parser.exit(1, f'Could not save new checkpoint at {args.output}: {error}\n')
    print(f'Model saved to {args.output}')


if __name__ == '__main__':
    main()
