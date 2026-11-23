import argparse
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from model.data_extract import TRAIN_FILE, VAL_FILE, VOCAB_FILE
from model.model import (
    CUSTOM_BASE_FORMAT, DATA_PATH, MODEL_PATH, CharacterTokenizer, GPTLanguageModel, block_size, encode,
    get_device, n_head, n_layer, vocab_size,
)


DEFAULT_OUTPUT = Path(MODEL_PATH).with_name('model-base-experiment.pth')


def load_splits(path):
    data = torch.tensor(encode(path.read_text(encoding='utf-8')), dtype=torch.long)
    boundary = len(data) * 9 // 10
    return _validate_splits({'train': data[:boundary], 'val': data[boundary:]}, path)


def load_extracted_splits(directory):
    train_path, validation_path = directory / TRAIN_FILE, directory / VAL_FILE
    if train_path.samefile(validation_path):
        raise ValueError('Training and validation must use separate files.')
    vocab_path = directory / VOCAB_FILE
    tokenizer = CharacterTokenizer(json.loads(vocab_path.read_text(encoding='utf-8')))
    splits = {}
    for name, path in (('train', train_path), ('val', validation_path)):
        try:
            splits[name] = torch.tensor(tokenizer.encode(path.read_text(encoding='utf-8')), dtype=torch.long)
        except ValueError as error:
            raise ValueError(f'{path}: {error}') from error
    return _validate_splits(splits, directory), tokenizer


def _validate_splits(splits, source):
    for name, split in splits.items():
        if len(split) <= block_size:
            raise ValueError(f'{source}: {name} split needs at least {block_size + 1} tokens.')
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
    parser = argparse.ArgumentParser(description='Train a new GPT base checkpoint on the bundled book or an extracted text corpus.')
    parser.add_argument('-batch_size', '--batch-size', type=int, required=True, help='Sequences per training step')
    parser.add_argument('--max_iters', type=int, default=200, help='Number of training steps (default: 200)')
    parser.add_argument('--eval-every', '--eval_iters', type=int, default=100,
                        help='Training steps between evaluations (default: 100)')
    parser.add_argument('--eval-batches', type=int, default=20,
                        help='Batches per split at each evaluation (default: 20)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--seed', type=int, help='Seed model initialization, dropout, and training batches')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default=get_device(allow_mps=True),
                        help='Training device (default: CUDA, Apple MPS, or CPU, in that order)')
    parser.add_argument('--heads', type=int, default=n_head, help=f'Attention heads per layer (default: {n_head})')
    parser.add_argument('--layers', type=int, default=n_layer, help=f'Transformer layers (default: {n_layer})')
    parser.add_argument('--corpus-dir', type=Path,
                        help='Directory produced by model.data_extract (default: train on the bundled book)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='New checkpoint path (default: model/model-base-experiment.pth)')
    args = parser.parse_args()
    if min(args.batch_size, args.max_iters, args.eval_every, args.eval_batches) < 1:
        parser.error('Batch size, training steps, evaluation interval, and evaluation batches must be positive.')
    if not isfinite(args.lr) or args.lr <= 0:
        parser.error('Learning rate must be positive and finite.')
    if args.seed is not None and not 0 <= args.seed < 2**64:
        parser.error('Seed must be an unsigned 64-bit integer.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('Apple MPS is unavailable.')
    if args.output.resolve() == Path(MODEL_PATH).resolve():
        parser.error('The bundled base checkpoint is protected. Choose a new path with --output.')
    if args.output.exists() or args.output.is_symlink():
        parser.error(f'Output already exists at {args.output}. Choose a new path with --output.')

    try:
        tokenizer = None
        if args.corpus_dir is None:
            splits = load_splits(Path(DATA_PATH))
            vocabulary_size = vocab_size
        else:
            splits, tokenizer = load_extracted_splits(args.corpus_dir)
            vocabulary_size = len(tokenizer.characters)
        sampler = None
        if args.seed is not None:
            torch.manual_seed(args.seed)
            sampler = torch.Generator().manual_seed(args.seed)
        model = GPTLanguageModel(vocabulary_size, heads=args.heads, layers=args.layers).to(args.device)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    device = args.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f'Using device: {device}')
    if tokenizer is not None:
        print(f'Corpus: {args.corpus_dir} | train tokens: {len(splits["train"]):,} | '
              f'validation tokens: {len(splits["val"]):,} | vocabulary size: {vocabulary_size:,}')
    print(f'batch_size={args.batch_size}  max_iters={args.max_iters}  eval_every={args.eval_every}  '
          f'eval_batches={args.eval_batches}  lr={args.lr}  seed={args.seed}')
    print(f'heads={args.heads}  layers={args.layers}  parameters={sum(p.numel() for p in model.parameters()):,}')

    for step in range(args.max_iters):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, splits, args.batch_size, args.eval_batches, device)
            print(f"step {step:>5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch(splits['train'], args.batch_size, device, sampler)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(model, splits, args.batch_size, args.eval_batches, device)
    print(f"Final    | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    state = model.state_dict()
    if tokenizer is not None:
        state = {
            'base_format': CUSTOM_BASE_FORMAT,
            'characters': list(tokenizer.characters),
            'model_state_dict': state,
        }
    try:
        save_checkpoint(args.output, state)
    except OSError as error:
        parser.exit(1, f'Could not save new checkpoint at {args.output}: {error}\n')
    print(f'Model saved to {args.output}')


if __name__ == '__main__':
    main()
