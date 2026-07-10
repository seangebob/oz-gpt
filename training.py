import torch
import mmap
import random
import argparse

from model import (
    GPTLanguageModel, vocab_size, encode,
    block_size, n_embd, n_head, n_layer, dropout,
    get_device, MODEL_PATH, DATA_PATH,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the GPT language model on wizard_of_oz.txt')
parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training')
parser.add_argument('--max_iters', type=int, default=200, help='Number of training iterations (default: 200)')
parser.add_argument('--eval_iters', type=int, default=100, help='How often to evaluate loss (default: 100)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')

args = parser.parse_args()

device = get_device()
print(f"Using device: {device}")
print(f"batch_size={args.batch_size}  max_iters={args.max_iters}  "
      f"eval_iters={args.eval_iters}  lr={args.lr}")


# ---------------------------------------------------------------------------
# Data loading — byte-offset 90/10 train / val split of a single file
# ---------------------------------------------------------------------------
def get_random_chunk(split):
    """Return a tensor of encoded characters from a random position in the file.

    The first 90 % of the file is used for training; the last 10 % for
    validation.  This gives a meaningful (if imperfect) val signal even with
    only one data file.
    """
    with open(DATA_PATH, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            boundary = int(file_size * 0.9)
            need = block_size * args.batch_size + 1  # +1 for the target offset

            if split == 'train':
                # Sample from [0, boundary)
                max_start = boundary - need
                if max_start < 0:
                    max_start = 0
                start_pos = random.randint(0, max_start)
            else:
                # Sample from [boundary, end)
                max_start = file_size - need
                if max_start < boundary:
                    max_start = boundary
                start_pos = random.randint(boundary, max_start)

            mm.seek(start_pos)
            block = mm.read(need)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_chunk(split)
    # Make sure we have enough data; clamp if the chunk was short
    max_ix = len(data) - block_size
    if max_ix < 1:
        max_ix = 1
    ix = torch.randint(max_ix, (args.batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model = GPTLanguageModel(vocab_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

for step in range(args.max_iters):
    if step % args.eval_iters == 0:
        losses = estimate_loss()
        print(f"step {step:>5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
losses = estimate_loss()
print(f"Final    | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

# Save model weights (state dict — portable & safe)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
