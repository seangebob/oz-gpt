# oz-gpt

An LLM built entirely from scratch using PyTorch. Trained on The Wizard of Oz, the model learns to generate text one character at a time using a Transformer architecture with multi-head self-attention.

---

## Features

- Character-level tokenisation (no external tokeniser required)
- Transformer architecture with:
  - Multi-head self-attention (`Head`, `MultiHeadAttention`)
  - Position-wise feed-forward layers (`FeedForward`)
  - Residual connections & layer normalisation (`Block`)
- Auto-regressive text generation
- Interactive chatbot interface
- CUDA support (falls back to CPU automatically)

---

## Project Structure

```
.
├── README.md
├── model/                # Core code & data
│   ├── __init__.py       # Package marker
│   ├── model.py          # Model architecture & shared config (hyperparameters, vocab, paths)
│   ├── training.py       # Training loop — trains and saves model weights to model-01.pth
│   ├── chatbot.py        # Interactive CLI chatbot that loads the trained model
│   ├── data_extract.py   # Utility to extract & split large .xz corpora into train/val text files
│   ├── _test_model.py    # Non-interactive smoke-test: loads the model and generates sample text
│   ├── wizard_of_oz.txt  # Training corpus (public domain)
│   └── model-01.pth      # Saved model weights (created after training — not tracked by git)
└── extra/                # Notebooks, experiments, and other supporting files
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (with optional CUDA support)
- `tqdm` (only needed for `data_extract.py`)

Install dependencies:

```bash
pip install torch tqdm
```

> **CUDA**: If you have an NVIDIA GPU, install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) for significantly faster training.

---

## Quick Start

### 1. Train the model

```bash
python -m model.training -batch_size 32
```

This trains on `wizard_of_oz.txt` and saves the weights to `model-01.pth`.

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `-batch_size` | *(required)* | Number of sequences per training step |
| `--max_iters` | `200` | Total number of training iterations |
| `--eval_iters` | `100` | How often to print the loss |
| `--lr` | `3e-4` | Learning rate |

Example with custom settings:
```bash
python -m model.training -batch_size 64 --max_iters 5000 --eval_iters 500 --lr 1e-3
```

### 2. Chat with the model

```bash
python -m model.chatbot
```

Type any prompt and the model will auto-regressively complete it. Type `quit` or `exit` to stop.

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--max_tokens` | `150` | Number of tokens to generate per response |

Example:
```bash
python -m model.chatbot --max_tokens 300
```

### 3. Run a quick smoke test

```bash
python -m model._test_model
```

Loads the model and generates 100 tokens from the prompt `"Dorothy"` — useful for verifying a trained checkpoint is working correctly.

---

## Training Your Own Corpus

If you want to train on a **larger dataset** (e.g., OpenWebText `.xz` shards), use `data_extract.py`:

1. Place `.xz`-compressed text files in the `model/` directory.
2. Run:
   ```bash
   python -m model.data_extract
   ```
3. This produces `output.train.txt`, `output.val.txt`, and `vocab.txt` with a 90/10 train/val split.

You would then update `DATA_PATH` in `model/model.py` to point at your new training file.

---

## Hyperparameters

All key hyperparameters live in `model/model.py` and are shared between training and inference:

| Parameter | Default | Description |
|---|---|---|
| `block_size` | `128` | Context window (max sequence length) |
| `n_embd` | `384` | Embedding dimension |
| `n_head` | `1` | Number of attention heads |
| `n_layer` | `1` | Number of Transformer blocks |
| `dropout` | `0.2` | Dropout probability |

> **Tip:** Increasing `n_head` and `n_layer` in `model/model.py` (e.g., `n_head=6`, `n_layer=6`) will produce a much more capable model but requires more memory and training time.

---

## How It Works

```
Input text
    │
    ▼
Character-level tokenisation  (encode / decode)
    │
    ▼
Token + Positional Embeddings  (nn.Embedding)
    │
    ▼
Transformer Blocks × n_layer
    │  ├── Multi-Head Self-Attention  (causal mask)
    │  ├── Layer Norm + Residual
    │  ├── Feed-Forward Network
    │  └── Layer Norm + Residual
    │
    ▼
Linear projection → logits (vocab_size)
    │
    ▼
Softmax → sample next character
```

---

## Future Improvements
- Tokenization and Data Preprocessing Pipeline (e.g. GPT from scratch) as the model uses tokenization and reads raw text chunks.
- Enhancing Modern Transformer Architectures
- Enhancing efficient Training & Optimization   

> **Tip:** For noticeably better text quality, increase `n_head` and `n_layer` in `model/model.py` (e.g., `n_head=4`, `n_layer=4`) and train with more iterations (`--max_iters 5000`).


---

## License

This project is released for educational purposes. The training data (*The Wizard of Oz* by L. Frank Baum) is in the public domain.
