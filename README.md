# oz-gpt

A character-level language model built from scratch using PyTorch. The base Transformer learns to continue text from L. Frank Baum's *Dorothy and the Wizard in Oz*. A separate fine-tuning command teaches it to reply to short Oz questions in the terminal.

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
│   ├── chatbot.py        # Interactive CLI for book completions and chat replies
│   ├── chat_training.py  # Fine-tunes the book checkpoint on user/assistant examples
│   ├── chat_data.py      # Chat format, reply-only targets, and checkpoint loading
│   ├── data_extract.py   # Utility to extract & split large .xz corpora into train/val text files
│   ├── _test_model.py    # Non-interactive smoke-test: loads the model and generates sample text
│   ├── wizard_of_oz.txt  # Training corpus (public domain)
│   └── model-01.pth      # Bundled base model weights
├── data/                 # Starter chat training and validation examples
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

### 2. Try book-style completions

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

## Fine-tune the model for chat

Run these commands from the repo root to fine-tune the bundled checkpoint and open the chat demo.

```bash
python -m model.chat_training
python -m model.chatbot --chat
```

Try `Who is Jim?` followed by `What kind?` in the same session. Chat uses the newest complete turns that fit in its 128-character context and leaves room for a reply. Type `/reset` to clear the conversation. Retrain an existing chat checkpoint to learn the new follow-up examples. The starter dataset focuses on short exchanges about Oz.

Each line in the training and validation files is a complete conversation in JSONL format. Messages alternate between `user` and `assistant`. Keep whole conversations in separate files for training and validation.

```json
{"messages":[{"role":"user","content":"Who is Eureka?"},{"role":"assistant","content":"Eureka is Dorothy's kitten."}]}
```

The trainer formats each turn with `User:` and `Assistant:` labels and appends `[END]` to the reply. It uses the prompt and any recent complete turns that fit as context. Only the current assistant reply and its end marker contribute to the loss. Keep each formatted turn short enough for the model's 128-character context and use characters from the base vocabulary.

Training starts from `model/model-01.pth` and saves the weights with the lowest validation reply loss, the model settings, and the ordered character vocabulary to `model/model-chat.pth`. The generated chat checkpoint is ignored by Git. Pass `--train-data` and `--val-data` to use your own files, `--steps` to change the default 1,600 training steps, or `--output` to save to a different path. Load a custom output with `python -m model.chatbot --chat path/to/checkpoint.pth`. Chat training automatically selects CUDA, Apple MPS, or CPU, and `--device` can override that choice.

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

- **Expand the Oz chat examples first.** Start with 50–100 new conversations grounded in [the bundled book](model/wizard_of_oz.txt). Prioritize questions the model has handled inconsistently, including who wrote the book, who the characters are, and what happened in key scenes.
  1. Set aside 20–30 complete conversations for a final test. Keep them and near-duplicate exchanges out of the [training](data/oz_chat_train.jsonl) and [validation](data/oz_chat_val.jsonl) sets.
  2. Add several natural phrasings for each training question, with short, fact-checked answers. Include pairs of conversations where the same follow-up, such as `What kind?`, calls for different answers based on the previous turn.
  3. Use the existing JSONL format and character vocabulary. Keep each formatted exchange short enough for the 128-character context, including space for the reply and end marker.
  4. Train with the current dataset and the expanded dataset using the same base weights, seed, batch size, and number of steps. Save separate checkpoints. Score both on factual accuracy, follow-ups, and whether replies stop cleanly using the same test set. Record those results alongside validation loss and training time.
- Tokenization and Data Preprocessing Pipeline (e.g. GPT from scratch) as the model uses tokenization and reads raw text chunks.
- Enhancing Modern Transformer Architectures
- Enhancing efficient Training & Optimization   

> **Tip:** For noticeably better text quality, increase `n_head` and `n_layer` in `model/model.py` (e.g., `n_head=4`, `n_layer=4`) and train with more iterations (`--max_iters 5000`).


---

## License

This project is released for educational purposes. The training data (*The Wizard of Oz* by L. Frank Baum) is in the public domain.
