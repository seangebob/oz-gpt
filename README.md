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
- CUDA and Apple MPS support for base training, with CPU fallback

---

## Project Structure

```
.
├── README.md
├── model/                # Core code & data
│   ├── __init__.py       # Package marker
│   ├── model.py          # Model architecture & shared config (hyperparameters, vocab, paths)
│   ├── training.py       # Trains a new base checkpoint on the bundled book
│   ├── chatbot.py        # Interactive CLI for book completions and chat replies
│   ├── book_lookup.py    # Searches the Oz book and extracts answers with citations
│   ├── chat_session.py   # Chat history and generation shared by the CLI and evaluator
│   ├── chat_training.py  # Fine-tunes the book checkpoint on user/assistant examples
│   ├── chat_evaluation.py # Compares checkpoints on held-out conversations
│   ├── chat_data.py      # Chat format, reply-only targets, and checkpoint loading
│   ├── data_extract.py   # Utility to extract & split large .xz corpora into train/val text files
│   ├── _test_model.py    # Non-interactive smoke-test: loads the model and generates sample text
│   ├── wizard_of_oz.txt  # Training corpus (public domain)
│   └── model-01.pth      # Bundled base model weights
├── data/                 # Chat training, validation, and held-out conversations
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

This trains a new model on the first 90% of `wizard_of_oz.txt` and uses the final 10% for validation. Each sequence starts at an independently sampled position in its split. The command saves a new checkpoint at `model/model-base-experiment.pth`. Use a new `--output` path for each experiment. The trainer protects the bundled checkpoint and refuses to replace existing output files.

**Training arguments**

| Flag | Default | Description |
|---|---|---|
| `--batch-size`, `-batch_size` | *(required)* | Number of sequences per training step |
| `--max_iters` | `200` | Total number of training iterations |
| `--eval-every`, `--eval_iters` | `100` | Number of training steps between evaluations |
| `--eval-batches` | `20` | Number of batches per split in each evaluation |
| `--lr` | `3e-4` | Learning rate |
| `--seed` | random | Seed for model initialization and training batches |
| `--device` | automatic | Use `cuda`, `mps`, or `cpu` |
| `--heads` | `1` | Attention heads per layer |
| `--layers` | `1` | Number of Transformer layers |
| `--output` | `model/model-base-experiment.pth` | Path for a new checkpoint |

Example with custom settings

```bash
python -m model.training --batch-size 64 --max_iters 5000 --eval-every 500 \
  --eval-batches 20 --lr 1e-3 --output model/model-base-5000.pth
```

### 2. Try book-style completions

```bash
python -m model.chatbot
```

Type any prompt and the bundled checkpoint will continue it. To try the checkpoint you just trained, run `python -m model.chatbot --checkpoint model/model-base-experiment.pth`. Type `quit` or `exit` to stop.

Use the retrained base checkpoint for book-style completions.

```bash
python -m model.chatbot --checkpoint model/model-base-retrained.pth
```

The selected checkpoint has four attention heads and two Transformer layers. Its validation loss was 1.622, compared with 2.495 for the original base checkpoint on the same split. [Base retraining results](evaluations/oz_base_retraining_comparison.json)

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

Open trained chat from the repo root.

```bash
python -m model.chatbot --chat
```

Chat loads `model/model-chat-retrained.pth` by default. In the 32-turn benchmark it stopped cleanly on every turn and scored 0/32 for factual accuracy. [Reviewed chat comparison](evaluations/oz_chat_repair_comparison.json)

Try `Who is Jim?` followed by `What kind?` in the same session. Chat keeps the newest turns that fit in its 128-character context. If the latest turn is too long, it keeps the previous question and uses the start of the previous answer when they fit with room for a new reply. Type `/reset` to clear the conversation. Rerun chat training after adding new follow-up examples. The starter dataset focuses on short exchanges about Oz.

To fine-tune your own copy, run `python -m model.chat_training` and open it with `python -m model.chatbot --chat model/model-chat-experiment.pth`. Each line in the training and validation files is a complete conversation in JSONL format. Messages alternate between `user` and `assistant`. Keep each fact and its paraphrases together when building a split for unseen facts, and check for duplicate replies across training and validation. A separate paraphrase benchmark can deliberately test new wording for a fact taught in training.

```json
{"messages":[{"role":"user","content":"Who is Eureka?"},{"role":"assistant","content":"Eureka is Dorothy's kitten."}]}
```

The trainer formats each turn with `User:` and `Assistant:` labels and appends `[END]` to the reply. It uses the same history packing and shortening as live chat. When a training reply is longer than the usual live reserve, it reserves enough room for the whole target. Only the current assistant reply and its end marker contribute to the loss. Keep each formatted turn short enough for the model's 128-character context and use characters from the base vocabulary.

Training starts from `model/model-base-retrained.pth`. Use `--base-checkpoint` to start from different base weights. Training saves the weights with the lowest validation reply loss, model settings, and ordered character vocabulary to a new file. The default training output is `model/model-chat-experiment.pth`. Pass `--train-data` and `--val-data` to use your own files, `--steps` to change the default 1,600 training steps, or `--output` to choose a new path. By default, the trainer balances examples by distinct reply. Use `--sampling uniform` to compare uniform sampling. The trainer selects CUDA, Apple MPS, or CPU automatically, and `--device` can override that choice.

### Try the chat accuracy experiment

The expanded data includes 92 new conversations with book line references. The experimental checkpoint used an additional training objective that predicts the complete answer from the question and any packed history. It uses the same character-level generator when you chat.

```bash
python -m model.chatbot --chat model/model-chat-facts-experiment.pth
```

Try `What was the number of lamps Zeb brought?` or `What was underfoot in the Gargoyle country?`. The experimental checkpoint answered 3 of 22 new phrasings of training facts correctly. It answered 0 of 14 turns about facts absent from chat training and 0 of 32 turns on the original benchmark. Plain `--chat` still loads the default checkpoint. [Reviewed experiment and per-turn results](evaluations/oz_chat_fact_expansion_comparison.json)

To repeat the training recipe, run the command below. `--dev-data` selects the checkpoint by reference matches on generated replies and breaks ties with validation loss. Keep the final test files reserved for scoring. A repeat on Apple MPS produced 1/12 development matches compared with 2/12 for the bundled experiment.

```bash
python -m model.chat_training --answer-class --sampling uniform --steps 1600 \
  --eval-every 800 --batch-size 8 --lr 0.00015 --seed 7 \
  --dev-data data/oz_chat_paraphrase_dev.jsonl --output model/model-chat-facts-repeat.pth
```

### Look up answers in the Oz book

Install the optional answer reader dependency and start book lookup mode.

```bash
python -m pip install 'transformers>=4.45,<6'
python -m model.chatbot --book
```

Try `Where was the Wizard born?`. Book lookup searches the bundled text and uses chapter headings to help rank relevant passages. When a passage cuts through a paragraph that matches the question, lookup can include the rest of the sentence and extends the citation to cover it. To interpret a short follow-up, book lookup restates the question using the earlier action and the people or objects involved. The reader checks eight passages first and can search up to 24 when it finds no answer or a weak follow-up answer. Answers from the wider search need a stronger match to the question or a specific phrase already seen in the initial passages. Each extracted answer cites the chapter and source lines. When two passages suggest the same answer, it can win a close choice. The reader needs higher confidence to answer based on a single passage. It can still select a passage that answers the question poorly, so check the citation for unfamiliar facts.

The first run downloads the [MiniLM extractive answer reader](https://huggingface.co/deepset/minilm-uncased-squad2). The reader runs locally on CPU. `--chat` continues to use the Oz checkpoint trained in this project.

### Evaluate chat checkpoints and book lookup

Run the same checkpoint with ordinary chat history and with history disabled to measure the effect of memory.

```bash
python -m model.chat_evaluation run --history generated --history none --output oz-memory.json
```

To compare training datasets, save each trained checkpoint to a different path. Run both checkpoints with the same history mode and test file.

```bash
python -m model.chat_evaluation run \
  --checkpoint model/model-chat-retrained.pth --checkpoint path/to/expanded-chat.pth \
  --history generated --output oz-training-comparison.json
```

The evaluator defaults to [the held-out conversations](data/oz_chat_test.jsonl), resets between conversations, and uses greedy decoding on CPU. The report records the data and checkpoint hashes, every reply, history use, cases where a previous answer was shortened, clean stops, and reply loss computed with reference history. Add `--history reference` to diagnose follow-ups with the expected earlier replies in context.

Case, spacing, and final punctuation variations of the reference answer are accepted automatically. For each remaining `"correct": null` in the JSON report, set `correct` to `true` if the reply conveys the expected fact, including valid paraphrases and unambiguous spelling errors. Set it to `false` for an incorrect, contradictory, or missing answer. The summary shows how many replies are pending review.

```bash
python -m model.chat_evaluation summarize oz-memory.json
```

Use reviewed correct first replies and correct follow-ups as the main scores. Reference matches give a quick check, and reply loss is a separate diagnostic where lower is better. Pass multiple reports to `summarize` to compare runs saved separately on the same test file and maximum reply length.

Use `--book` to evaluate book lookup against the same conversations. Add `--checkpoint` to include a chat model in the report. Book lookup reports citations and abstentions. Set `"source_supports_reply"` to `true` only when the cited passage supports the answer to that specific question. The summary counts answers that are both correct and supported.

```bash
python -m model.chat_evaluation run --book --checkpoint model/model-chat-retrained.pth --output oz-book-comparison.json
```

To investigate missed replies, add `--trace-passages` to save the ranked passages, the search query after follow-up rewriting, and the passage the reader selected.

```bash
python -m model.chat_evaluation run --book --trace-passages --output oz-passage-review.json
```

For each missed reply, open its `passage_trace`. Check each candidate's `reader_text` if present, or its `text` otherwise. Set `evidence_ranks` to the ranks that support the expected answer, such as `[1, 3]`. Use `[]` if the reader received no supporting passage and leave `null` for cases you haven't reviewed. Record tricky calls in `review_note`. Run `python -m model.chat_evaluation summarize oz-passage-review.json` to see the review totals.

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

## Repair roadmap

Issue numbers refer to the project bug review. Complete each stage in this order.

1. **Make base training safe · complete.** **#4** sample each sequence across its full split → **#7** give evaluation frequency and batch count separate flags → **#6** save training runs to new checkpoint paths.
2. **Retrain the base model · complete.** **#5** compared the samplers and model sizes on the same validation split and saved the selected base weights at `model/model-base-retrained.pth`. [Base retraining results](evaluations/oz_base_retraining_comparison.json)
3. **Rebuild and evaluate the trained chat model.** **#2** held out complete conversation families and removed validation reply overlap → **#8** shared history formatting between training and live chat → **#3** trained and published a fresh checkpoint → **#1** expanded training and tested new phrasings and unseen facts separately. Unseen-fact accuracy remains open at 0/14. [Reviewed chat comparison](evaluations/oz_chat_repair_comparison.json) · [Fact expansion results](evaluations/oz_chat_fact_expansion_comparison.json)
4. **Improve book lookup and check it on fresh questions.** **#9** → **#11** → **#12** → **#10**. Use the order below.

### Remaining repairs in order

1. **#1 Improve factual accuracy in trained chat.** First retain enough of the previous question and answer for follow-ups and check a development set where the same short question has different answers. The experiment packed history on 0 of 4 known-fact paraphrase follow-ups. Then compare additional paraphrases and broader supervised fact coverage across multiple seeds. Keep a fresh unseen-fact test for the next final comparison and score first replies and follow-ups separately.
2. **Prepare fresh book lookup questions · complete.** The 20-turn development set and the 29-turn final set were frozen before tuning. Both include book line references for each expected answer. The original 32-turn benchmark remains the regression check. [Development questions](data/oz_book_lookup_dev.jsonl) · [Final questions](data/oz_book_lookup_test.jsonl)
3. **#9 Retain the resolved subject across turns.** Test conversations with at least three turns where a later question refers back to a named character or event.
4. **#11 Repair question rewriting.** Resolve `her` and lowercase names, handle irregular verbs such as `see` and `go`, and recognize short follow-ups such as “Why?” and “What about Zeb?” Check standalone questions that contain pronouns.
5. **#12 Normalize search terms.** Make forms such as `rescue` and `rescued`, `live` and `lived`, and `name` and `named` match. Check that nearby unrelated words stay distinct.
6. **#10 Count independent evidence.** Count a sentence shared by overlapping passages once. Retune answer selection on the development set and check that the Hungry Tiger answer still cites the jury’s appointment of its spokesman.
7. **Score book lookup on the fresh test set.** Record factual first replies, follow-up accuracy, citation support, and abstentions.

### Cleanup follow-ups

- Add ignore rules for Python bytecode, `.DS_Store`, and notebook checkpoints. Remove the 14 tracked `.pyc` files, the tracked `.DS_Store`, and the tracked files under `extra/.ipynb_checkpoints/` from Git.
- Fix `model/data_extract.py` so one `.xz` shard can produce usable training and validation splits. Connect the extracted text and vocabulary to the optional custom-corpus training workflow. Remove the committed empty outputs and retire the old Python source stored as `model/model-01.pkl`.
- Plan removal of the book’s byte-order mark together with vocabulary migration and checkpoint retraining. Keep the local `oz-*.json` reports for manual review. Apple MPS support for base training shipped with Step 2.

## Future Improvements

- **Expand the Oz chat examples.** Start with 50–100 new conversations grounded in [the bundled book](model/wizard_of_oz.txt). Prioritize questions the model has handled inconsistently, including who wrote the book, who the characters are, and what happened in key scenes.
  1. Set aside 20–30 complete conversations for a final test. Keep them and near-duplicate exchanges out of the [training](data/oz_chat_train.jsonl) and [validation](data/oz_chat_val.jsonl) sets.
  2. Add several natural phrasings for each training question, with short, fact-checked answers. Include pairs of conversations where the same follow-up, such as `What kind?`, calls for different answers based on the previous turn.
  3. Use the existing JSONL format and character vocabulary. Keep each formatted exchange short enough for the 128-character context, including space for the reply and end marker.
  4. Train with the current dataset and the expanded dataset using the same base weights, seed, batch size, and number of steps. Save separate checkpoints. Score both on factual accuracy, follow-ups, and whether replies stop cleanly using the same test set. Record those results alongside validation loss and training time.
- Tokenization and Data Preprocessing Pipeline (e.g. GPT from scratch) as the model uses tokenization and reads raw text chunks.
- Enhancing Modern Transformer Architectures
- Enhancing efficient Training & Optimization   

Use `--heads` and `--layers` to compare model sizes on the same corpus split. Load the resulting base checkpoint with `python -m model.chatbot --checkpoint path/to/checkpoint.pth`.


---

## License

This project is released for educational purposes. The training data (*The Wizard of Oz* by L. Frank Baum) is in the public domain.
