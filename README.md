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

Use `--compact-history` to train with short `Q:` and `A:` labels on earlier turns. The current turn keeps the `User:` and `Assistant:` labels. If space is tight, the trainer and chat session shorten the previous question and reply at word boundaries. The checkpoint selects its history format automatically when you load it. During training, `--history-answer-dropout 0.5` hides the earlier assistant replies on half the sampled examples so the model can practice using the previous question on its own.

Use `--init-chat-checkpoint` to start from a previous chat run with a fresh optimizer. This keeps its vocabulary and history format. You can pass `--compact-history` to switch a legacy checkpoint to the compact format during fine-tuning. Choose a new path with `--output`.

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

### Try the compact-history checkpoint

The compact-history experiment added 108 sourced conversations. Its final memory test uses three follow-up questions. Each follows six different earlier questions and calls for a different answer in each conversation. The first questions use wording held out of training.

```bash
python -m model.chatbot --chat model/model-chat-memory-experiment.pth
```

Ask `What did the Wizard wear when he joined the others for dinner?` followed by `What color was it?`. After `/reset`, try `What did Zeb change into at the palace?` and ask `What color was it?` again. Both first replies describe a velvet costume. The Wizard wore black and Zeb wore maroon.

On the final memory test the selected checkpoint scored 7/18 first replies and 9/18 follow-ups. It kept history on all 18 follow-ups. The same weights scored 0/18 follow-ups with history disabled and 11/18 with the expected previous replies supplied. The default checkpoint scored 0/18 follow-ups. The compact checkpoint remains opt-in because performance varied across seeds and it scored 0/11 on facts absent from the chat data. [Reviewed memory benchmark and training runs](evaluations/oz_chat_compact_history_comparison.json)

```bash
python -m model.chat_training --compact-history --history-answer-dropout 0.5 \
  --answer-class --sampling uniform --steps 3200 --eval-every 800 \
  --batch-size 8 --lr 0.00015 --seed 7 --dev-data data/oz_chat_memory_dev.jsonl \
  --output model/model-chat-memory-repeat.pth
```

### Look up answers in the Oz book

Install the optional answer reader dependency and start book lookup mode.

```bash
python -m pip install 'transformers>=4.45,<6'
python -m model.chatbot --book
```

Try `Where was the Wizard born?`. Book lookup searches the bundled text and uses chapter headings to help rank relevant passages. When a passage cuts through a paragraph that matches the question, lookup can include the rest of the sentence and extends the citation to cover it. Short follow-ups use the earlier action and can carry a resolved subject across turns. The reader checks eight passages first and can search up to 24 when it finds no answer or a weak follow-up answer. Each extracted answer cites the chapter and source lines. Agreement across independent sentences can settle a close choice. Overlapping windows that contain the same sentence count once.

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

Evaluate book lookup on the fresh 29-turn benchmark from the repo root.

```bash
python -m model.chat_evaluation run --book --test-data data/oz_book_lookup_test.jsonl --output oz-book-fresh.json
```

After the four book lookup repairs, first replies scored 12/20 and later turns improved from 4/9 to 5/9. The context-dependent subset scored 4/8 in both runs. One standalone trial question improved when it stopped using the previous turn. Correct answers with supporting citations rose from 15/29 to 16/29, and abstentions fell from seven to six. The original regression set stayed at 20/32 for correct, cited answers. [Reviewed results and per-turn citations](evaluations/oz_book_lookup_repair_comparison.json)

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
4. **Repair book lookup and check it on fresh questions · complete.** **#9** retained the resolved subject → **#11** repaired question rewriting → **#12** normalized common verb forms → **#10** counted evidence by sentence and checked the Hungry Tiger citation. [Fresh and regression benchmarks](evaluations/oz_book_lookup_repair_comparison.json)

### Remaining repairs in order

1. **#1 Improve factual accuracy in trained chat.** The compact-history checkpoint retained prior context on all 18 paired test follow-ups and answered 9 correctly. It also retained context on all four follow-ups in the earlier paraphrase test, where it scored 0/4. Next improve factual first replies and broaden the kinds of follow-ups taught in training. Test broader fact coverage across seeds and reserve a fresh set of facts absent from chat training before promoting a new default. [Memory benchmark results](evaluations/oz_chat_compact_history_comparison.json)
2. **Improve book lookup on the misses from the fresh benchmark.** Review whether the retrieved passages contain the correct answer, then investigate when an incorrect first reply becomes the next turn’s subject. Use a new final set for the next tuning round. [Development questions](data/oz_book_lookup_dev.jsonl) · [Final questions](data/oz_book_lookup_test.jsonl)

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
