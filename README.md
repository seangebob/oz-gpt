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

Install dependencies:

```bash
pip install torch
```

> **CUDA**: If you have an NVIDIA GPU, install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) for significantly faster training.

---

## Quick Start

### 1. Train the model

```bash
python -m model.training -batch_size 32
```

This trains a new model on the first 90% of `wizard_of_oz.txt` and uses the final 10% for validation. The trainer strips the leading UTF-8 byte-order mark as it reads the book. Each sequence starts at an independently sampled position in its split. The command saves the new weights and their ordered character list at `model/model-base-experiment.pth`. Use a new `--output` path for each experiment. The trainer protects the bundled checkpoint and refuses to replace existing output files.

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
| `--corpus-dir` | bundled book | Use the train, validation, and vocabulary files from the extractor |
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

Training starts from `model/model-base-retrained.pth`. Use `--base-checkpoint` to start from different Oz base weights. The chat run inherits the base checkpoint’s character order. Training saves the weights with the lowest validation reply loss, model settings, and ordered character vocabulary to a new file. The default training output is `model/model-chat-experiment.pth`. Pass `--train-data` and `--val-data` to use your own files, `--steps` to change the default 1,600 training steps, or `--output` to choose a new path. By default, the trainer balances examples by distinct reply. Use `--sampling uniform` to compare uniform sampling. The trainer selects CUDA, Apple MPS, or CPU automatically, and `--device` can override that choice.

Use `--compact-history` to train with short `Q:` and `A:` labels on earlier turns. The current turn keeps the `User:` and `Assistant:` labels. If space is tight, the trainer and chat session shorten the previous question and reply at word boundaries. The checkpoint selects its history format automatically when you load it. During training, `--history-answer-dropout 0.5` hides the earlier assistant replies on half the sampled examples so the model can practice using the previous question on its own.

Use `--init-chat-checkpoint` to start from a previous chat run with a fresh optimizer. This keeps its vocabulary and history format. You can pass `--compact-history` to switch a legacy checkpoint to the compact format during fine-tuning. Choose a new path with `--output`.

Repeat `--dev-data` to evaluate separate development files. Add `--select-by balanced-first` to prioritize the lowest first-reply match rate across those files. Ties use the average first-reply rate, total matches, then validation loss. The default selection uses total development matches with validation loss as a tiebreaker.

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

### Try the broader fact coverage checkpoint

This experiment extends the compact-history checkpoint with 180 sourced conversations covering 30 facts and events. It adds follow-ups about who acted, how many, why, and what happened next. The selected run replayed each new first reply once during training.

```bash
python -m model.chatbot --chat model/model-chat-coverage-experiment.pth
```

Ask `What animals remained in the Wizard's performing act when the princess had her pet?` and then `How many were there?`. After `/reset`, ask `What weapons did the Wizard pull from the satchel?` and repeat the follow-up. The answers are eight piglets and two revolvers.

On the fresh paraphrase test the coverage checkpoint improved first replies from 1/30 to 8/30 and follow-ups from 0/20 to 5/20. It answered both turns correctly in 3/20 multi-turn conversations. With history disabled its follow-up score was 0/20. On the earlier memory test it got 3/18 first replies and 11/18 follow-ups, compared with 7/18 and 9/18 for the starting checkpoint. It remains opt-in and scored 0/14 on facts absent from chat training. [Reviewed coverage benchmark and training runs](evaluations/oz_chat_fact_coverage_comparison.json)

### Try the generator paraphrase experiment

```bash
python -m model.chatbot --chat model/model-chat-paraphrase-experiment.pth
```

This checkpoint fine-tunes the coverage generator on five question phrasings for each of 16 additional book facts. Try `What did Ozma say the Wizard would be if he made his home in Oz?` from a fresh conversation. The checkpoint answered 5/16 on each of two reserved wording sets, with a different set of facts correct in each. Its coverage benchmark score was 6/50, compared with 13/50 for the starting checkpoint. Both scored zero on the 16-question and 14-question unseen-fact sets. [Training settings and reviewed results](evaluations/oz_chat_generator_training_comparison.json)

Use `--reply-start-weight 4` with `model.chat_training` to reproduce the controlled experiment that increased the loss weight on the first 12 answer characters. The paraphrase checkpoint uses the ordinary reply loss.

### Use the trained first-reply classifier

The classifier learns from the first replies in the chat training examples. It selects an answer when its score clears a threshold and the question fits that answer's training questions. A question with two or more key words must share at least two with them, so one familiar name cannot select an unrelated fact. A question with one key word must also open with a question word that the answer was taught with, or with none, so "Who is Jim?" and "Jim?" select the Jim answer but "Where is Jim?" goes to book lookup or the generator. With the command below, the coverage checkpoint generates replies on lower scores and on later turns. The terminal labels classifier replies. Type `/reset` to start a fresh conversation on another topic.

```bash
python -m model.chatbot --chat model/model-chat-coverage-experiment.pth --first-reply-classifier
```

With the classifier enabled, the coverage checkpoint scored 12/18 on memory first replies and 21/30 on coverage first replies. The strongest previous scores were 7/18 and 8/30. It scored 11/18 and 8/20 on their follow-ups. All configurations scored 0/16 on fresh facts absent from chat training. The separate balanced replay experiment reached 4/18 and 6/30 on first replies using generation alone. [Reviewed first-reply and generalization results](evaluations/oz_chat_first_reply_generalization_comparison.json)

To fit a classifier from the current chat dataset, run `python -m model.first_reply --output model/model-first-replies-local.pth` and pass that path to `--first-reply-classifier`.

### Use cited book answers in trained chat

```bash
python -m model.chatbot --chat model/model-chat-coverage-experiment.pth \
  --first-reply-classifier --book-assist
```

On detailed factual questions, book assistance checks the proposed answer type and looks for cues such as a named speaker or a named character in a time clause. A different extract can take precedence when its passage covers at least 75% of the question’s distinct search terms. For `What` and `Where` questions with weaker overlap, the reader can confirm an answer in a shorter excerpt. Equivalent answers keep the classifier reply and its history. Cited replies can carry their source into a follow-up.

If a proposed extract fails the scene or answer-type checks, chat looks for another supporting passage and can abstain when there is no classifier answer. When lookup finds no candidate, the generator remains available for conversational prompts such as `What happened next?`. The terminal shows citations and labels classifier replies, generator replies, and abstentions.

With the same coverage checkpoint and classifier, the scene checks raised book-assisted accuracy from 32/56 to 38/56 across four frozen benchmarks. Incorrect answers fell from 24 to 15, with three abstentions. On a separate 11-turn scene check, accuracy rose from 6/11 to 7/11 and unsupported citations went from one to three. Two of the new citations came from the wrong points in the story. [Reviewed scene routing results](evaluations/oz_chat_scene_routing_comparison.json) · [Earlier routing comparison](evaluations/oz_chat_unseen_routing_comparison.json)

Run both configurations with identical settings and review the resulting replies and citations.

```bash
python -m model.chat_evaluation run --checkpoint model/model-chat-coverage-experiment.pth \
  --first-reply-classifier --book-assist --trace-passages \
  --test-data data/oz_chat_book_assist_test.jsonl --output oz-chat-book-assist.json
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

Evaluate book lookup on the 29-turn audit benchmark from the repo root.

```bash
python -m model.chat_evaluation run --book --test-data data/oz_book_lookup_test.jsonl --output oz-book-fresh.json
```

After the four book lookup repairs, first replies scored 12/20 and later turns improved from 4/9 to 5/9. The context-dependent subset scored 4/8 in both runs. One standalone trial question improved when it stopped using the previous turn. Correct answers with supporting citations rose from 15/29 to 16/29, and abstentions fell from seven to six. The original regression set stayed at 20/32 for correct, cited answers. [Reviewed results and per-turn citations](evaluations/oz_book_lookup_repair_comparison.json)

Use the 40-turn generalization set to reproduce the comparison of follow-up context and answer selection. [Fresh conversations](data/oz_book_lookup_generalization_test.jsonl) · [Reviewed comparison](evaluations/oz_book_lookup_generalization_comparison.json)

```bash
python -m model.chat_evaluation run --book --history generated --trace-passages \
  --test-data data/oz_book_lookup_generalization_test.jsonl --output oz-book-generalization.json
```

To investigate missed replies, add `--trace-passages` to save the ranked passages, the search query after follow-up rewriting, and the passage the reader selected.

```bash
python -m model.chat_evaluation run --book --trace-passages --output oz-passage-review.json
```

For each missed reply, open its `passage_trace`. Check each candidate's `reader_text` if present, or its `text` otherwise. Set `evidence_ranks` to the ranks that support the expected answer, such as `[1, 3]`. Use `[]` if the reader received no supporting passage and leave `null` for cases you haven't reviewed. Record tricky calls in `review_note`. Run `python -m model.chat_evaluation summarize oz-passage-review.json` to see the review totals.

---

## Training Your Own Corpus

Put one or more UTF-8 text shards compressed with `.xz` in a directory, then extract, train, and run text completion.

```bash
python -m model.data_extract --input-dir path/to/shards
python -m model.training --corpus-dir model/extracted-corpus --batch-size 8 \
  --output model/extracted-corpus/custom-base.pth
python -m model.chatbot --checkpoint model/extracted-corpus/custom-base.pth
```

The extractor defaults to reading shards from `model/` and publishing to `model/extracted-corpus/`. Use `--output-dir` to choose a different new directory for another run. It splits one shard at the 90% character boundary. With multiple shards, it sorts by filename and splits by file count, keeping every shard whole. The extractor saves `output.train.txt`, `output.val.txt`, and `vocab.txt`, a JSON array of distinct characters found across both splits.

Each split needs at least 129 characters for the model's 128-character context. The trainer loads both splits into memory and uses the saved vocabulary. Custom checkpoints include their vocabulary so you can run text completion from the checkpoint alone. For Oz chat fine-tuning, use a base checkpoint with the bundled Oz vocabulary.

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
3. **Rebuild and evaluate the trained chat model.** **#2** held out complete conversation families and removed validation reply overlap → **#8** shared history formatting between training and live chat → **#3** trained and published a fresh checkpoint → **#1** expanded training and tested new phrasings and unseen facts separately. Unseen-fact accuracy remains open for the generator alone. [Reviewed chat comparison](evaluations/oz_chat_repair_comparison.json) · [Final comparison](evaluations/oz_chat_final_comparison.json)
4. **Repair book lookup and check it on fresh questions · complete.** **#9** retained the resolved subject → **#11** repaired question rewriting → **#12** normalized common verb forms → **#10** counted evidence by sentence and checked the Hungry Tiger citation. [Fresh and regression benchmarks](evaluations/oz_book_lookup_repair_comparison.json)

### Remaining repairs in order

#### 1. Improve trained chat on unseen facts

Cited answers can take precedence over unrelated first-reply classifier answers. Book-assisted chat reached 38/56 across the four frozen unseen-fact benchmarks. The standalone generator still scores 0/16 and 0/14 on the two original unseen-fact tests after an excerpt-training trial. [Scene routing results](evaluations/oz_chat_scene_routing_comparison.json) · [Reviewed routing and training results](evaluations/oz_chat_unseen_routing_comparison.json)

1. **Freeze the baselines · complete.** On the same 56 turns and chat checkpoint, the generator alone and the generator with the classifier each scored 0/56. Book-assisted chat scored 32/56 with supporting citations. The snapshot records each reply, benchmark and checkpoint hashes, and rerun commands. [Frozen baselines](evaluations/oz_chat_unseen_baselines.json)
2. **Build evidence-grounded training examples · complete.** The training set has 72 examples across 12 fact families. The development set has 18 examples across six separate families. For each question, two excerpts support different answers and a third calls for `Not enough evidence.` [Training examples](data/oz_chat_evidence_train.jsonl) · [Development examples](data/oz_chat_evidence_dev.jsonl)
3. **Test whether the generator follows the evidence · complete.** Two evidence-trained runs each abstained on 4/8 insufficient excerpts and scored 0/16 on supported questions from eight held-out fact families. They matched 68/72 and 72/72 training replies. The experimental checkpoints stay separate from the released models. [Held-out test](data/oz_chat_evidence_test.jsonl) · [Reviewed comparison](evaluations/oz_chat_evidence_comparison.json)
4. **Improve answer routing · complete.** Added scene and answer-type checks, focused rereads, and explicit abstention for rejected book extracts. Frozen accuracy rose from 32/56 to 38/56. The separate scene check rose from 6/11 to 7/11 and exposed two new citations from the wrong point in the story. [Reviewed answers and citations](evaluations/oz_chat_scene_routing_comparison.json)
5. **Run a final comparison · complete.** On 16 paired questions, every generator checkpoint scored 0/16 both with and without the supplied excerpt. Live book assistance scored 7/16 with supporting citations. Across the existing 67 unseen-fact turns, live assistance scored 45/67, including 5/9 follow-ups. The two evidence-trained checkpoints also lost reference matches on known facts, so the published checkpoint stays in service. [Final comparison and reviewed replies](evaluations/oz_chat_final_comparison.json)

The evidence files use the existing chat JSONL format. Each user message puts a `Book:` excerpt on its first line and the question on the next. The row records the excerpt’s source lines and whether it supports an answer. Each excerpt comes from the bundled book, with whitespace normalized. The training and development source ranges are disjoint and outside the previously annotated ranges. Use `--sampling uniform` for an evidence-only run to keep the dataset’s two-to-one balance of supported and insufficient answers. The frozen benchmarks serve as regression checks for evidence training. [Frozen baselines](evaluations/oz_chat_unseen_baselines.json)

#### 2. Improve `--book` follow-ups on new conversations

All three previously reviewed misses now have supporting answers. The 40-turn benchmark rose from 37/40 to 40/40, with 20/20 first replies and 20/20 follow-ups. The separate follow-up check scored 7/12 in both runs, with 6/6 first replies and 1/6 follow-ups supported. [Remaining follow-up review](evaluations/oz_book_remaining_followups_comparison.json) · [Earlier dialogue review](evaluations/oz_book_dialogue_comparison.json)

1. **Keep the scene when the question says “below.”** Resolve it to the platform in the Gargoyles’ prison and retrieve the empty rooms underneath.
2. **Keep the scene when the question says “right afterward.”** Find Zeb yawning immediately after the Wizard discusses the Gargoyles’ rest. The current answer names Zeb correctly and cites a different chapter.
3. **Recover the Sawhorse’s answer about splinters.** Retrieve and verify its statement that it never feels a break or splinter.
4. **Resolve “she” to Dorothy in the Wizard’s tricks exchange.** Carry both Dorothy and the Wizard into the search so the reply identifies his tools and machinery.
5. **Select the Scarecrow’s complete statement.** Return his point that good thoughts would stay clear. The current extract starts his next thought.
6. **Verify the fixes on two sets.** Keep these five misses as regression cases and score a separate set of new conversations for answer accuracy and citation support.

#### 3. Plan and verify the vocabulary migration

1. **Inventory the checkpoints · complete.** The two bundled base checkpoints contain raw weights. The chat checkpoints store their character order. [Checkpoint vocabulary inventory](#checkpoint-vocabulary-inventory)
2. **Define compatibility for existing checkpoints · complete.** Raw base weights use the frozen 81-character mapping. New base runs on the bundled book ignore its leading byte-order mark and save their ordered vocabulary. Chat training inherits the order of the checkpoint it starts from. [Checkpoint vocabulary inventory](#checkpoint-vocabulary-inventory)
3. **Test a migration on a copy · complete.** Remapped the input embeddings, output weights, and output bias by character in two base and two chat copies. All 42,000 shared-character logits matched exactly across 12 checkpoint and prompt pairs. [Migration check results](evaluations/oz_vocabulary_migration.json)
4. **Verify the normal commands.** Test text completion, trained chat, and `--book`. Compare legacy checkpoint behavior and verify that citations still point to the correct lines.
5. **Decide what needs retraining.** Use the compatibility and benchmark results to decide which checkpoints can be reused, remapped, or retrained.

##### Checkpoint vocabulary inventory

The files below are in `model/`. The inventory includes eight tracked checkpoints and the ignored local `model-chat.pth` when present.

| Checkpoints | Character order |
| --- | --- |
| `model-01.pth`, `model-base-retrained.pth` | Raw weights with 81 embedding and output rows. The loader uses the frozen order from the original book. |
| `model-chat-retrained.pth`, `model-chat-facts-experiment.pth`, local `model-chat.pth` | Chat format 1. Each stores an ordered `characters` list of 81 entries. |
| `model-chat-coverage-experiment.pth`, `model-chat-memory-experiment.pth`, `model-chat-paraphrase-experiment.pth` | Chat format 2 with compact history. Each stores an ordered `characters` list of 81 entries. |
| `model-first-replies-v2.pth` | Default classifier, format 2. It has the same weights as format 1 and adds the key words and question words learned for each answer. It uses no generator character IDs. |
| `model-first-replies.pth` | Classifier format 1, kept for reproducing earlier reports. It stores ordered feature terms and answer labels and uses no generator character IDs. |

All six inspected chat checkpoints store the same legacy order. The book’s leading byte-order mark (`U+FEFF`) occupies ID 80 in that order. Skipping the mark leaves IDs 0–79 intact. Raw base checkpoints keep all 81 IDs even if the book changes, and chat checkpoints keep the IDs saved with their weights. Standalone `--book` lookup decodes the text with `utf-8-sig`. [Base loaders](model/model.py) · [Chat loader](model/chat_data.py) · [Book lookup](model/book_lookup.py)

New bundled-book base runs skip the leading mark and save the resulting 80-character order in `characters` alongside `model_state_dict` in a `base_format: 1` checkpoint. This is the same format used by `--corpus-dir`. Text completion uses the saved order. Oz chat training and evaluation accept either the legacy or the 80-character Oz order. [Base trainer](model/training.py) · [Chat trainer](model/chat_training.py) · [Evaluation command](model/chat_evaluation.py)

##### Reproduce the migration check

```bash
OMP_NUM_THREADS=2 python -m evaluations.check_vocab_migration \
  --checkpoint model/model-base-retrained.pth --output /tmp/model-base-retrained-80.pth
```

Pass a legacy Oz base or chat checkpoint and a new output path. The check compares raw scores and probabilities normalized over the 80 shared characters before saving the copy. Full-vocabulary probabilities can change because the original 81-way softmax assigns some probability to `U+FEFF`. The copies used in the experiment were reloaded and checked against the source tensors. [Per-prompt measurements](evaluations/oz_vocabulary_migration.json)

### Cleanup follow-ups

- **Generated-file cleanup · complete.** Git ignores Python bytecode, macOS metadata, notebook checkpoints, and local corpus outputs. The empty extractor outputs and the legacy Python source named `model-01.pkl` were retired from Git. The source remains available in Git history.
- **Custom-corpus training · complete.** The extractor handles single-shard and multi-shard corpora, and the base trainer reads the separate splits and vocabulary. Its checkpoints work with the text completion CLI. [Run the workflow](#training-your-own-corpus)
- Keep the local `oz-*.json` reports for manual review. Apple MPS support for base training shipped with Step 2.

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
