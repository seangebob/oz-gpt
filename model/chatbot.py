import torch
import argparse
import sys
from pathlib import Path

from model.chat_data import CHAT_CHECKPOINT, END_OF_TURN, format_prompt, load_chat_checkpoint
from model.model import (
    GPTLanguageModel, block_size, vocab_size, encode, decode,
    get_device, MODEL_PATH,
)

# CLI arguments
parser = argparse.ArgumentParser(description='Chat with the trained GPT language model')
parser.add_argument('--max_tokens', type=int, default=150,
                    help='Number of tokens to generate per prompt (default: 150)')
parser.add_argument('--chat', nargs='?', const=CHAT_CHECKPOINT, type=Path, metavar='CHECKPOINT',
                    help='Use an Oz chat checkpoint (default path: model/model-chat.pth)')
args = parser.parse_args()

device = get_device(allow_mps=args.chat is not None)
checkpoint = args.chat or Path(MODEL_PATH)
train_command = 'python -m model.chat_training' if args.chat else 'python -m model.training -batch_size 32'

# Load model
if not checkpoint.is_file():
    print(f"ERROR: No model file found at {checkpoint}")
    print(f"Train a model with {train_command}")
    sys.exit(1)

print("Loading model parameters...")
try:
    if args.chat:
        state, tokenizer = load_chat_checkpoint(checkpoint)
        model = GPTLanguageModel(len(tokenizer.characters))
        stop_ids = tokenizer.encode(END_OF_TURN)
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = GPTLanguageModel(vocab_size)
    model.load_state_dict(state)
    model.to(device)
except Exception as e:
    print(f"ERROR: Failed to load model — {e}")
    print(f"Train a model with {train_command}")
    sys.exit(1)

model.eval()
print("Model loaded successfully!\n")

# Interactive chat loop
print("=" * 50)
print("  GPT Language Model — Interactive Chat")
print("=" * 50)
print(f"  Device : {device}")
print(f"  Tokens : {args.max_tokens} per response")
print("  Type 'quit' or 'exit' to stop.")
print("=" * 50)
print()

while True:
    try:
        prompt = input("Prompt:\n")
        if prompt.lower().strip() in ('quit', 'exit'):
            print("Goodbye!")
            break
        if not prompt.strip():
            print("(Empty prompt — please enter text)\n")
            continue

        if args.chat:
            try:
                token_ids = tokenizer.encode(format_prompt(prompt.strip()))
            except ValueError as error:
                print(f"({error})")
                continue
            remaining = block_size + 1 - len(token_ids)
            if remaining <= len(stop_ids):
                print("(Prompt is too long for the chat context. Try a shorter message.)")
                continue
            context = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(context, min(args.max_tokens, remaining), stop_ids=stop_ids)
            reply = tokenizer.decode(generated[0, len(token_ids):].tolist())
            print(f"\nAssistant: {reply.removesuffix(END_OF_TURN)}\n")
            continue

        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        if context.numel() == 0:
            print("(No recognizable characters in prompt — "
                  "try using letters found in the training text.)\n")
            continue

        with torch.no_grad():
            generated = model.generate(context.unsqueeze(0),
                                       max_new_tokens=args.max_tokens)
        output_text = decode(generated[0].tolist())
        print(f"\nCompletion:\n{output_text}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break
