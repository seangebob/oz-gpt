import torch
import argparse
import sys
import os

from model.model import (
    GPTLanguageModel, vocab_size, encode, decode,
    get_device, MODEL_PATH,
)

# CLI arguments
parser = argparse.ArgumentParser(description='Chat with the trained GPT language model')
parser.add_argument('--max_tokens', type=int, default=150,
                    help='Number of tokens to generate per prompt (default: 150)')
args = parser.parse_args()

device = get_device()

# Load model
model = GPTLanguageModel(vocab_size).to(device)

if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: No model file found at {MODEL_PATH}")
    print("Please train the model first by running:")
    print("  python training.py -batch_size 32")
    sys.exit(1)

print("Loading model parameters...")
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
except Exception as e:
    print(f"ERROR: Failed to load model — {e}")
    print("The model file may be corrupt or from an older format.")
    print("Please retrain by running:")
    print("  python training.py -batch_size 32")
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
