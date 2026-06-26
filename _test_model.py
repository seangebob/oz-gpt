"""Non-interactive test: load the model and generate text from a prompt."""
import torch
from model import GPTLanguageModel, vocab_size, encode, decode, get_device, MODEL_PATH

device = get_device()
model = GPTLanguageModel(vocab_size).to(device)

print("Loading model...")
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print("Model loaded OK")

prompt = "Dorothy"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
print(f"\nPrompt: {prompt}")
with torch.no_grad():
    generated = model.generate(context.unsqueeze(0), max_new_tokens=100)
output = decode(generated[0].tolist())
print(f"Output:\n{output}")
