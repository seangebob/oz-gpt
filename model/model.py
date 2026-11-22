import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# Hyperparameters  (keep in sync between training and inference)
block_size = 128
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2

# Path to the saved model weights
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-01.pth")
RETRAINED_BASE_PATH = os.path.join(os.path.dirname(MODEL_PATH), "model-base-retrained.pth")

# Path to the training corpus
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wizard_of_oz.txt")


def get_device(*, allow_mps=False):
    """Select CUDA, optionally Apple MPS, or CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if allow_mps and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'



# Vocabulary & encoding  (built from wizard_of_oz.txt)
chars = ""
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[i] for i in l])


# Model architecture
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, *, heads=n_head, layers=n_layer):
        super().__init__()
        if heads < 1 or n_embd % heads or layers < 1:
            raise ValueError(f'Use at least one layer and a positive head count that divides {n_embd}.')
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, index):
        device = index.device
        length = index.size(1)

        tok_emb = self.token_embedding_table(index)                        # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(length, device=device))
        x = tok_emb + pos_emb   # (B, T, C)
        return self.ln_f(self.blocks(x))

    def forward(self, index, targets=None):
        logits = self.lm_head(self.hidden_states(index))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens, *, stop_ids=(), sample=True):
        """Generate new tokens, optionally stopping at a token sequence for one prompt."""
        if stop_ids and index.size(0) != 1:
            raise ValueError('A stop sequence requires a single prompt.')
        stop_ids = list(stop_ids)
        stop_length = len(stop_ids)

        for step in range(max_new_tokens):
            # Crop to the last block_size tokens so positional embeddings stay valid
            index_cond = index[:, -block_size:]
            logits, _loss = self.forward(index_cond)
            logits = logits[:, -1, :]             # (B, C)
            if sample:
                probs = F.softmax(logits, dim=-1)
                index_next = torch.multinomial(probs, num_samples=1)
            else:
                index_next = logits.argmax(dim=-1, keepdim=True)
            index = torch.cat((index, index_next), dim=1)         # (B, T+1)
            if stop_length and step + 1 >= stop_length:
                if index[0, -stop_length:].tolist() == stop_ids:
                    break
        return index


def model_dimensions(state):
    if not isinstance(state, dict) or 'token_embedding_table.weight' not in state:
        raise ValueError('Expected a GPT state dictionary.')
    heads = sum(key.startswith('blocks.0.sa.heads.') and key.endswith('.key.weight') for key in state)
    layers = sum(key.startswith('blocks.') and key.endswith('.ln1.weight') for key in state)
    return heads, layers


def model_from_state(state, vocabulary_size):
    heads, layers = model_dimensions(state)
    model = GPTLanguageModel(vocabulary_size, heads=heads, layers=layers)
    model.load_state_dict(state)
    return model


def load_base_model(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return model_from_state(state, vocab_size)
