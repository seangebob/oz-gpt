from collections import deque
from dataclasses import dataclass

import torch

from model.chat_data import END_MARKER, END_OF_TURN, format_prompt, format_turn, with_history
from model.model import block_size


DEFAULT_MAX_TOKENS = 150
REPLY_RESERVE = 32  # Character slots for a short reply, including its end marker.


@dataclass(frozen=True)
class ChatReply:
    text: str
    ended_with_marker: bool
    history_used: bool


class ChatSession:
    def __init__(self, model, tokenizer, *, device='cpu', max_tokens=DEFAULT_MAX_TOKENS):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.stop_ids = tokenizer.encode(END_OF_TURN)
        self.history = deque()

    def reset(self):
        self.history.clear()

    def remember(self, user_text, reply):
        self.history.append(format_turn(user_text.strip(), reply.strip()))
        while sum(map(len, self.history)) > block_size:
            self.history.popleft()

    @torch.inference_mode()
    def reply(self, user_text, *, remember=True):
        user_text = user_text.strip()
        if not user_text:
            raise ValueError('Empty prompt — please enter text')
        if END_MARKER in user_text:
            raise ValueError(f'{END_MARKER} is reserved for the model\'s end-of-turn marker.')

        prompt = format_prompt(user_text)
        remaining = block_size + 1 - len(prompt)
        if remaining <= len(self.stop_ids):
            raise ValueError('Prompt is too long for the chat context. Try a shorter message.')
        reserve = min(self.max_tokens, REPLY_RESERVE, remaining)
        formatted = with_history(prompt, self.history, block_size + 1 - reserve)
        token_ids = self.tokenizer.encode(formatted)
        remaining = block_size + 1 - len(token_ids)
        context = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            context, min(self.max_tokens, remaining), stop_ids=self.stop_ids, sample=False,
        )
        output = self.tokenizer.decode(generated[0, len(token_ids):].tolist())
        ended = output.endswith(END_OF_TURN)
        text = output.removesuffix(END_OF_TURN).strip()
        if remember and ended and text and END_MARKER not in text:
            self.remember(user_text, text)
        return ChatReply(text=text, ended_with_marker=ended, history_used=formatted != prompt)
