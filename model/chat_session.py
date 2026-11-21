from collections import deque
from dataclasses import dataclass
from textwrap import shorten

import torch

from model.chat_data import END_MARKER, END_OF_TURN, format_prompt, format_turn, with_history
from model.model import block_size


DEFAULT_MAX_TOKENS = 150
REPLY_RESERVE = 32  # Character slots for a short reply, including its end marker.
HISTORY_PACKING = 'shorten_latest_reply_if_needed'


@dataclass(frozen=True)
class ChatReply:
    text: str
    ended_with_marker: bool
    history_used: bool
    history_compacted: bool


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
        self.history.append((user_text.strip(), reply.strip()))
        while len(self.history) > 1 and sum(len(format_turn(*turn)) for turn in self.history) > block_size:
            self.history.popleft()

    def _pack_history(self, prompt, budget):
        turns = [format_turn(*turn) for turn in self.history]
        formatted = with_history(prompt, turns, budget)
        if formatted != prompt or not self.history:
            return formatted, False

        user_text, previous_reply = self.history[-1]
        reply_budget = budget - len(prompt) - len(format_turn(user_text, ''))
        if reply_budget <= 0:
            return prompt, False
        shortened = shorten(previous_reply, width=reply_budget, placeholder='', break_on_hyphens=False)
        if not shortened:
            return prompt, False
        return format_turn(user_text, shortened) + prompt, True

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
        formatted, compacted = self._pack_history(prompt, block_size + 1 - reserve)
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
        return ChatReply(
            text=text, ended_with_marker=ended, history_used=formatted != prompt,
            history_compacted=compacted,
        )
