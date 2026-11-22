from collections import deque
from dataclasses import dataclass

import torch

from model.chat_data import END_MARKER, END_OF_TURN, REPLY_RESERVE, format_compact_turn, format_prompt, format_turn, pack_history
from model.model import block_size


DEFAULT_MAX_TOKENS = 150


@dataclass(frozen=True)
class ChatReply:
    text: str
    ended_with_marker: bool
    history_used: bool
    history_compacted: bool
    reply_method: str = 'generator'


class ChatSession:
    def __init__(self, model, tokenizer, *, device='cpu', max_tokens=DEFAULT_MAX_TOKENS,
                 first_reply_classifier=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.first_reply_classifier = first_reply_classifier
        self.stop_ids = tokenizer.encode(END_OF_TURN)
        self.history = deque()
        self.reset()

    def reset(self):
        self.history.clear()
        self._first_reply = True

    def remember(self, user_text, reply):
        self.history.append((user_text.strip(), reply.strip()))
        formatter = format_compact_turn if self.tokenizer.compact_history else format_turn
        while len(self.history) > 1 and sum(len(formatter(*turn)) for turn in self.history) > block_size:
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
        formatted, compacted = pack_history(prompt, self.history, block_size + 1 - reserve,
                                            compact_history=self.tokenizer.compact_history)
        token_ids = self.tokenizer.encode(formatted)
        remaining = block_size + 1 - len(token_ids)
        limit = min(self.max_tokens, remaining)
        if self.first_reply_classifier is not None and self._first_reply and not self.history:
            answer = self.first_reply_classifier.predict(user_text)
            if answer and len(answer) + len(self.stop_ids) <= limit and set(answer) <= self.tokenizer.ids.keys():
                if remember:
                    self.remember(user_text, answer)
                self._first_reply = False
                return ChatReply(answer, False, False, False, reply_method='classifier')

        context = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            context, limit, stop_ids=self.stop_ids, sample=False,
        )
        self._first_reply = False
        output = self.tokenizer.decode(generated[0, len(token_ids):].tolist())
        ended = output.endswith(END_OF_TURN)
        text = output.removesuffix(END_OF_TURN).strip()
        if remember and ended and text and END_MARKER not in text:
            self.remember(user_text, text)
        return ChatReply(
            text=text, ended_with_marker=ended, history_used=formatted != prompt,
            history_compacted=compacted,
        )
