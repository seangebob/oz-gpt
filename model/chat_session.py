from collections import deque
from dataclasses import dataclass
import re

import torch

from model.book_lookup import Passage, PassageTrace, uses_book_history
from model.chat_data import END_MARKER, END_OF_TURN, REPLY_RESERVE, format_compact_turn, format_prompt, format_turn, pack_history
from model.model import block_size


DEFAULT_MAX_TOKENS = 150
QUESTION_START = re.compile(r'^(?:who|whom|whose|what|which|when|where|why|how|is|are|was|were|did|does|do|'
                            r'can|could|would|will|has|have|describe|tell me)\b', re.IGNORECASE)
UNANCHORED_EVENT = re.compile(
    r'(?:what (?:happened|came) (?:next|afterwards?)|why did (?:that|this|it) (?:happen|occur))\W*',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatReply:
    text: str
    ended_with_marker: bool
    history_used: bool
    history_compacted: bool
    reply_method: str = 'generator'
    source: Passage | None = None
    passage_trace: PassageTrace | None = None


class ChatSession:
    def __init__(self, model, tokenizer, *, device='cpu', max_tokens=DEFAULT_MAX_TOKENS,
                 first_reply_classifier=None, book_session=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.first_reply_classifier = first_reply_classifier
        self.book_session = book_session
        self.stop_ids = tokenizer.encode(END_OF_TURN)
        self.history = deque()
        self.reset()

    def reset(self):
        self.history.clear()
        self._first_reply = True
        if self.book_session is not None:
            self.book_session.reset()

    def remember(self, user_text, reply):
        self._remember_generation(user_text, reply)
        if self.book_session is not None:
            self.book_session.remember(user_text, reply)

    def _remember_generation(self, user_text, reply):
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

        book_reply = None
        if (self.book_session is not None and (user_text.endswith('?') or QUESTION_START.match(user_text))
                and not UNANCHORED_EVENT.fullmatch(user_text)
                and (self._first_reply or self.book_session.previous_source is not None
                     or not uses_book_history(user_text))):
            book_reply = self.book_session.reply(user_text, remember=remember, remember_abstentions=False)
            if book_reply.source is not None:
                self._first_reply = False
                if remember:
                    if set(book_reply.text) <= self.tokenizer.ids.keys():
                        self._remember_generation(user_text, book_reply.text)
                    else:
                        # Keep the cited turn in book history. Generation requires its own vocabulary.
                        self.history.clear()
                return ChatReply(book_reply.text, False, book_reply.history_used, False,
                                 reply_method='book', source=book_reply.source, passage_trace=book_reply.passage_trace)

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
            history_compacted=compacted, passage_trace=book_reply.passage_trace if book_reply else None,
        )
