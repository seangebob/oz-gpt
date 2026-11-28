from collections import deque
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import re

import torch

from model.book_lookup import (
    BOOK_NAME, COORDINATED_ACTOR, FEMALE_NAME, IDENTITY_QUESTION, MALE_NAME, NAME_ANSWER, NO_ANSWER, PERSONAL_REFERENCE, PLURAL_PERSON,
    QUESTION_ACTOR, REPORTING_ACTOR, Passage, PassageTrace, search_terms, uses_book_history,
)
from model.chat_data import END_MARKER, END_OF_TURN, REPLY_RESERVE, format_compact_turn, format_prompt, format_turn, pack_history
from model.model import block_size


DEFAULT_MAX_TOKENS = 150
MIN_BOOK_OVERRIDE_COVERAGE = 0.75
QUESTION_START = re.compile(r'^(?:who|whom|whose|what|which|when|where|why|how|is|are|was|were|did|does|do|'
                            r'can|could|would|will|has|have|describe|tell me)\b', re.IGNORECASE)
UNANCHORED_EVENT = re.compile(
    r'(?:what (?:happened|came) (?:next|afterwards?)|why did (?:that|this|it) (?:happen|occur))\W*',
    re.IGNORECASE,
)
SIZE_QUESTION = re.compile(r'^how (?:big|small|large|tall|high|wide)\b', re.IGNORECASE)
BARE_COUNT = re.compile(
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|dozen)[.!]?",
    re.IGNORECASE,
)
SCENE_CLAUSE = re.compile(r'\b(?:when|while|after|before|during)\b', re.IGNORECASE)
PROPER_WORD = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b")


def _names_subject(question):
    return bool(BOOK_NAME.search(question) or any(match.start() > 0 for match in PROPER_WORD.finditer(question)))


def _is_person_pronoun(word):
    # "What did he pull?" points back to the person asked about. "What was it made of?" usually points to the thing
    # in the reply, such as Dorothy's suit-case, which only the chat history holds.
    return PERSONAL_REFERENCE.fullmatch(word) is not None or PLURAL_PERSON.fullmatch(word) is not None


def _asked_person(question, answer):
    """Return the person an earlier question asks about, or '' when it doesn't name them."""
    question = question.replace('’', "'").strip()
    if identity := IDENTITY_QUESTION.fullmatch(question):
        return identity[1]
    if re.match(r'who\b', question, re.IGNORECASE):
        # "Who drove the buggy?" asks about whoever the answer names. "Who is Ozma's father?" asks about her father,
        # not Ozma.
        name = answer.strip(' .?!"')
        return name if NAME_ANSWER.fullmatch(name) else ''
    actor = QUESTION_ACTOR.match(question)
    if not actor or actor[1].endswith("'s") or question.startswith("'s", actor.end()):
        return ''
    # "What did Zeb and the Wizard take off the hinges?" asks about both of them.
    pair = COORDINATED_ACTOR.match(question, actor.end())
    return f'{actor[1]} and {pair[1]}' if pair else actor[1]


def _person_key(name):
    return name.casefold().removesuffix("'s").removeprefix('the ')


def _fits_pronoun(name, pronoun):
    pronoun = pronoun.lower()
    if ' and ' in name:
        return pronoun in {'they', 'them', 'their'}
    if pronoun in {'she', 'her', 'hers'}:
        return MALE_NAME.fullmatch(name) is None
    if pronoun in {'he', 'him', 'his'}:
        return FEMALE_NAME.fullmatch(name) is None
    return True


def _replaces_pronouns_with(question, resolved, person):
    """Return whether rewriting question as resolved only replaced personal pronouns, each with person."""
    before, after = (re.findall(r"[\w'-]+", text.replace('’', "'")) for text in (question, resolved))
    changes = [(before[i1:i2], ' '.join(after[j1:j2]))
               for tag, i1, i2, j1, j2 in SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
               if tag != 'equal']
    return bool(person) and bool(changes) and all(
        len(old) == 1 and _is_person_pronoun(old[0]) and _person_key(new) == _person_key(person)
        and _fits_pronoun(new.removesuffix("'s"), old[0])
        for old, new in changes)


def _scene_terms(question):
    anchors = set()
    reporting = REPORTING_ACTOR.search(question)
    # A common noun such as "the group" rarely appears under that wording in the book.
    if reporting and any(character.isupper() for character in reporting[1]):
        anchors.update(search_terms(reporting[1]))
    time_clause = SCENE_CLAUSE.search(question.partition(' ')[2])
    if time_clause:
        setting = time_clause.string[time_clause.end():]
        for name in PROPER_WORD.findall(setting):
            anchors.update(search_terms(name))
        for name in BOOK_NAME.finditer(setting):
            anchors.update(search_terms(name[0]))
    return anchors


def _answer_type_matches(question, text):
    if SIZE_QUESTION.match(question) and BARE_COUNT.fullmatch(text):
        return False
    if question.casefold().startswith('where '):
        name = re.sub(r'^(?:the |princess )+', '', text.strip(' .!?'), flags=re.IGNORECASE)
        return not BOOK_NAME.fullmatch(name)
    return True


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
        self._follows_unanswered = False
        self._follows_uncited = False
        if self.book_session is not None:
            self.book_session.reset()

    def remember(self, user_text, reply):
        self._remember_generation(user_text, reply)
        if self.book_session is not None:
            self.book_session.remember(user_text, reply)
            self._follows_unanswered = self._follows_uncited = False

    def _resolves_person(self, question):
        # "What did he pull?" after "Who is Jim?" becomes "What did Jim pull?". "Where did they go with Dorothy?" names
        # Dorothy but "they" has no referent, and after "Who is Ozma's father?", "What happened to him?" would become a
        # question about Ozma. A pronoun left after the subject is filled in, as in "How did Jim make them scatter?",
        # may refer to something else in the question.
        resolved = self.book_session.resolve(question)
        person = _asked_person(self.book_session.previous_question or '', self.book_session.previous_answer or '')
        return _replaces_pronouns_with(question, resolved, person) and _names_subject(resolved)

    def _remember_uncited(self, user_text, reply, *, trusted):
        # A classifier or generated reply has no passage behind it. A follow-up about a person, such as "What did he
        # pull?" after "Who is Jim?", can still look up the earlier subject. "What kind?" continues the reply itself.
        # Only a classifier fact, not generated text, helps resolve the pronoun.
        self._remember_generation(user_text, reply)
        if self.book_session is not None:
            self.book_session.remember(user_text, reply if trusted else '')
            self._follows_unanswered = False
            self._follows_uncited = True

    def _remember_unanswered(self, user_text):
        # Without an answer to cite, only a question that names someone, directly or through a resolved
        # pronoun, can replace the earlier subject. "Thanks, that makes sense." or "Really?" leaves it in place.
        asks = user_text.endswith('?') or QUESTION_START.match(user_text)
        if not asks or not _names_subject(self.book_session.resolve(user_text)):
            return
        # Keep the newest subject for follow-ups without storing an unverified answer.
        self.history.clear()
        self.book_session.remember(user_text, '')
        self._follows_unanswered = True
        self._follows_uncited = False

    def _remember_generation(self, user_text, reply):
        self.history.append((user_text.strip(), reply.strip()))
        formatter = format_compact_turn if self.tokenizer.compact_history else format_turn
        while len(self.history) > 1 and sum(len(formatter(*turn)) for turn in self.history) > block_size:
            self.history.popleft()

    def _route_book_reply(self, question, book_reply, classified, question_terms):
        if book_reply.source is None:
            return None
        trace = book_reply.passage_trace
        # A resolved follow-up already carries the earlier scene into lookup.
        anchors = set() if book_reply.history_used else _scene_terms(question)

        def accepts(passage, answer):
            # A scene can name someone just outside the extract, so check its whole chapter.
            in_scene = not anchors or anchors.issubset(self.book_session.chapter_terms(passage))
            return in_scene and _answer_type_matches(question, answer)

        eligible = accepts(book_reply.source, book_reply.text)
        if eligible:
            if classified is None:
                return book_reply
            classifier_terms = set(search_terms(classified))
            if not set(search_terms(book_reply.text)).difference(classifier_terms):
                return None
            passage_terms = set(search_terms(book_reply.source.text))
            if len(question_terms & passage_terms) >= MIN_BOOK_OVERRIDE_COVERAGE * len(question_terms):
                return book_reply
            # Relax the overlap check only for concrete fact or location questions.
            if (not question.casefold().startswith(('what ', 'where '))
                    or classifier_terms.issubset(passage_terms)):
                return None

        reread = self.book_session.confirm_candidates(trace, accepts, source=book_reply.source if eligible else None)
        if reread is None:
            return None
        return replace(book_reply, text=reread.span.text, source=reread.passage,
                       passage_trace=replace(trace, reread=reread))

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
        follows_unanswered = self._follows_unanswered
        follows_uncited = self._follows_uncited
        classified = None
        if self.first_reply_classifier is not None and self._first_reply and not self.history:
            answer = self.first_reply_classifier.predict(user_text)
            if answer and len(answer) + len(self.stop_ids) <= limit and set(answer) <= self.tokenizer.ids.keys():
                classified = answer

        asks = user_text.endswith('?') or QUESTION_START.match(user_text)
        if (classified is None and self.book_session is not None and asks
                and self.book_session.outside_names(user_text)):
            # The generator would invent an Oz answer for a name the book never mentions, as in "Who wrote Hamlet?"
            # The turn leaves the conversation and its book subject unchanged.
            return ChatReply(NO_ANSWER, False, False, False, reply_method='abstain')

        book_reply = None
        question_terms = set(search_terms(user_text)) if classified and self.book_session is not None else set()
        if (self.book_session is not None and (user_text.endswith('?') or QUESTION_START.match(user_text))
                and not UNANCHORED_EVENT.fullmatch(user_text)
                and (classified is None or len(question_terms) >= 2)
                and (self._first_reply or follows_unanswered or self.book_session.previous_source is not None
                     or not uses_book_history(user_text)
                     or (follows_uncited and self._resolves_person(user_text)))):
            book_reply = self.book_session.reply(user_text, remember=False, include_trace=True)
            routed = self._route_book_reply(user_text, book_reply, classified, question_terms)
            if routed is not None:
                self._first_reply = False
                if remember:
                    self.book_session.remember_reply(routed)
                    self._follows_unanswered = self._follows_uncited = False
                    if set(routed.text) <= self.tokenizer.ids.keys():
                        self._remember_generation(user_text, routed.text)
                    else:
                        # Keep the cited turn in book history. Generation requires its own vocabulary.
                        self.history.clear()
                trace = routed.passage_trace if self.book_session.trace_passages else None
                return ChatReply(routed.text, False, routed.history_used, False,
                                 reply_method='book', source=routed.source, passage_trace=trace)
            if book_reply.source is not None and classified is None:
                self._first_reply = False
                if remember:
                    self._remember_unanswered(user_text)
                trace = book_reply.passage_trace if self.book_session.trace_passages else None
                return ChatReply(NO_ANSWER, False, book_reply.history_used, False,
                                 reply_method='abstain', passage_trace=trace)

        trace = book_reply.passage_trace if book_reply and self.book_session.trace_passages else None

        if classified is not None:
            if remember:
                self._remember_uncited(user_text, classified, trusted=True)
            self._first_reply = False
            return ChatReply(classified, False, False, False, reply_method='classifier',
                             passage_trace=trace)

        context = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            context, limit, stop_ids=self.stop_ids, sample=False,
        )
        self._first_reply = False
        output = self.tokenizer.decode(generated[0, len(token_ids):].tolist())
        ended = output.endswith(END_OF_TURN)
        text = output.removesuffix(END_OF_TURN).strip()
        if remember:
            if ended and text and END_MARKER not in text:
                self._remember_uncited(user_text, text, trusted=False)
            elif self.book_session is not None:
                self._remember_unanswered(user_text)
        return ChatReply(
            text=text, ended_with_marker=ended, history_used=formatted != prompt,
            history_compacted=compacted, passage_trace=trace,
        )
