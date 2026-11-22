from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
import hashlib
from math import log1p
from pathlib import Path
import re

import torch


BOOK_PATH = Path(__file__).with_name('wizard_of_oz.txt')
READER_MODEL = 'deepset/minilm-uncased-squad2'
READER_REVISION = '934656cdda79824eabf503ed56e15c01ddbdbe3f'
NO_ANSWER = "I couldn't find an answer supported by the book."
PASSAGE_WORDS = 165
PASSAGE_STEP = 100
PASSAGE_CONTEXT_WORDS = 24
SEARCH_LIMIT = 8
EXPANDED_SEARCH_LIMIT = 24
FOLLOWUP_RECHECK_MARGIN = 5.0
EXPANDED_MATCH_FRACTION = 0.75
MIN_ANSWER_MARGIN = 2.0
MIN_SINGLE_PASSAGE_MARGIN = 3.25
ANSWER_AGREEMENT_BONUS = 0.5
CHAPTER_HEADING_WEIGHT = 0.28

CHAPTER_HEADING = re.compile(r'CHAPTER (\d+)\.', re.IGNORECASE)
SENTENCE_END = re.compile(r"""[.!?]["'’”_\])]*$""")
ABBREVIATION = re.compile(r'(?:[A-Z]|Mr|Mrs|Ms|Dr|St)\.$')
SHORT_FOLLOW_UP = re.compile(
    r'^(?:why|how many|what kind|which one)\W*$|^why (?:that|this) name\W*$'
    r'|\b(?:there|that|those|too|else)\W*$', re.IGNORECASE,
)
PRONOUN = re.compile(r'\b(?:he|she|it|they|him|her|them|his|hers|its|their)\b', re.IGNORECASE)
BOOK_NAMES = (
    r'Braided Man|Cowardly Lion|Hungry Tiger|Tin Woodman|Woggle-Bug|Nick Chopper|'
    r'Uncle Henry|Uncle Bill|Aunt Em|Jellia Jamb|Dorothy|Eureka|Gwig|Jim|Ozma|'
    r'Sawhorse|Scarecrow|Tik-tok|Wizard|Zeb(?:ediah)?'
)
BOOK_NAME = re.compile(rf'\b(?:{BOOK_NAMES})\b', re.IGNORECASE)
NAME = rf"(?:(?i:the) )?(?:(?i:{BOOK_NAMES})\b|[A-Z][\w'-]+(?: [A-Z][\w'-]+){{0,3}})"
NAMED_SUBJECT = re.compile(rf'(?i:who|what) (?i:is|was) ({NAME})')
QUESTION_ACTOR = re.compile(
    rf"(?i:who|what|where|when|why|how(?: long| much| high| far| often)?) "
    rf"(?i:did|does|was|were|is|has|had|could|would) ({NAME})(?= |'s\b|$)",
)
ABOUT_SUBJECT = re.compile(rf'(?i:what about) ({NAME})\W*')
NAME_ANSWER = re.compile(r"(?:(?i:the) )?[A-Z][a-z][\w'-]*(?: [A-Z][a-z][\w'-]*){0,3}")
PAST_ACTION = re.compile(rf'(?i:what did) (?P<actor>{NAME}) (?P<verb>[a-z]+)(?P<rest> .*)?')
CHOICE_SUBJECT = re.compile(
    r"which ((?!of\b)[\w'-]+(?: [\w'-]+){0,3}?) ((?:was|were|went|fell|came|had|got|[\w-]{3,}ed)\b.*)",
    re.IGNORECASE,
)
PLURAL_OBJECT = re.compile(r"\b(?i:the|these|those) ((?:[a-z][\w'-]* ){0,4}[a-z][\w'-]*s)$")
RETURN_QUESTION = re.compile(rf'(?i:where (?:did|does)) ({NAME}) (?i:return)')
ITEM_LOCATION = re.compile(r'where (?:is|was) (?:it|that)(?: (really|actually))?\W*', re.IGNORECASE)
MISSING_EVENT = re.compile(r'went missing|(?:was|were) (?:lost|missing)|got lost|disappeared', re.IGNORECASE)
PRETENDED_SOURCE = re.compile(r' to (take|pull|get|bring)\b.*\b(?:out of|from) .+', re.IGNORECASE)
RESCUE_QUESTION = re.compile(rf'what (rescued|saved|freed) ({NAME})', re.IGNORECASE)
TRAVEL_QUESTION = re.compile(rf'(?i:how did) ({NAME}) (?i:reach|get to|arrive in) (.+)')
MEANS_ANSWER = re.compile(r"(?i:a|an|the) [\w'-]+(?: [\w'-]+){0,5}|[A-Z][\w'-]+(?: [A-Z][\w'-]+){1,4}")
ACTION_ACTOR = re.compile(r'who ([a-z]+)( .+)?', re.IGNORECASE)
ACTION_REASON = re.compile(r'why (?:was|were) (?:he|she|they) ([a-z]+)ing\W*', re.IGNORECASE)
LOST_ITEMS = re.compile(rf'(?i:where did) ({NAME}) (?i:lose them)\W*')
PAST_TENSE = {
    'bring': 'brought', 'buy': 'bought', 'catch': 'caught', 'come': 'came', 'do': 'did', 'drink': 'drank',
    'eat': 'ate', 'fall': 'fell', 'find': 'found', 'fly': 'flew', 'get': 'got', 'give': 'gave', 'go': 'went',
    'hide': 'hid', 'hold': 'held', 'leave': 'left', 'make': 'made', 'meet': 'met', 'put': 'put', 'run': 'ran',
    'say': 'said', 'see': 'saw', 'set': 'set', 'shoe': 'shod', 'speak': 'spoke', 'take': 'took', 'wear': 'wore',
}
BASE_VERB = {past: verb for verb, past in PAST_TENSE.items()}
AFTER_OBJECT = set('about after again along and around as at back before behind by down for from go in into leave '
                   'near off on or out over through to under up with'.split())
STOP_WORDS = set(
    'a an and are as at be been but by can could did do does for from had has have he her him his '
    'how i in is it its me of on or she should that the their them then there these they this those '
    'to too us was were what when where which who whom why will with would you your'.split()
)


def search_terms(text):
    for word in re.findall(r'[a-z0-9]+', text.casefold()):
        if len(word) < 2 or word in STOP_WORDS:
            continue
        for suffix in ('ing', 'ed', 's'):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        yield word


def _past_tense(verb):
    return PAST_TENSE.get(verb, re.sub(r'([^aeiou])y$', r'\1i', verb) + ('d' if verb.endswith('e') else 'ed'))


def _possessive(question, match):
    pronoun = match[0].lower()
    if pronoun != 'her':
        return pronoun in {'his', 'hers', 'its', 'their'}
    following = re.match(r'\s+([a-z]+)\b', question[match.end():], re.IGNORECASE)
    return following is not None and following[1].lower() not in AFTER_OBJECT


def _uses_history(question):
    if SHORT_FOLLOW_UP.search(question) or ABOUT_SUBJECT.fullmatch(question):
        return True
    pronouns = list(PRONOUN.finditer(question))
    if not pronouns:
        return False
    main_question = question.rsplit(', ', 1)[-1].rstrip('?.!')
    if all(_possessive(question, match) for match in pronouns):
        return not (QUESTION_ACTOR.match(main_question) or BOOK_NAME.search(question[:pronouns[0].start()]))
    return True


def followup_question(question, previous_question, previous_answer):
    previous_question = previous_question.rstrip('?.!')
    named = NAMED_SUBJECT.fullmatch(previous_question)
    actor = named[1] if named else ''
    name = previous_answer.strip(' .?!"')
    if not actor and re.match(r'who\b', previous_question, re.IGNORECASE) and NAME_ANSWER.fullmatch(name):
        actor = name
    if not actor and (subject := QUESTION_ACTOR.match(previous_question)):
        actor = subject[1].removesuffix("'s")
    if actor and re.fullmatch(r'why (?:that|this) name\W*', question, re.IGNORECASE):
        return f'Why was {actor} given that name?'

    if about := ABOUT_SUBJECT.fullmatch(question):
        prior_subject = actor if actor and actor in previous_question else ''
        if not prior_subject and (mentioned := BOOK_NAME.search(previous_question)):
            prior_subject = mentioned[0]
        if prior_subject:
            return re.sub(rf'\b{re.escape(prior_subject)}\b', lambda _: about[1], previous_question) + '?'

    if re.fullmatch(r'why\W*', question, re.IGNORECASE):
        if prior_action := ACTION_ACTOR.fullmatch(previous_question):
            verb, rest = prior_action.groups()
            if base := BASE_VERB.get(verb.lower()):
                return f"Why did {actor or 'someone'} {base}{rest or ''}?"
        if re.match(r'(?:what|where|when|how) (?:did|does|was|were|is|could)\b', previous_question, re.IGNORECASE):
            return 'Why ' + previous_question.split(' ', 1)[1] + '?'

    returning = RETURN_QUESTION.fullmatch(previous_question)
    if returning and re.fullmatch(r'who (?:went|came) (?:too|along)\W*', question, re.IGNORECASE):
        actor = returning[1]
        return f'Who went with {actor} when {actor} returned?'

    if re.fullmatch(r'who (?:used|operated|wielded) (?:it|that)\W*', question, re.IGNORECASE):
        rescue = RESCUE_QUESTION.fullmatch(previous_question)
        if rescue:
            verb, person = rescue[1].lower(), rescue[2]
            if MEANS_ANSWER.fullmatch(name):
                means = re.sub(r'^(A|An|The) ', lambda match: match[0].lower(), name)
                return f'Who used {means} to {verb[:-1]} {person}?'
            return f'Who {verb} {person}?'
        travel = TRAVEL_QUESTION.fullmatch(previous_question)
        if travel:
            return f'Who brought {travel[1]} to {travel[2]}?'

    reason = ACTION_REASON.fullmatch(question)
    if reason and (prior_action := ACTION_ACTOR.fullmatch(previous_question)):
        stem = reason[1].lower()
        if prior_action[1].lower() in {_past_tense(stem), _past_tense(stem + 'e')}:
            return f"Why was someone {stem}ing{prior_action[2] or ''}?"

    item = ''
    action = PAST_ACTION.fullmatch(previous_question)
    choice = CHOICE_SUBJECT.fullmatch(previous_question)
    location = ITEM_LOCATION.fullmatch(question)
    if action:
        actor, verb, rest = action.group('actor', 'verb', 'rest')
        verb = verb.lower()
        if location and location[1] and verb == 'pretend' and (source := PRETENDED_SOURCE.fullmatch(rest or '')):
            return f'Where did {actor} really {source[1].lower()} the item from?'
        if verb in {'show', 'reveal', 'display', 'produce'} and re.fullmatch(r'where was it\W*', question, re.IGNORECASE):
            return f'Where did {actor} produce the item from?'
        item = f'the item {actor} {_past_tense(verb)}{rest or ""}'
    elif choice:
        if location and MISSING_EVENT.fullmatch(choice[2]):
            return f'Where did the {choice[1]} really end up?'
        item = f'the {choice[1]} that {choice[2]}'
    elif re.match(r'what\b', previous_question, re.IGNORECASE) and not re.match(
        r'what (?:do|did|does|is|are|was|were|can|could)\b', previous_question, re.IGNORECASE,
    ):
        item = f'the thing that {previous_question[5:]}'

    plural = PLURAL_OBJECT.search(previous_question)
    plural_item = plural[1] if plural else ''
    if plural_item and (lost := LOST_ITEMS.fullmatch(question)):
        return f'Where did the {plural_item} go when {lost[1]} lost them?'
    if (item or actor) and re.fullmatch(r'what kind\W*', question, re.IGNORECASE):
        return f'What kind of thing is {item or actor}?'
    if plural_item and re.fullmatch(r'how many\W*', question, re.IGNORECASE):
        return f'How many {plural_item}?'

    def replace_pronoun(match):
        pronoun = match[0].lower()
        if pronoun in {'they', 'them', 'their'}:
            referent = f'the {plural_item}' if plural_item else ''
        elif pronoun in {'it', 'its'}:
            referent = item or actor
        else:
            referent = actor
        if not referent:
            return match[0]
        return f"{referent}'s" if _possessive(question, match) else referent

    expanded = PRONOUN.sub(replace_pronoun, question)
    if expanded != question:
        return expanded
    event = re.sub(r'^\s*(?:who|what|where|when|why|how)\b\s*', '', previous_question, flags=re.IGNORECASE)
    return f'{question.rstrip("?")} (context: {event})?'


@dataclass(frozen=True)
class Passage:
    section: str
    start_line: int
    end_line: int
    text: str
    heading: str = ''


@dataclass(frozen=True)
class PassageEdge:
    text: str
    line: int
    paragraph_terms: frozenset[str]


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    margin: float
    context_end: int | None = None


@dataclass(frozen=True)
class PassageTrace:
    query: str
    previous_question: str
    previous_answer: str
    candidates: tuple[tuple[Passage, float], ...]
    spans: tuple[AnswerSpan, ...]


@dataclass(frozen=True)
class BookReply:
    text: str
    source: Passage | None
    history_used: bool
    passage_trace: PassageTrace | None = None


def book_passages(text):
    for passage, _, _ in _passage_windows(text):
        yield passage


def _passage_windows(text):
    lines = text.splitlines()
    chapters = []
    for index, line in enumerate(lines):
        match = CHAPTER_HEADING.fullmatch(line.strip())
        if match:
            chapters.append((index, f'Chapter {match[1]}'))

    front_end = chapters[0][0] if chapters else len(lines)
    contents = next((i for i, line in enumerate(lines[:front_end]) if line.strip() == 'LIST OF CHAPTERS'), front_end)
    preface = next((i for i, line in enumerate(lines[:contents]) if line.strip() == 'To My Readers'), contents)
    sections = [(0, preface, 'Title page', ''), (preface, contents, 'Preface', '')]
    for (start, label), (end, _) in zip(chapters, chapters[1:] + [(len(lines), '')]):
        heading = next((line.strip() for line in lines[start + 1:end] if line.strip()), '')
        sections.append((start, end, label, heading if heading.isupper() else ''))

    for start, end, label, heading in sections:
        words = [(word, number + 1) for number in range(start, end) for word in lines[number].split()]
        sentences = [0]
        paragraphs = [0]
        for position, (word, line) in enumerate(words[:-1], start=1):
            next_word, next_line = words[position]
            if next_line > line + 1:
                paragraphs.append(position)
                sentences.append(position)
            elif (SENTENCE_END.search(word) and not ABBREVIATION.fullmatch(word)
                  and not next_word.lstrip("\"'`_“‘([")[:1].islower()):
                sentences.append(position)
        sentences.append(len(words))
        paragraphs.append(len(words))
        for offset in range(0, len(words), PASSAGE_STEP):
            limit = min(offset + PASSAGE_WORDS, len(words))
            window = words[offset:limit]
            passage = Passage(label, window[0][1], window[-1][1], ' '.join(word for word, _ in window), heading)
            first = sentences[bisect_right(sentences, offset) - 1]
            last = sentences[bisect_left(sentences, limit)]
            prefix = suffix = None
            if 0 < offset - first <= PASSAGE_CONTEXT_WORDS:
                paragraph_end = min(paragraphs[bisect_right(paragraphs, offset)], limit)
                prefix = PassageEdge(
                    ' '.join(word for word, _ in words[first:offset]),
                    words[first][1],
                    frozenset(search_terms(' '.join(word for word, _ in words[offset:paragraph_end]))),
                )
            if 0 < last - limit <= PASSAGE_CONTEXT_WORDS:
                paragraph_start = max(offset, paragraphs[bisect_left(paragraphs, limit) - 1])
                suffix = PassageEdge(
                    ' '.join(word for word, _ in words[limit:last]),
                    words[last - 1][1],
                    frozenset(search_terms(' '.join(word for word, _ in words[paragraph_start:limit]))),
                )
            yield passage, prefix, suffix
            if limit == len(words):
                break


class BookIndex:
    def __init__(self, passages, *, passage_edges=None):
        self.passages = list(passages)
        if not self.passages:
            raise ValueError('The book contains no searchable passages.')
        self.passage_edges = passage_edges or {}
        self.frequencies = [Counter(search_terms(passage.text)) for passage in self.passages]
        self.span_lengths = [sum(terms.values()) for terms in self.frequencies]
        self.heading_terms = {
            passage.section: set(search_terms(passage.heading)) for passage in self.passages if passage.heading
        }
        self.average_length = max(sum(self.span_lengths) / len(self.span_lengths), 1)
        document_counts = Counter(word for terms in self.frequencies for word in terms)
        total = len(self.passages)
        self.weights = {
            word: log1p((total - count + 0.5) / (count + 0.5))
            for word, count in document_counts.items()
        }

    @classmethod
    def from_text(cls, text):
        passages, edges = [], {}
        for passage, prefix, suffix in _passage_windows(text):
            passages.append(passage)
            if prefix or suffix:
                edges[passage] = (prefix, suffix)
        return cls(passages, passage_edges=edges)

    def _with_sentence_context(self, passage, question_terms):
        prefix, suffix = self.passage_edges.get(passage, (None, None))
        # Keep extra sentence text where the surrounding paragraph matches the question.
        prefix = prefix if prefix and prefix.paragraph_terms & question_terms else None
        suffix = suffix if suffix and suffix.paragraph_terms & question_terms else None
        if not (prefix or suffix):
            return passage
        text = ' '.join(part for part in (prefix.text if prefix else '', passage.text, suffix.text if suffix else '') if part)
        return Passage(
            passage.section,
            prefix.line if prefix else passage.start_line,
            suffix.line if suffix else passage.end_line,
            text,
            passage.heading,
        )

    def search(self, question, previous_question='', *, limit=SEARCH_LIMIT):
        query = dict.fromkeys(search_terms(question), 1.0)
        question_terms = set(query)
        for word in dict.fromkeys(search_terms(previous_question)):
            query[word] = query.get(word, 0.0) + 0.6

        heading_scores = {
            section: CHAPTER_HEADING_WEIGHT * sum(
                weight * self.weights.get(word, 0.0) for word, weight in query.items() if word in terms
            )
            for section, terms in self.heading_terms.items()
        }
        scores = []
        for passage, terms, length in zip(self.passages, self.frequencies, self.span_lengths):
            adjustment = 1.2 * (0.25 + 0.75 * length / self.average_length)
            score = sum(
                weight * self.weights[word] * (2.2 * terms[word]) / (terms[word] + adjustment)
                for word, weight in query.items() if word in terms
            )
            if score > 0:
                scores.append((passage, score + heading_scores.get(passage.section, 0.0)))
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:limit]
        return [(self._with_sentence_context(passage, question_terms), score) for passage, score in ranked]


class ExtractiveReader:
    def __init__(self):
        try:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Install the optional book lookup dependency with pip install "transformers>=4.45,<6".') from error
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL, revision=READER_REVISION, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            READER_MODEL, revision=READER_REVISION, use_safetensors=True,
        ).eval()

    @torch.inference_mode()
    def read(self, question, passages):
        question_ids = self.tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=80)
        question = self.tokenizer.decode(question_ids)
        batch = self.tokenizer(
            [question] * len(passages), [passage.text for passage in passages],
            max_length=384, padding=True, truncation='only_second',
            return_offsets_mapping=True, return_tensors='pt',
        )
        offsets = batch.pop('offset_mapping')
        result = self.model(**batch)
        answers = []
        for row, passage in enumerate(passages):
            context = batch.sequence_ids(row)
            first = context.index(1)
            last = len(context) - context[::-1].index(1)
            positions = torch.arange(last - first)
            lengths = positions[None, :] - positions[:, None]
            start = result.start_logits[row]
            end = result.end_logits[row]
            scores = start[first:last, None] + end[None, first:last]
            scores.masked_fill_((lengths < 0) | (lengths >= 24), float('-inf'))
            score, best = scores.flatten().max(dim=0)
            i, j = divmod(best.item(), last - first)
            begin = offsets[row, first + i, 0].item()
            finish = offsets[row, first + j, 1].item()
            margin = (score - start[0] - end[0]).item()
            context_end = offsets[row, last - 1, 1].item()
            answers.append(AnswerSpan(passage.text[begin:finish].strip(), margin, context_end))
        return answers


def _select_answer(candidates, spans):
    strongest_search_score = candidates[0][1]
    answers = {}
    for (passage, score), span in zip(candidates, spans):
        if not (span.margin >= MIN_ANSWER_MARGIN and span.text and span.text in passage.text[:span.context_end]):
            continue
        key = ' '.join(span.text.casefold().split()).strip(' .!?,')
        key = re.sub(r'^(?:a|an|the) ', '', key)
        if key:
            strength = span.margin + 2 * score / strongest_search_score
            answers.setdefault(key, []).append((strength, span, passage))

    ranked = []
    for matches in answers.values():
        strength, span, passage = max(matches, key=lambda match: match[0])
        agrees = any(other_passage != passage for _, _, other_passage in matches)
        if agrees:
            strength += ANSWER_AGREEMENT_BONUS
        elif span.margin < MIN_SINGLE_PASSAGE_MARGIN:
            continue
        ranked.append((strength, span.text, passage))

    _, text, source = max(ranked, key=lambda match: match[0], default=(0.0, NO_ANSWER, None))
    return text, source


def _select_expanded_answer(question, candidates, spans, initial_count):
    question_terms = set(search_terms(question))
    eligible = candidates[:initial_count]
    eligible_spans = spans[:initial_count]
    initial_texts = [
        passage.text[:span.context_end].casefold()
        for (passage, _), span in zip(eligible, eligible_spans)
    ]
    for candidate, span in zip(candidates[initial_count:], spans[initial_count:]):
        passage, _ = candidate
        matched = len(question_terms.intersection(search_terms(passage.text[:span.context_end])))
        enough_terms = matched >= EXPANDED_MATCH_FRACTION * len(question_terms)
        # Allow a specific phrase already mentioned in the original shortlist.
        repeated_phrase = (
            matched >= 2 and len(set(search_terms(span.text))) >= 2
            and any(span.text.casefold() in text for text in initial_texts)
        )
        if enough_terms or repeated_phrase:
            eligible.append(candidate)
            eligible_spans.append(span)
    return _select_answer(eligible, eligible_spans)


class BookSession:
    def __init__(self, index, reader, *, trace_passages=False):
        self.index = index
        self.reader = reader
        self.trace_passages = trace_passages
        self.reset()

    @classmethod
    def from_book(cls, path=BOOK_PATH, *, trace_passages=False):
        book = path.read_bytes()
        index = BookIndex.from_text(book.decode('utf-8-sig'))
        session = cls(index, ExtractiveReader(), trace_passages=trace_passages)
        session.book_sha256 = hashlib.sha256(book).hexdigest()
        return session

    def reset(self):
        self.previous_question = ''
        self.previous_answer = ''

    def remember(self, question, answer):
        self._remember_resolved(self._followup(question) or question, answer)

    def _followup(self, question):
        if self.previous_question and _uses_history(question):
            return followup_question(question, self.previous_question, self.previous_answer)
        return None

    def _remember_resolved(self, question, answer):
        # Avoid accumulating repeated context annotations across turns.
        resolved, annotation, _ = question.partition(' (context: ')
        self.previous_question = resolved + '?' if annotation else question
        self.previous_answer = answer

    def reply(self, question, *, remember=True):
        question = question.strip()
        if not question:
            raise ValueError('Empty prompt — please enter text')
        rewritten = self._followup(question)
        history_used = rewritten is not None
        earlier_question = self.previous_question if history_used else ''
        earlier_answer = self.previous_answer if history_used else ''
        reader_question = rewritten or question
        candidates = self.index.search(reader_question, earlier_question)
        text, source = NO_ANSWER, None
        spans = []
        if candidates:
            spans = self.reader.read(reader_question, [passage for passage, _ in candidates])
            text, source = _select_answer(candidates, spans)
            weak_followup = history_used and any(
                passage == source and span.margin < FOLLOWUP_RECHECK_MARGIN
                for (passage, _), span in zip(candidates, spans)
            )
            if source is None or weak_followup:
                expanded = self.index.search(reader_question, earlier_question, limit=EXPANDED_SEARCH_LIMIT)
                initial_count = len(candidates)
                extra_passages = [passage for passage, _ in expanded[initial_count:]]
                if extra_passages:
                    spans += self.reader.read(reader_question, extra_passages)
                    candidates = expanded
                    text, source = _select_expanded_answer(reader_question, candidates, spans, initial_count)
        trace = None
        if self.trace_passages:
            trace = PassageTrace(reader_question, earlier_question, earlier_answer, tuple(candidates), tuple(spans))
        if remember:
            self._remember_resolved(reader_question, text if source else '')
        return BookReply(text, source, history_used, trace)
