from __future__ import annotations

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
SEARCH_LIMIT = 8
MIN_ANSWER_MARGIN = 2.0
CHAPTER_HEADING_WEIGHT = 0.28

CHAPTER_HEADING = re.compile(r'CHAPTER (\d+)\.', re.IGNORECASE)
FOLLOW_UP = re.compile(
    r'\b(?:he|she|it|they|him|her|them|his|hers|its|their|there|that|those|too|else)\b'
    r'|^(?:how many|what kind|which one)\W*$', re.IGNORECASE,
)
PRONOUN = re.compile(r'\b(?:he|she|it|they|him|them|his|hers|its|their)\b', re.IGNORECASE)
NAMED_SUBJECT = re.compile(r"(?i:who|what) (?i:is|was) ([A-Z][\w'-]+(?: [A-Z][\w'-]+){0,3})[?.!]*")
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


def followup_question(question, previous_question, previous_answer):
    named = NAMED_SUBJECT.fullmatch(previous_question)
    topic = named[1] if named else previous_answer.strip(' .?!"')
    if topic:
        if re.fullmatch(r'what kind\W*', question, re.IGNORECASE):
            return f'What kind of thing is {topic}?'
        if re.fullmatch(r'how many\W*', question, re.IGNORECASE):
            return f'How many {topic}?'

        def replace_pronoun(match):
            return f"{topic}'s" if match[0].lower() in {'his', 'hers', 'its', 'their'} else topic

        expanded = PRONOUN.sub(replace_pronoun, question)
        if expanded != question and not re.search(r'\bit\b', question, re.IGNORECASE):
            return expanded
        question = expanded
    event = re.sub(r'^\s*(?:who|what|where|when|why|how)\b\s*', '', previous_question, flags=re.IGNORECASE)
    return f'{question.rstrip("?")} (context: {event.rstrip("?.!")})?'


@dataclass(frozen=True)
class Passage:
    section: str
    start_line: int
    end_line: int
    text: str
    heading: str = ''


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    margin: float


@dataclass(frozen=True)
class BookReply:
    text: str
    source: Passage | None
    history_used: bool


def book_passages(text):
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
        for offset in range(0, len(words), PASSAGE_STEP):
            window = words[offset:offset + PASSAGE_WORDS]
            yield Passage(label, window[0][1], window[-1][1], ' '.join(word for word, _ in window), heading)
            if offset + PASSAGE_WORDS >= len(words):
                break


class BookIndex:
    def __init__(self, passages):
        self.passages = list(passages)
        if not self.passages:
            raise ValueError('The book contains no searchable passages.')
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

    def search(self, question, previous_question='', previous_answer=''):
        query = dict.fromkeys(search_terms(question), 1.0)
        for word in dict.fromkeys(search_terms(previous_question)):
            query[word] = query.get(word, 0.0) + 0.6
        for word in dict.fromkeys(search_terms(previous_answer)):
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
        return sorted(scores, key=lambda item: item[1], reverse=True)[:SEARCH_LIMIT]


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
            answers.append(AnswerSpan(passage.text[begin:finish].strip(), margin))
        return answers


class BookSession:
    def __init__(self, index, reader):
        self.index = index
        self.reader = reader
        self.reset()

    @classmethod
    def from_book(cls, path=BOOK_PATH):
        book = path.read_bytes()
        session = cls(BookIndex(book_passages(book.decode('utf-8-sig'))), ExtractiveReader())
        session.book_sha256 = hashlib.sha256(book).hexdigest()
        return session

    def reset(self):
        self.previous_question = ''
        self.previous_answer = ''

    def remember(self, question, answer):
        self.previous_question = question
        self.previous_answer = answer

    def reply(self, question, *, remember=True):
        question = question.strip()
        if not question:
            raise ValueError('Empty prompt — please enter text')
        history_used = bool(self.previous_question and FOLLOW_UP.search(question))
        earlier_question = self.previous_question if history_used else ''
        earlier_answer = self.previous_answer if history_used else ''
        reader_question = question
        if history_used:
            reader_question = followup_question(question, earlier_question, earlier_answer)
        candidates = self.index.search(reader_question, earlier_question, earlier_answer)
        reply = BookReply(NO_ANSWER, None, history_used)
        if candidates:
            spans = self.reader.read(reader_question, [passage for passage, _ in candidates])
            strongest_search_score = candidates[0][1]
            ranked = [
                (span.margin + 2 * score / strongest_search_score, span.text, passage)
                for (passage, score), span in zip(candidates, spans)
                if span.margin >= MIN_ANSWER_MARGIN and span.text and span.text in passage.text
            ]
            if ranked:
                _, text, passage = max(ranked, key=lambda item: item[0])
                reply = BookReply(text, passage, history_used)
        if remember:
            self.remember(question, reply.text if reply.source else '')
        return reply
