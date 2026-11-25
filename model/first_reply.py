import argparse
from collections import Counter
from math import log1p
from pathlib import Path
import re

import torch
from torch.nn.functional import normalize, one_hot

from model.book_lookup import search_terms
from model.chat_data import END_MARKER, conversation_turns, read_conversations
from model.chat_training import DATA_DIR
from model.training import save_checkpoint


FIRST_REPLY_CHECKPOINT = Path(__file__).with_name('model-first-replies-v2.pth')
FORMAT_VERSION = 2
SUPPORTED_FORMATS = (1, FORMAT_VERSION)
FEATURE_WEIGHTS = (0.45, 0.15, 0.40)
RIDGE = 1.0
MIN_SCORE = 0.24
MIN_MARGIN = 0.06
WH_WORDS = frozenset('who whom whose what which when where why how'.split())
QUESTION_WORDS = WH_WORDS | frozenset(
    'is are was were am do does did can could would will should shall may might must has have had'.split()
)


def question_word(question):
    """Return the word that makes this a question, or '' for a bare name, greeting or request."""
    words = re.findall(r'[a-z]+', question.casefold())
    # A conversational opening does not hide the question: "And where is Jim?", "Okay, so where is Jim?"
    asked = next((word for word in words if word in WH_WORDS), None)
    if asked:
        return asked
    for word in words:
        if word in QUESTION_WORDS:
            return word
        if any(search_terms(word)):
            # A key word came first, so a later "is" belongs to a statement such as "Jim is here".
            return ''
    return ''


def question_features(text):
    words = re.findall(r'[a-z0-9]+', text.casefold())
    padded = [f'_{word}_' for word in words]
    return (
        set(words),
        {' '.join(pair) for pair in zip(words, words[1:])},
        {word[i:i + n] for word in padded for n in (2, 3, 4) for i in range(len(word) - n + 1)},
    )


def features_for_channel(examples, index, idf):
    matrix = torch.zeros((len(examples), len(index)), dtype=idf.dtype)
    for row, terms in enumerate(examples):
        positions = [index[term] for term in terms if term in index]
        matrix[row, positions] = idf[positions]
    return normalize(matrix, dim=1)


class FirstReplyClassifier:
    def __init__(self, state):
        if not isinstance(state, dict) or state.get('first_reply_format') not in SUPPORTED_FORMATS:
            raise ValueError('Expected a checkpoint created by model.first_reply.')
        answers, channels = state.get('answers'), state.get('channels')
        if (not isinstance(answers, list) or len(answers) < 2
                or any(not isinstance(text, str) or not text.strip() or END_MARKER in text for text in answers)
                or len(set(answers)) != len(answers)):
            raise ValueError('Expected at least two distinct, nonempty answers in the classifier.')
        if not isinstance(channels, list) or len(channels) != len(FEATURE_WEIGHTS):
            raise ValueError('The checkpoint must contain three feature groups.')

        self.answers = tuple(answers)
        self.channels = []
        for channel in channels:
            if not isinstance(channel, dict):
                raise ValueError('Invalid classifier feature weights.')
            terms, idf, coefficients = channel.get('terms'), channel.get('idf'), channel.get('coefficients')
            if (not isinstance(terms, list) or any(not isinstance(term, str) for term in terms)
                    or len(set(terms)) != len(terms) or not isinstance(idf, torch.Tensor)
                    or not isinstance(coefficients, torch.Tensor) or idf.shape != (len(terms),)
                    or coefficients.shape != (len(terms), len(answers)) or idf.dtype != torch.float32
                    or coefficients.dtype != torch.float32 or not torch.isfinite(idf).all()
                    or not torch.isfinite(coefficients).all()):
                raise ValueError('Invalid classifier feature weights.')
            self.channels.append(({term: i for i, term in enumerate(terms)}, idf, coefficients))

        # Format 1 checkpoints do not record which question words taught each answer.
        self.answer_terms = self.answer_question_words = None
        if state['first_reply_format'] >= 2:
            answer_terms = state.get('answer_terms')
            if (not isinstance(answer_terms, list) or len(answer_terms) != len(answers)
                    or any(not isinstance(terms, list) or any(not isinstance(term, str) for term in terms)
                           for terms in answer_terms)):
                raise ValueError('Invalid classifier answer terms.')
            self.answer_terms = tuple(frozenset(terms) for terms in answer_terms)
            answer_question_words = state.get('answer_question_words')
            if (not isinstance(answer_question_words, list) or len(answer_question_words) != len(answers)
                    or any(not isinstance(words, list) or any(word not in QUESTION_WORDS for word in words)
                           for words in answer_question_words)):
                raise ValueError('Invalid classifier answer question words.')
            self.answer_question_words = tuple(frozenset(words) for words in answer_question_words)

    def _question_supports(self, question, answer):
        if self.answer_terms is None:
            return True
        words = set(search_terms(question))
        learned = self.answer_terms[answer]
        # One familiar name should not make an answer fit a question about something else.
        if len(words) >= 2:
            return len(words & learned) >= 2
        if not words:
            return len(re.findall(r'[a-z0-9]+', question.casefold())) > 1
        word = next(iter(words))
        possessive = re.search(rf"\b{re.escape(word)}\w*['’]s\b", self.answers[answer].casefold())
        # A single key word must also be asked the way this answer was taught: "Who is Jim?" but not "Where is Jim?"
        asked = question_word(question)
        return word in learned and possessive is None and (not asked or asked in self.answer_question_words[answer])

    @torch.inference_mode()
    def predict(self, question):
        scores = torch.zeros(len(self.answers))
        for terms, (index, idf, coefficients) in zip(question_features(question), self.channels):
            positions = sorted(index[term] for term in terms if term in index)
            if positions:
                values = idf[positions]
                scores += normalize(values, dim=0) @ coefficients[positions]
        values, ranks = scores.topk(2)
        answer = ranks[0].item()
        if values[0] >= MIN_SCORE and values[0] - values[1] >= MIN_MARGIN and self._question_supports(question, answer):
            return self.answers[answer]
        return None


def fit_first_replies(examples):
    classes = dict.fromkeys(reply for _, reply in examples)
    if len(classes) < 2:
        raise ValueError('Use training conversations with at least two different first replies.')
    answer_ids = {reply: i for i, reply in enumerate(classes)}
    labels = one_hot(torch.tensor([answer_ids[reply] for _, reply in examples]), len(classes)).double()
    features = [question_features(question) for question, _ in examples]
    kernel = torch.eye(len(examples), dtype=torch.float64) * RIDGE
    groups = []
    for weight, channel in zip(FEATURE_WEIGHTS, zip(*features)):
        counts = Counter(term for terms in channel for term in terms)
        terms = sorted(counts)
        index = {term: i for i, term in enumerate(terms)}
        idf = torch.tensor([log1p(len(examples) / (1 + counts[term])) for term in terms], dtype=torch.float64)
        matrix = features_for_channel(channel, index, idf)
        kernel += weight * (matrix @ matrix.T)
        groups.append((weight, terms, idf, matrix))

    dual_weights = torch.linalg.solve(kernel, labels)
    answer_terms = [set() for _ in classes]
    answer_question_words = [set() for _ in classes]
    for question, reply in examples:
        terms = set(search_terms(question))
        answer_terms[answer_ids[reply]].update(terms)
        if len(terms) == 1 and question_word(question):
            answer_question_words[answer_ids[reply]].add(question_word(question))
    return {
        'first_reply_format': FORMAT_VERSION,
        'answers': list(classes),
        'answer_terms': [sorted(terms) for terms in answer_terms],
        'answer_question_words': [sorted(words) for words in answer_question_words],
        'channels': [
            {'terms': terms, 'idf': idf.float(), 'coefficients': (weight * matrix.T @ dual_weights).float()}
            for weight, terms, idf, matrix in groups
        ],
    }


def load_first_reply_classifier(path):
    return FirstReplyClassifier(torch.load(path, map_location='cpu', weights_only=True))


def main():
    parser = argparse.ArgumentParser(description='Train a classifier on the first replies in Oz chat conversations.')
    parser.add_argument('--train-data', type=Path, default=DATA_DIR / 'oz_chat_train.jsonl')
    parser.add_argument('--output', required=True, type=Path, help='Path for a new classifier checkpoint')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error('The output path already exists. Choose a new path with --output.')

    try:
        examples = []
        with args.train_data.open(encoding='utf-8') as source:
            for line, conversation in read_conversations(source, args.train_data):
                try:
                    examples.append(list(conversation_turns(conversation))[0])
                except ValueError as error:
                    raise ValueError(f'{args.train_data}:{line}: {error}') from error
        state = fit_first_replies(examples)
        save_checkpoint(args.output, state)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'First-reply training error: {error}\n')

    print(f'Trained on {len(examples)} first replies across {len(state["answers"])} answer classes.')
    print(f'Saved classifier to {args.output}.')


if __name__ == '__main__':
    main()
