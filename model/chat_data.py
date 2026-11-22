import json
from pathlib import Path
from textwrap import shorten

import torch

from model.model import block_size, dropout, model_dimensions, n_embd


CHAT_CHECKPOINT = Path(__file__).with_name('model-chat-retrained.pth')
CHAT_TRAIN_OUTPUT = Path(__file__).with_name('model-chat-experiment.pth')
CHAT_FORMAT_VERSION = 1
REPLY_RESERVE = 32
END_MARKER = '[END]'
END_OF_TURN = '\n' + END_MARKER
IGNORE_TARGET = -100  # PyTorch cross-entropy's default ignore_index
MODEL_CONFIG = {
    'block_size': block_size,
    'n_embd': n_embd,
    'dropout': dropout,
}


class ChatTokenizer:
    def __init__(self, characters):
        if (
            not isinstance(characters, (list, tuple))
            or not characters
            or any(not isinstance(char, str) or len(char) != 1 for char in characters)
            or len(set(characters)) != len(characters)
        ):
            raise ValueError('Expected an ordered vocabulary of distinct characters.')
        self.characters = tuple(characters)
        self.ids = {char: index for index, char in enumerate(characters)}
        self.encode(format_prompt('') + END_OF_TURN)

    def encode(self, text):
        unknown = sorted(set(text).difference(self.ids))
        if unknown:
            raise ValueError(f'Characters outside the checkpoint vocabulary: {unknown!r}')
        return [self.ids[char] for char in text]

    def decode(self, ids):
        return ''.join(self.characters[index] for index in ids)


def format_prompt(user_text):
    return f'User: {user_text}\nAssistant: '


def format_turn(user_text, reply):
    return format_prompt(user_text) + reply + END_OF_TURN + '\n'


def with_history(prompt, history, max_prompt_chars):
    """Prepend the newest consecutive complete turns that fit the prompt budget."""
    for turn in reversed(history):
        if len(turn) + len(prompt) > max_prompt_chars:
            break
        prompt = turn + prompt
    return prompt


def pack_history(prompt, history, budget):
    formatted = with_history(prompt, [format_turn(*turn) for turn in history], budget)
    if formatted != prompt or not history:
        return formatted, False

    user_text, previous_reply = history[-1]
    reply_budget = budget - len(prompt) - len(format_turn(user_text, ''))
    if reply_budget <= 0:
        return prompt, False
    shortened = shorten(previous_reply, width=reply_budget, placeholder='', break_on_hyphens=False)
    if not shortened:
        return prompt, False
    return format_turn(user_text, shortened) + prompt, True


def _message_text(message, role):
    if not isinstance(message, dict) or message.get('role') != role:
        raise ValueError(f'Expected a message with role {role!r}.')
    text = message.get('content')
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f'Expected nonempty content for {role!r}.')
    if END_MARKER in text:
        raise ValueError('Message content cannot contain the reserved [END] marker.')
    return text.strip()


def conversation_turns(conversation):
    messages = conversation.get('messages') if isinstance(conversation, dict) else None
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError('Expected one or more complete user/assistant pairs in messages.')
    for index in range(0, len(messages), 2):
        yield _message_text(messages[index], 'user'), _message_text(messages[index + 1], 'assistant')


def _conversation_examples(conversation, tokenizer):
    history = []
    for turn, (user_text, assistant_text) in enumerate(conversation_turns(conversation), start=1):
        reply = assistant_text + END_OF_TURN
        prompt = format_prompt(user_text)
        # Shifting by one yields at most block_size inputs and targets.
        if len(prompt) + len(reply) > block_size + 1:
            raise ValueError(f'Turn {turn} exceeds the {block_size}-character context. Shorten the prompt or reply.')
        reserve = min(block_size + 1 - len(prompt), max(REPLY_RESERVE, len(reply)))
        prompt, _ = pack_history(prompt, history, block_size + 1 - reserve)

        tokens = tokenizer.encode(prompt + reply)
        # The final prompt character predicts the first character of the reply.
        targets = [IGNORE_TARGET] * (len(prompt) - 1) + tokens[len(prompt):]
        yield tokens[:-1], targets
        history.append((user_text, assistant_text))


def read_conversations(source, path):
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            conversation = json.loads(line)
        except ValueError as error:
            raise ValueError(f'{path}:{line_number}: {error}') from error
        yield line_number, conversation


def encode_conversations(conversations, tokenizer, path):
    examples = []
    for line_number, conversation in conversations:
        try:
            examples.extend(_conversation_examples(conversation, tokenizer))
        except ValueError as error:
            raise ValueError(f'{path}:{line_number}: {error}') from error
    if not examples:
        raise ValueError(f'{path} contains no usable conversations.')
    return examples


def load_examples(path, tokenizer):
    with path.open(encoding='utf-8') as source:
        return encode_conversations(read_conversations(source, path), tokenizer, path)


def collate_examples(examples):
    length = max(len(inputs) for inputs, _ in examples)
    inputs = torch.zeros((len(examples), length), dtype=torch.long)
    targets = torch.full_like(inputs, IGNORE_TARGET)
    for row, (input_ids, target_ids) in enumerate(examples):
        inputs[row, :len(input_ids)] = torch.tensor(input_ids)
        targets[row, :len(target_ids)] = torch.tensor(target_ids)
    return inputs, targets


def load_chat_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or state.get('chat_format') != CHAT_FORMAT_VERSION:
        raise ValueError('Expected a chat checkpoint created by model.chat_training.')
    weights = state.get('model_state_dict')
    heads, layers = model_dimensions(weights)
    if state.get('model_config') != {**MODEL_CONFIG, 'n_head': heads, 'n_layer': layers}:
        raise ValueError('The checkpoint architecture differs from model/model.py.')
    tokenizer = ChatTokenizer(state.get('characters'))
    return weights, tokenizer
