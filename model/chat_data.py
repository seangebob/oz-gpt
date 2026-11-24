import json
from pathlib import Path
from textwrap import shorten

import torch

from model.model import CharacterTokenizer, block_size, dropout, model_dimensions, n_embd


CHAT_CHECKPOINT = Path(__file__).with_name('model-chat-retrained.pth')
CHAT_TRAIN_OUTPUT = Path(__file__).with_name('model-chat-experiment.pth')
CHAT_FORMAT_VERSION = 1
COMPACT_CHAT_FORMAT_VERSION = 2
LEGACY_HISTORY_PACKING = 'shorten_latest_reply_if_needed'
COMPACT_HISTORY_PACKING = 'compact_question_and_reply'
REPLY_RESERVE = 32
END_MARKER = '[END]'
END_OF_TURN = '\n' + END_MARKER
IGNORE_TARGET = -100  # PyTorch cross-entropy's default ignore_index
MODEL_CONFIG = {
    'block_size': block_size,
    'n_embd': n_embd,
    'dropout': dropout,
}


class ChatTokenizer(CharacterTokenizer):
    def __init__(self, characters, *, compact_history=False):
        super().__init__(characters)
        self.compact_history = compact_history
        self.history_packing = COMPACT_HISTORY_PACKING if compact_history else LEGACY_HISTORY_PACKING
        self.encode(format_prompt('') + END_OF_TURN)
        if compact_history:
            self.encode(format_compact_turn('', ''))


def format_prompt(user_text):
    return f'User: {user_text}\nAssistant: '


def format_turn(user_text, reply):
    return format_prompt(user_text) + reply + END_OF_TURN + '\n'


def format_compact_turn(user_text, reply):
    return f'Q: {user_text}\nA: {reply}\n'


def reference_matches(reply, expected):
    reply_key = ' '.join(reply.casefold().split()).rstrip('.!?').rstrip()
    expected_key = ' '.join(expected.casefold().split()).rstrip('.!?').rstrip()
    return bool(expected_key) and reply_key == expected_key


def with_history(prompt, history, max_prompt_chars):
    """Prepend the newest consecutive complete turns that fit the prompt budget."""
    for turn in reversed(history):
        if len(turn) + len(prompt) > max_prompt_chars:
            break
        prompt = turn + prompt
    return prompt


def pack_history(prompt, history, budget, *, compact_history=False):
    formatter = format_compact_turn if compact_history else format_turn
    formatted = with_history(prompt, [formatter(*turn) for turn in history], budget)
    if formatted != prompt or not history:
        return formatted, False

    user_text, previous_reply = history[-1]
    if compact_history:
        available = budget - len(prompt) - len(format_compact_turn('', ''))
        first_reply_word = next(iter(previous_reply.split()), '')
        if available <= len(first_reply_word):
            return prompt, False
        # Leave room for both parts. Short answers give their unused space to the question.
        reply_width = min(len(previous_reply), max(available // 3, len(first_reply_word)))
        question = shorten(user_text, width=available - reply_width, placeholder='', break_on_hyphens=False)
        reply = shorten(previous_reply, width=available - len(question), placeholder='', break_on_hyphens=False)
        if question and reply:
            return format_compact_turn(question, reply) + prompt, True
        return prompt, False

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


def _conversation_examples(conversation, tokenizer, *, mask_history_replies=False):
    history = []
    for turn, (user_text, assistant_text) in enumerate(conversation_turns(conversation), start=1):
        reply = assistant_text + END_OF_TURN
        prompt = format_prompt(user_text)
        # Shifting by one yields at most block_size inputs and targets.
        if len(prompt) + len(reply) > block_size + 1:
            raise ValueError(f'Turn {turn} exceeds the {block_size}-character context. Shorten the prompt or reply.')
        reserve = min(block_size + 1 - len(prompt), max(REPLY_RESERVE, len(reply)))
        prompt, _ = pack_history(prompt, history, block_size + 1 - reserve,
                                 compact_history=tokenizer.compact_history)

        tokens = tokenizer.encode(prompt + reply)
        # The final prompt character predicts the first character of the reply.
        targets = [IGNORE_TARGET] * (len(prompt) - 1) + tokens[len(prompt):]
        yield tokens[:-1], targets
        history.append((user_text, '?' if mask_history_replies else assistant_text))


def read_conversations(source, path):
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            conversation = json.loads(line)
        except ValueError as error:
            raise ValueError(f'{path}:{line_number}: {error}') from error
        yield line_number, conversation


def encode_conversations(conversations, tokenizer, path, *, mask_history_replies=False):
    examples = []
    for line_number, conversation in conversations:
        try:
            examples.extend(_conversation_examples(conversation, tokenizer, mask_history_replies=mask_history_replies))
        except ValueError as error:
            raise ValueError(f'{path}:{line_number}: {error}') from error
    if not examples:
        raise ValueError(f'{path} contains no usable conversations.')
    return examples


def load_examples(path, tokenizer, *, mask_history_replies=False):
    with path.open(encoding='utf-8') as source:
        return encode_conversations(read_conversations(source, path), tokenizer, path,
                                    mask_history_replies=mask_history_replies)


def collate_examples(examples):
    length = max(len(inputs) for inputs, _ in examples)
    inputs = torch.zeros((len(examples), length), dtype=torch.long)
    targets = torch.full_like(inputs, IGNORE_TARGET)
    for row, (input_ids, target_ids) in enumerate(examples):
        inputs[row, :len(input_ids)] = torch.tensor(input_ids)
        targets[row, :len(target_ids)] = torch.tensor(target_ids)
    return inputs, targets


def load_chat_checkpoint(path):
    return unpack_chat_checkpoint(torch.load(path, map_location='cpu', weights_only=True))


def unpack_chat_checkpoint(state):
    if not isinstance(state, dict) or state.get('chat_format') not in (CHAT_FORMAT_VERSION, COMPACT_CHAT_FORMAT_VERSION):
        raise ValueError('Expected a chat checkpoint created by model.chat_training.')
    weights = state.get('model_state_dict')
    heads, layers = model_dimensions(weights)
    if state.get('model_config') != {**MODEL_CONFIG, 'n_head': heads, 'n_layer': layers}:
        raise ValueError('The checkpoint architecture differs from model/model.py.')
    tokenizer = ChatTokenizer(state.get('characters'), compact_history=state['chat_format'] == COMPACT_CHAT_FORMAT_VERSION)
    return weights, tokenizer
