"""Copy a legacy Oz checkpoint and compare its character scores after dropping the BOM."""

import argparse
from copy import copy
from hashlib import sha256
import io
import json
from pathlib import Path

import torch

from model.chat_data import format_compact_turn, format_prompt, format_turn, unpack_chat_checkpoint
from model.model import (
    BASE_FORMAT_VERSION, LEGACY_OZ_CHARACTERS, OZ_CHARACTERS, CharacterTokenizer, model_dimensions, model_from_state,
    unpack_base_checkpoint,
)
from model.training import save_checkpoint


CHARACTER_ROWS = ('token_embedding_table.weight', 'lm_head.weight', 'lm_head.bias')
SCORE_ATOL = 1e-5
SCORE_RTOL = 1e-5


def remap_character_rows(weights, source_tokenizer, target_characters):
    rows = torch.tensor(source_tokenizer.encode(''.join(target_characters)), dtype=torch.long)
    mapped = copy(weights)
    for name in CHARACTER_ROWS:
        mapped[name] = weights[name].index_select(0, rows)
    return mapped, rows


def fixed_prompts(chat_tokenizer):
    if chat_tokenizer is None:
        return (
            'Dorothy and the Wizard',
            'CHAPTER 12.\n\nThe Gargoyles',
            '"Where are we?" asked Zeb, as the buggy rolled through the streets.',
        )
    turn = format_compact_turn if chat_tokenizer.compact_history else format_turn
    return (
        format_prompt('Who is Jim?'),
        format_prompt('What did the Wizard use?'),
        turn('Who is Jim?', 'The cab-horse.') + format_prompt('What did he pull?'),
    )


@torch.inference_mode()
def check_scores(original, migrated, source_tokenizer, target_tokenizer, shared_ids, prompts):
    source_model = model_from_state(original, len(source_tokenizer.characters)).eval()
    target_model = model_from_state(migrated, len(target_tokenizer.characters)).eval()
    removed_id = source_tokenizer.ids['\ufeff']
    results = []
    for prompt in prompts:
        source_ids = torch.tensor([source_tokenizer.encode(prompt)])
        target_ids = torch.tensor([target_tokenizer.encode(prompt)])
        source_logits, _ = source_model(source_ids)
        target_logits, _ = target_model(target_ids)
        shared_logits = source_logits.index_select(-1, shared_ids)
        torch.testing.assert_close(target_logits, shared_logits, atol=SCORE_ATOL, rtol=SCORE_RTOL)
        source_probs, target_probs = shared_logits.softmax(dim=-1), target_logits.softmax(dim=-1)
        torch.testing.assert_close(target_probs, source_probs, atol=SCORE_ATOL, rtol=SCORE_RTOL)
        results.append({
            'prompt': prompt,
            'positions_checked': source_ids.size(1),
            'max_logit_difference': (target_logits - shared_logits).abs().max().item(),
            'max_shared_probability_difference': (target_probs - source_probs).abs().max().item(),
            'max_legacy_bom_probability': source_logits.softmax(dim=-1)[..., removed_id].max().item(),
        })
    return results


def migrate_copy(source, output):
    source_bytes = source.read_bytes()
    payload = torch.load(io.BytesIO(source_bytes), map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and 'chat_format' in payload:
        weights, tokenizer = unpack_chat_checkpoint(payload)
        chat_tokenizer = tokenizer
        metadata = {key: payload[key] for key in ('chat_format', 'model_config')}
    else:
        weights, tokenizer = unpack_base_checkpoint(payload)
        model_dimensions(weights)
        chat_tokenizer = None
        metadata = {'base_format': BASE_FORMAT_VERSION}

    if set(tokenizer.characters) != set(LEGACY_OZ_CHARACTERS):
        raise ValueError('Expected the legacy Oz character set, including its byte-order mark.')
    target = CharacterTokenizer(OZ_CHARACTERS)
    migrated, shared_ids = remap_character_rows(weights, tokenizer, target.characters)
    results = check_scores(weights, migrated, tokenizer, target, shared_ids, fixed_prompts(chat_tokenizer))
    save_checkpoint(output, {**metadata, 'characters': list(target.characters), 'model_state_dict': migrated})
    return {
        'source': str(source),
        'source_sha256': sha256(source_bytes).hexdigest(),
        'copy_sha256': sha256(output.read_bytes()).hexdigest(),
        'format': metadata,
        'prompts': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='Legacy Oz base or chat checkpoint')
    parser.add_argument('--output', required=True, type=Path, help='New checkpoint path for the migrated copy')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output already exists. Choose a new path with --output.')
    try:
        result = migrate_copy(args.checkpoint, args.output)
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        parser.exit(1, f'Migration check failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
