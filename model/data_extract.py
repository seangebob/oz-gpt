import argparse
import json
import lzma
from pathlib import Path
from tempfile import TemporaryDirectory


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / 'extracted-corpus'
TRAIN_FILE = 'output.train.txt'
VAL_FILE = 'output.val.txt'
VOCAB_FILE = 'vocab.txt'
CHUNK_SIZE = 1024 * 1024


def _chunks(stream):
    return iter(lambda: stream.read(CHUNK_SIZE), '')


def _copy_shards(shards, output, vocabulary):
    size = 0
    for shard in shards:
        with lzma.open(shard, 'rt', encoding='utf-8') as source:
            for chunk in _chunks(source):
                size += output.write(chunk)
                vocabulary.update(chunk)
    return size


def _split_shard(shard, train, validation, spool_path, vocabulary):
    # Count decompressed characters while spooling to disk to keep memory bounded.
    with spool_path.open('w+', encoding='utf-8') as spool:
        size = _copy_shards([shard], spool, vocabulary)
        boundary = size * 9 // 10
        remaining = boundary
        spool.seek(0)
        for chunk in _chunks(spool):
            cut = min(len(chunk), remaining)
            train.write(chunk[:cut])
            validation.write(chunk[cut:])
            remaining -= cut
    return boundary, size - boundary


def extract_corpus(input_dir, output_dir):
    """Extract sorted UTF-8 shards into a new directory with train, validation, and vocabulary files."""
    shards = sorted(path for path in input_dir.glob('*.xz') if path.is_file())
    if not shards:
        raise ValueError(f'No .xz files found in {input_dir}.')
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Choose a new output directory. {output_dir} already exists.')

    vocabulary = set()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.extract-', dir=output_dir.parent) as temporary:
        staging = Path(temporary) / 'corpus'
        staging.mkdir()
        train_path, validation_path = staging / TRAIN_FILE, staging / VAL_FILE
        with train_path.open('w', encoding='utf-8') as train, validation_path.open('w', encoding='utf-8') as validation:
            if len(shards) == 1:
                counts = _split_shard(shards[0], train, validation, Path(temporary) / 'source.txt', vocabulary)
            else:
                boundary = len(shards) * 9 // 10
                counts = (
                    _copy_shards(shards[:boundary], train, vocabulary),
                    _copy_shards(shards[boundary:], validation, vocabulary),
                )
        if not all(counts):
            raise ValueError('Corpus extraction requires both splits to contain text.')
        (staging / VOCAB_FILE).write_text(json.dumps(sorted(vocabulary), ensure_ascii=False) + '\n', encoding='utf-8')
        # Publish the entire split together. A concurrent run cannot replace a populated directory.
        staging.rename(output_dir)
    return counts


def main():
    parser = argparse.ArgumentParser(description='Extract UTF-8 .xz text shards into training and validation splits.')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR,
                        help='Directory containing .xz shards (default: model/)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='New directory for the files (default: model/extracted-corpus/)')
    args = parser.parse_args()
    try:
        train_count, validation_count = extract_corpus(args.input_dir, args.output_dir)
    except (OSError, EOFError, ValueError, lzma.LZMAError) as error:
        parser.error(str(error))
    print(f'Extracted {train_count:,} training and {validation_count:,} validation characters to {args.output_dir}.')


if __name__ == '__main__':
    main()
