import json
import lzma
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from model import data_extract, training
from model.model import CUSTOM_BASE_FORMAT, GPTLanguageModel, encode, load_base_model, load_completion_model, vocab_size


class BaseTrainingTests(unittest.TestCase):
    @patch('model.training.block_size', 2)
    def test_corpus_split_and_batch_keep_targets_inside_the_split(self):
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / 'corpus.txt'
            corpus.write_text('a' * 27 + 'b' * 3, encoding='utf-8')
            splits = training.load_splits(corpus)

        self.assertEqual(splits['train'].tolist(), encode('a' * 27))
        self.assertEqual(splits['val'].tolist(), encode('b' * 3))
        with patch('model.training.torch.randint', return_value=torch.tensor([0, 23, 24])) as randint:
            inputs, targets = training.get_batch(torch.arange(27), 3, 'cpu')
        randint.assert_called_once_with(25, (3,), generator=None)
        self.assertEqual(inputs.tolist(), [[0, 1], [23, 24], [24, 25]])
        self.assertEqual(targets.tolist(), [[1, 2], [24, 25], [25, 26]])
        self.assertTrue(targets.is_contiguous())

    def test_evaluation_uses_its_own_repeatable_batches(self):
        model = Mock(training=True, side_effect=lambda inputs, targets: (None, targets.float().mean()))
        data = torch.arange(training.block_size * 3)
        splits = {'train': data, 'val': data + 1000}
        original_rng = torch.get_rng_state()

        first = training.estimate_loss(model, splits, 1, 3, 'cpu')
        second = training.estimate_loss(model, splits, 1, 3, 'cpu')

        self.assertEqual(first, second)
        self.assertEqual(model.call_count, 12)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        model.train.assert_called_with(True)

    def test_checkpoint_writer_never_replaces_existing_output(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            training.save_checkpoint(path, {'weight': torch.tensor([3])})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                training.save_checkpoint(path, {'weight': torch.tensor([7])})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_base_checkpoint_loader_restores_head_and_layer_counts(self):
        for heads, layers in ((1, 1), (4, 2)):
            with self.subTest(heads=heads, layers=layers), TemporaryDirectory() as directory:
                path = Path(directory) / 'checkpoint.pth'
                model = GPTLanguageModel(vocab_size, heads=heads, layers=layers)
                training.save_checkpoint(path, model.state_dict())
                restored = load_base_model(path)
                self.assertEqual(len(restored.blocks), layers)
                self.assertEqual(len(restored.blocks[0].sa.heads), heads)
                self.assertTrue(torch.equal(restored.lm_head.weight, model.lm_head.weight))


class CustomCorpusTests(unittest.TestCase):
    @patch('model.data_extract.CHUNK_SIZE', 17)
    def test_single_shard_splits_unicode_text_and_trainer_reads_both_files_verbatim(self):
        text = 'abc 🪴\n' * 270 + 'é' * 180
        boundary = len(text) * 9 // 10
        with TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'document.xz').write_bytes(lzma.compress(text.encode('utf-8')))
            extracted = source / 'extracted'
            counts = data_extract.extract_corpus(source, extracted)
            splits, tokenizer = training.load_extracted_splits(extracted)

        self.assertEqual(counts, (boundary, len(text) - boundary))
        self.assertEqual(tokenizer.characters, tuple(sorted(set(text))))
        self.assertEqual(tokenizer.decode(splits['train'].tolist()), text[:boundary])
        self.assertEqual(tokenizer.decode(splits['val'].tolist()), text[boundary:])
        self.assertNotIn('é', text[:boundary])

    def test_extraction_splits_whole_shards_by_name(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            (source / '02.xz').write_bytes(lzma.compress(b'validation'))
            (source / '01.xz').write_bytes(lzma.compress(b'training'))
            extracted = source / 'extracted'
            data_extract.extract_corpus(source, extracted)

            self.assertEqual((extracted / data_extract.TRAIN_FILE).read_text(), 'training')
            self.assertEqual((extracted / data_extract.VAL_FILE).read_text(), 'validation')
            with self.assertRaises(FileExistsError):
                data_extract.extract_corpus(source, extracted)
            self.assertEqual((extracted / data_extract.TRAIN_FILE).read_text(), 'training')

    def test_corrupt_shard_leaves_no_partial_corpus_directory(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            good = source / '01.xz'
            bad = source / '02.xz'
            good.write_bytes(lzma.compress(b'training'))
            bad.write_bytes(b'invalid')
            extracted = source / 'extracted'

            with self.assertRaises(lzma.LZMAError):
                data_extract.extract_corpus(source, extracted)

            self.assertEqual(set(source.iterdir()), {good, bad})

    def test_trainer_rejects_validation_text_missing_from_the_saved_vocabulary(self):
        with TemporaryDirectory() as directory:
            corpus = Path(directory)
            (corpus / data_extract.TRAIN_FILE).write_text('a' * 130, encoding='utf-8')
            (corpus / data_extract.VAL_FILE).write_text('é' * 130, encoding='utf-8')
            (corpus / data_extract.VOCAB_FILE).write_text(json.dumps(['a']), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, r'output\.val\.txt: Characters outside'):
                training.load_extracted_splits(corpus)

    def test_completion_restores_custom_token_order_and_oz_fine_tuning_rejects_it(self):
        characters = ['🪴', '\n', 'é', 'a']
        model = GPTLanguageModel(len(characters), heads=2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.pth'
            training.save_checkpoint(path, {
                'base_format': CUSTOM_BASE_FORMAT,
                'characters': characters,
                'model_state_dict': model.state_dict(),
            })

            restored, tokenizer = load_completion_model(path)
            self.assertEqual(tokenizer.encode('a🪴'), [3, 0])
            self.assertEqual(tokenizer.decode([1, 2]), '\né')
            self.assertTrue(torch.equal(restored.lm_head.weight, model.lm_head.weight))
            with self.assertRaisesRegex(ValueError, 'bundled Oz vocabulary'):
                load_base_model(path)


if __name__ == '__main__':
    unittest.main()
