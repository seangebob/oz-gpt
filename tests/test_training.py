from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from model import training
from model.model import GPTLanguageModel, encode, load_base_model, vocab_size


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


if __name__ == '__main__':
    unittest.main()
