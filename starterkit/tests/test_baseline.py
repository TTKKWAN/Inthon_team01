"""Check pair labels, split isolation, and pooling without downloading any models."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_model import FrozenEncoders, SimilarityMLP
from dataset import EpisodePairs, natural_key, split_indices


def test_one_positive_one_negative_and_no_self_or_duplicate_caption():
    descriptions = ["duplicate", "duplicate", "other", "third"]
    features = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    pairs = EpisodePairs(features, features, descriptions, seed=11)
    for index in range(len(pairs)):
        images, texts, labels = pairs[index]
        negative = pairs.negative_indices[index]
        assert labels.tolist() == [1, 0]
        assert images[:, 0].tolist() == [index, index]
        assert texts[:, 0].tolist() == [index, negative]
        assert index != negative
        assert descriptions[index] != descriptions[negative]


def test_negative_resampling_is_reproducible_and_split_local():
    # These features carry original IDs, so a negative from another split is detectable.
    features = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    indices = [1, 4, 8, 12, 15, 19]
    pairs = EpisodePairs(features[indices], features[indices], [str(i) for i in indices], seed=7)
    first = pairs.negative_indices[:]
    pairs.set_epoch(1)
    assert pairs.negative_indices != first
    for index in range(len(pairs)):
        assert int(pairs[index][1][1, 0]) in indices
    pairs.set_epoch(0)
    assert pairs.negative_indices == first


def test_episode_split_is_disjoint_reproducible_and_has_negative_candidates():
    train, valid = split_indices(list(range(20)), seed=3)
    assert not set(train) & set(valid)
    assert set(train) | set(valid) == set(range(20))
    assert len(train) >= 2 and len(valid) >= 2
    assert (train, valid) == split_indices(list(range(20)), seed=3)
    with pytest.raises(ValueError):
        split_indices([0, 1, 2])
    with pytest.raises(ValueError):
        EpisodePairs(torch.zeros(2, 1), torch.zeros(2, 1), ["same", "same"])


def test_numeric_frame_order():
    assert sorted(["10.jpg", "2.jpg", "1.jpg"], key=natural_key) == ["1.jpg", "2.jpg", "10.jpg"]


def test_frame_mean_counts_every_frame_even_with_short_final_batch():
    encoder = FrozenEncoders.__new__(FrozenEncoders)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.image_dim = 1
    encoder.image_batch_size = 2

    class Frame:
        def __init__(self, value):
            self.value = value

        def convert(self, mode):
            return self

    encoder.image_processor = lambda images, **kwargs: {
        "pixel_values": torch.tensor([[image.value] for image in images]),
    }
    encoder.image_model = lambda pixel_values: SimpleNamespace(
        last_hidden_state=pixel_values.unsqueeze(1),
    )
    assert encoder.encode_images([Frame(1), Frame(3), Frame(8)]).item() == 4


def test_text_pooling_excludes_padding():
    encoder = FrozenEncoders.__new__(FrozenEncoders)
    encoder.device = torch.device("cpu")
    encoder.text_batch_size = 8
    encoder.max_text_length = 384

    class Inputs(dict):
        def to(self, device):
            return self

    encoder.tokenizer = lambda *args, **kwargs: Inputs({
        "input_ids": torch.tensor([[2.0, 4.0, 100.0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    })
    encoder.text_model = lambda input_ids, attention_mask: SimpleNamespace(
        last_hidden_state=input_ids.unsqueeze(-1),
    )
    assert encoder.encode_texts(["placeholder"]).item() == 3


def test_mlp_scalar_probabilities_and_bce_update():
    torch.manual_seed(3)
    model = SimilarityMLP(image_dim=2, text_dim=3, hidden_dim=4, dropout=0)
    images, texts = torch.randn(6, 2), torch.randn(6, 3)
    scores = model(images, texts)
    assert scores.shape == (6,)
    assert ((0 <= scores) & (scores <= 1)).all()
    before = model.network[-1].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.logits(images, texts), torch.tensor([1.0, 0.0] * 3),
    )
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.network[-1].weight)
