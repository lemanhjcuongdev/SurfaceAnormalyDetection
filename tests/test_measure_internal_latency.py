"""State loading and validity rules of the internal latency measurement (no model, no images)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_internal_latency import is_valid, load_state, parity  # noqa: E402


class Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.empty(0))


class Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("memory_bank", torch.empty(0))
        self.inner = Inner()
        self.layer = torch.nn.Linear(2, 2)


def test_load_state_resizes_empty_buffers():
    source = Toy()
    source.memory_bank = torch.ones(4, 3)
    source.inner.mean = torch.full((2, 2), 7.0)
    target = Toy()
    load_state(target, source.state_dict())
    assert target.memory_bank.shape == (4, 3)
    assert torch.equal(target.inner.mean, source.inner.mean)
    assert torch.equal(target.layer.weight, source.layer.weight)


def test_load_state_rejects_unknown_keys():
    state = Toy().state_dict()
    state["unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        load_state(Toy(), state)


def test_parity_reports_relative_difference_and_changed_decisions():
    stored = np.array([10.0, 20.0, 40.0])
    measured = np.array([10.1, 20.0, 40.0])
    check = parity(measured, stored, [(measured, stored, 10.05)])
    assert check["max_relative_difference"] == pytest.approx(0.01)
    assert check["decisions_changed"] == 1


@pytest.mark.parametrize(
    ("check", "records", "ac", "expected"),
    [
        ({"max_relative_difference": 0.0005, "decisions_changed": 0}, 297, True, True),
        ({"max_relative_difference": 0.002, "decisions_changed": 0}, 297, True, False),
        ({"max_relative_difference": 0.0005, "decisions_changed": 1}, 297, True, False),
        ({"max_relative_difference": 0.0005, "decisions_changed": 0}, 296, True, False),
        ({"max_relative_difference": 0.0005, "decisions_changed": 0}, 297, False, False),
    ],
)
def test_is_valid_needs_every_condition(check, records, ac, expected):
    assert is_valid(check, 0.001, records, 297, ac) is expected
