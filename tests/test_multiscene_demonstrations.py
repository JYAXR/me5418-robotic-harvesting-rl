"""Verify episode boundaries, detached data and honest multi-scene return targets."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.multiscene_demonstrations import (
    build_multiscene_batch,
    slice_transition_batch,
)


def trajectory(rewards, position=0.0):
    count = len(rewards)
    states = torch.arange((count + 1) * 2, dtype=torch.float32).reshape(count + 1, 2) + position
    terminated = torch.zeros(count, dtype=torch.bool)
    if count:
        terminated[-1] = True
    return TransitionBatch(states[:-1].clone(), torch.arange(count) % 2,
                           torch.tensor(rewards, dtype=torch.float32), states[1:].clone(),
                           terminated, torch.zeros(count, dtype=torch.bool))


def test_complete_returns_never_cross_episode_resets():
    first, second = trajectory([1, 2]), trajectory([100, 200, 400], 50)
    batch, returns, episodes = build_multiscene_batch([first, second], [10000, 10001], 0.5)
    assert returns.tolist() == [2.0, 2.0, 300.0, 400.0, 400.0]
    assert batch.terminated.tolist() == [False, True, False, False, True]
    assert [(row["seed"], row["start"], row["stop"], row["transitions"])
            for row in episodes] == [(10000, 0, 2, 2), (10001, 2, 5, 3)]
    assert [row["undiscounted_return"] for row in episodes] == [3.0, 700.0]
    assert [row["initial_discounted_return"] for row in episodes] == [2.0, 300.0]
    assert [row["minimum_return"] for row in episodes] == [2.0, 300.0]
    assert [row["maximum_return"] for row in episodes] == [2.0, 400.0]
    assert len(batch.actions) == len(first.actions) + len(second.actions)
    for original, metadata in zip((first, second), episodes):
        restored = slice_transition_batch(batch, metadata["start"], metadata["stop"])
        for name in original.__dataclass_fields__:
            assert torch.equal(getattr(restored, name), getattr(original, name))


def test_combined_data_returns_and_slices_are_detached_and_independent():
    first = trajectory([1, 2])
    first.observations.requires_grad_(True)
    first.rewards.requires_grad_(True)
    batch, returns, _ = build_multiscene_batch([first], [3], 0.99)
    copied = slice_transition_batch(batch, 0, 2)
    for name in first.__dataclass_fields__:
        original, combined, sliced = getattr(first, name), getattr(batch, name), getattr(copied, name)
        assert not combined.requires_grad and not sliced.requires_grad
        assert len({original.data_ptr(), combined.data_ptr(), sliced.data_ptr()}) == 3
    assert not returns.requires_grad and returns.grad_fn is None
    assert first.rewards.grad is None
    copied.rewards.fill_(999)
    batch.observations.fill_(999)
    assert batch.rewards.tolist() == [1.0, 2.0]
    assert first.observations[0, 0].item() == 0
    assert returns.tolist() == pytest.approx([2.98, 2.0])


@pytest.mark.parametrize("seeds", [[1], [1, 1], [1, -1], [True, 2], [1, 2.5], [1, "2"]])
def test_reject_mismatched_or_invalid_scene_seeds(seeds):
    with pytest.raises(ValueError):
        build_multiscene_batch([trajectory([1]), trajectory([2])], seeds, 0.99)


@pytest.mark.parametrize("case", ["partial", "interior_terminal", "truncated", "shuffled",
                                  "empty", "observation_width", "reward_dtype", "state_dtype",
                                  "action_dtype", "negative_action", "nonfinite_reward"])
def test_reject_invalid_episodes_without_changing_inputs(case):
    first, second = trajectory([1, 2]), trajectory([3, 4], 10)
    original = first.rewards.clone()
    if case == "partial":
        second = replace(second, terminated=torch.tensor([False, False]))
    elif case == "interior_terminal":
        second = replace(second, terminated=torch.tensor([True, True]))
    elif case == "truncated":
        second = replace(second, truncated=torch.tensor([False, True]))
    elif case == "shuffled":
        second = TransitionBatch(*(getattr(second, name).flip(0) for name in second.__dataclass_fields__))
    elif case == "empty":
        second = trajectory([])
    elif case == "observation_width":
        second = replace(second, observations=second.observations[:, :1],
                         next_observations=second.next_observations[:, :1])
    elif case == "reward_dtype":
        second = replace(second, rewards=second.rewards.double())
    elif case == "state_dtype":
        second = replace(second, observations=second.observations.double(),
                         next_observations=second.next_observations.double())
    elif case == "action_dtype":
        second = replace(second, actions=second.actions.float())
    elif case == "negative_action":
        second = replace(second, actions=torch.tensor([-1, 0]))
    else:
        second = replace(second, rewards=torch.tensor([1.0, float("inf")]))
    with pytest.raises(ValueError):
        build_multiscene_batch([first, second], [1, 2], 0.99)
    assert torch.equal(first.rewards, original)


def test_empty_collection_and_non_batch_input_are_rejected():
    with pytest.raises(ValueError):
        build_multiscene_batch([], [], 0.99)
    with pytest.raises(TypeError):
        build_multiscene_batch([None], [1], 0.99)


@pytest.mark.parametrize("start, stop", [(-1, 1), (0, 0), (2, 1), (0, 3), (False, 1), (0, 1.5)])
def test_invalid_slice_cannot_silently_drop_or_wrap_transitions(start, stop):
    with pytest.raises(ValueError):
        slice_transition_batch(trajectory([1, 2]), start, stop)


def test_slice_rejects_misaligned_fields():
    batch = trajectory([1, 2])
    with pytest.raises(ValueError, match="transition count"):
        slice_transition_batch(replace(batch, rewards=batch.rewards[:1]), 0, 1)
