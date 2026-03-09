"""Regression tests for checkpoint scheduling logic."""

from navi_actor.training.ppo_trainer import _should_save_checkpoint


def test_interval_crossing_triggers_save_without_modulo_alignment() -> None:
    """Save should trigger once a full interval elapses even off-boundary."""
    # Prior modulo logic could miss this case: 2048 % 2000 != 0.
    assert _should_save_checkpoint(step_id=2048, last_checkpoint_step=0, checkpoint_every=2000)


def test_interval_not_reached_does_not_trigger() -> None:
    """No save until the configured interval has elapsed."""
    assert not _should_save_checkpoint(step_id=1999, last_checkpoint_step=0, checkpoint_every=2000)


def test_zero_or_negative_interval_disables_saving() -> None:
    """Disabled checkpoint interval should never trigger saves."""
    assert not _should_save_checkpoint(step_id=5000, last_checkpoint_step=0, checkpoint_every=0)
    assert not _should_save_checkpoint(step_id=5000, last_checkpoint_step=0, checkpoint_every=-1)


def test_subsequent_intervals_use_last_checkpoint_step() -> None:
    """Subsequent saves are measured from the last saved step."""
    assert not _should_save_checkpoint(step_id=3900, last_checkpoint_step=2000, checkpoint_every=2000)
    assert _should_save_checkpoint(step_id=4000, last_checkpoint_step=2000, checkpoint_every=2000)
