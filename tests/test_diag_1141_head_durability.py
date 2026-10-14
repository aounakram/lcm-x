"""Count-only diagnostic for refused #1141 leaf heads."""
import logging

import pytest

from tests.test_issue_1141_unattributed_leaf import (
    NOOP,
    _engine,
    _history,
    _two_passes,
    summarize,  # noqa: F401 -- shared fixture
)

PREFIX = "LCM leaf head durability (diag #1141): "


@pytest.fixture
def rotated(tmp_path, summarize):  # noqa: F811 -- imported pytest fixture
    engine = _engine(tmp_path)
    history = _history()
    engine._ingest_messages(history)
    assert engine.rotate_active_session(apply=True)["ok"]
    engine.shutdown()
    engine = _engine(tmp_path)
    try:
        yield engine, history
    finally:
        engine.shutdown()


def _diagnostic(caplog):
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith(PREFIX)]
    assert len(lines) == 1
    return dict(field.split("=", 1) for field in lines[0][len(PREFIX):].split()), lines[0]


def test_frontier_advance_logs_durable_head_without_changing_refusal(rotated, summarize, caplog):  # noqa: F811
    engine, history = rotated
    _two_passes(engine, history, summarize, caplog)
    fields, line = _diagnostic(caplog)
    assert int(fields["below"]) == engine._store_complete_frontier()
    assert fields["none"] == fields["below_unaligned"] == "0"
    assert fields["head_rows"] == fields["below"]
    assert fields["view_rows"] == str(len(history))
    assert engine._last_compression_noop_reason == NOOP
    print(line)


def test_unstored_host_head_row(rotated, caplog):
    engine, history = rotated
    engine._current_compress_messages = [{"role": "user", "content": "never stored"}, *history]
    engine._current_compress_store_ids_by_message_id = {}
    with caplog.at_level(logging.WARNING):
        engine._log_leaf_head_durability(engine._store_complete_frontier())
    fields, _ = _diagnostic(caplog)
    assert fields["none"] == "1"
    assert fields["none_roles"] == "user:1"
    assert fields["none_max_chars"] == "12"
    assert fields["none_first_head_index"] == "0"


def test_recorded_host_rewrite_is_below_not_none(rotated, caplog):
    engine, history = rotated
    frontier = engine._store_complete_frontier()
    stored = engine._store.get_session_messages_after(engine._session_id, after_store_id=0)[0]
    host = dict(history[0])
    engine._watch_stored_user_rows([(host, stored, stored["store_id"])])
    host["content"] = host["content"].rstrip() + "\n"
    engine._capture_host_rewrites([host])
    assert host["content"] != stored["content"]
    assert engine._message_replay_identity(host, strip_carrier=False) in engine._stored_row_forms(stored)
    engine._current_compress_messages = [host]
    engine._current_compress_store_ids_by_message_id = {}
    with caplog.at_level(logging.WARNING):
        engine._log_leaf_head_durability(frontier)
    fields, _ = _diagnostic(caplog)
    assert fields["below"] == "1"
    assert fields["none"] == "0"


def test_diagnostic_failure_preserves_compaction_result(rotated, summarize, monkeypatch, caplog):  # noqa: F811
    engine, history = rotated
    diagnostic = engine._log_leaf_head_durability
    with monkeypatch.context() as baseline:
        baseline.setattr(engine, "_log_leaf_head_durability", lambda frontier: None)
        _two_passes(engine, history, summarize, caplog)
    expected = (engine._last_compression_status, engine._last_compression_noop_reason)
    engine._unattributed_leaf_warned.clear()
    engine._no_progress_hold = None
    caplog.clear()

    def fail_forms(row):
        raise ValueError("must never appear in the log")

    def failing_diagnostic(frontier):
        with monkeypatch.context() as failure:
            failure.setattr(engine, "_stored_row_forms", fail_forms)
            diagnostic(frontier)

    monkeypatch.setattr(engine, "_log_leaf_head_durability", failing_diagnostic)
    _two_passes(engine, history, summarize, caplog)
    assert (engine._last_compression_status, engine._last_compression_noop_reason) == expected
    failures = [record.getMessage() for record in caplog.records if "diag #1141) failed:" in record.getMessage()]
    assert failures == ["LCM leaf head durability (diag #1141) failed: ValueError"]
    assert "must never appear" not in caplog.text
