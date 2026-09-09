"""Tests for ExecutionContext."""

from __future__ import annotations

from ai_agent_framework.core.context import ExecutionContext


class TestExecutionContext:
    def test_default_session_id_is_generated(self):
        context = ExecutionContext()
        assert context.session_id

    def test_record_appends_to_history(self):
        context = ExecutionContext()
        context.record({"agent_name": "a", "output": "x"})
        context.record({"agent_name": "b", "output": "y"})
        assert len(context.history) == 2
        assert context.history[0]["agent_name"] == "a"

    def test_child_overrides_fields_without_mutating_original(self):
        context = ExecutionContext(user_id="u1", current_agent=None)
        child = context.child(current_agent="agent_x")
        assert child.current_agent == "agent_x"
        assert context.current_agent is None
        assert child.user_id == "u1"

    def test_child_shares_history_reference_semantics_are_independent(self):
        context = ExecutionContext()
        context.record({"a": 1})
        child = context.child(current_agent="x")
        # dataclasses.replace shares the same list object unless overridden;
        # verify it does not silently diverge in a way that hides history.
        assert child.history == context.history
