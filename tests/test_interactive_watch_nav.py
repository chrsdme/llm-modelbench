from llm_modelbench.interactive_nav import InteractiveState


STATUS = {"completed_models": [
    {"model": "alpha", "quality_avg": 80, "tps_avg": 10, "failures": 2, "weak": 1},
    {"model": "bravo", "quality_avg": 90, "tps_avg": 20, "failures": 0, "weak": 0},
]}


def test_interactive_state_refresh_search_sort_selection_and_detail():
    state = InteractiveState()
    state.refresh_rows(STATUS)
    assert len(state.rows) == 2
    state.handle_key("j")
    assert state.selected == 1
    state.handle_key("k")
    assert state.selected == 0
    state.handle_key("s")
    assert state.sort_key == "quality_avg"
    assert [row["model"] for row in state.visible_rows()] == ["bravo", "alpha"]
    state.handle_key("S")
    assert [row["model"] for row in state.visible_rows()] == ["alpha", "bravo"]
    state.handle_key("s")
    assert state.sort_key == "tps_avg"
    state.handle_key("s")
    assert state.sort_key == "failures"
    state.handle_key("/")
    state.handle_key("b")
    state.handle_key("\n")
    assert [row["model"] for row in state.visible_rows()] == ["bravo"]
    state.handle_key("\n")
    assert state.detail_open is True
    state.handle_key("\x1b")
    assert state.detail_open is False
