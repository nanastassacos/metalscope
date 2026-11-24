"""Asking the served model to grade and reading its grades back (learn/grade.py), as the chat calibration does: no
model, a fake server answering with canned JSON."""
import pytest

from metalscope.learn import grade as G


def test_grade_objects_are_found_in_any_reply_and_scores_read_as_given():
    find = lambda text: [g for v in G._json_values(text) for g in G._entries(v)]
    real = '[{"id": "T01", "style": 3, "coherence": 1}, {"id": "T02", "style": 1, "coherence": 3}]'
    assert [g["id"] for g in find("Grades [below]: " + real + " Note: [T03] was hard.")] == ["T01", "T02"]
    assert find('{"grades": [{"id": "T02", "style": 1}]}') == [{"id": "T02", "style": 1}]
    assert find('{"T03": {"style": 2, "coherence": 1}}') == [{"style": 2, "coherence": 1, "id": "T03"}]
    assert [g["id"] for g in find('[{"id": "T01", "style": 1}, {"id": "T02", "sty')] == ["T01"]    # cut off mid-array
    assert find("I cannot grade these.") == []
    assert [G._score(v) for v in (2, "3", 1.0, 1.5, 4, True, None)] == [2, 3, 1, None, None, None, None]


def test_through_a_server_a_thought_cut_off_in_the_content_is_no_answer():
    class Server:
        def __init__(self, content, finish="stop", reasoning=None):
            self.message, self.finish, self.bodies = dict(content=content, reasoning_content=reasoning), finish, []

        def post(self, path, body, timeout=60, headers=None):
            assert headers == {"x-scope-trace": "off"}       # a job's own request: never traced, so no chat in the tally
            self.bodies.append(body)
            return dict(choices=[dict(message=self.message, finish_reason=self.finish)])
    draft = 'drafting [{"id": "T01", "style": 3, "coherence": 3}]'
    s = Server(draft, finish="length")
    assert G.server_ask(s, "m", thinking=True)("p") == ""                     # no parser, cut mid-thought: no answer
    assert s.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": True} and s.bodies[-1]["max_tokens"] == G.MAX_TOKENS[True]
    assert G.server_ask(s, "m", thinking=False)("p") == draft
    assert G.server_ask(Server("[1]"), "m", thinking=True)("p") == "[1]"     # thinking parsed apart: the content answers
    assert G.server_ask(Server("[1, 2", "length", "hm"), "m", thinking=True)("p") == "[1, 2"
    assert G.server_ask(Server("hm</think>[1]"), "m", thinking=True)("p") == "[1]"
    assert G.server_ask(s, "m")("p") == draft and s.bodies[-1]["temperature"] == 0 and not G.THINKING


def test_thinking_off_on_a_model_with_no_switch_asks_its_lowest_effort():
    """Inkling 2's template ignores ``enable_thinking``: thinking off asks ``reasoning_effort`` "none" as well (its
    "Thinking effort level: 0"); every other model, and thinking on, the switch alone."""
    off = {"enable_thinking": False}
    assert G.thinking_fields("Inkling-2-Mini-oQ4e-mtp", False) == dict(chat_template_kwargs=off, reasoning_effort="none")
    assert G.thinking_fields("Inkling-2-Mini-oQ4e-mtp", True) == dict(chat_template_kwargs={"enable_thinking": True})
    assert G.thinking_fields("Qwen3.8-27B-4bit", False) == dict(chat_template_kwargs=off)
    # It still thinks at "none", so its grader gets the thinking budget; the others the short one.
    assert G.thinks_anyway("Inkling-2-Mini-oQ4e-mtp") and not G.thinks_anyway("Qwen3.8-27B-4bit")

    class S:
        def post(self, path, body, **kw):
            self.body = body
            return {"choices": [{"message": {"content": "[]"}}]}
    s = S()
    G.server_ask(s, "Inkling-2-Mini-oQ4e-mtp")("p")
    assert s.body["max_tokens"] == G.MAX_TOKENS[True] and s.body["reasoning_effort"] == "none"
    G.server_ask(s, "Qwen3.8-27B-4bit")("p")
    assert s.body["max_tokens"] == G.MAX_TOKENS[False]


def test_the_learn_job_has_no_grading_step():
    from metalscope.learn import job as J
    for steps, weights in ((J.LearnJob.STEPS, J.LearnJob.WEIGHTS), (J.LearnJob.DS4_STEPS, J.LearnJob.DS4_WEIGHTS)):
        keys = [k for k, _, _ in steps]
        assert keys[keys.index("ladder") + 1] == "test" and "grades" not in keys
        assert "grading" not in weights and sum(weights.values()) == pytest.approx(1.0)

