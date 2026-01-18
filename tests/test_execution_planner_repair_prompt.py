from src.agents.execution_planner import _compress_text_preserve_ends


def test_compress_text_preserve_ends_keeps_tail():
    head = "HEAD"
    tail = "Tail KPI Accuracy"
    middle = "x" * 200
    text = head + middle + tail
    compressed = _compress_text_preserve_ends(text, max_chars=60, head=20, tail=20)
    assert "HEAD" in compressed
    assert "Accuracy" in compressed
    assert "..." in compressed
    assert len(compressed) <= 60 + len("\n...\n")
