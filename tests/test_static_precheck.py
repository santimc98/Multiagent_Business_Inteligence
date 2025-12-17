from src.graph.graph import detect_undefined_names


def test_precheck_allows_dunder_main_block():
    code = """
if __name__ == "__main__":
    print("ok")
"""
    assert detect_undefined_names(code) == []


def test_precheck_flags_missing_function():
    code = """
def main():
    df = load_data_with_dialect()
    return df
"""
    undefined = detect_undefined_names(code)
    assert "load_data_with_dialect" in undefined
