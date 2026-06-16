from evaluation.tasks.sandbox_gsm8k import evaluate_samples, _extract_gold_answer, _to_number

GOOD = """def simple_math_problem():
    eggs = 16 - 3 - 4
    return eggs * 2
"""
FENCED = "Here is the code:\n```python\ndef simple_math_problem():\n    return 18\n```\nDone."
WRONG = "def simple_math_problem():\n    return 17\n"
NOFN = "def other():\n    return 18\n"
BADIMPORT = "def simple_math_problem():\n    import os\n    return os.getpid()\n"

def test_gold_parse():
    assert _extract_gold_answer("blah\n#### 18") == 18
    assert _extract_gold_answer("#### 1,024") == 1024
    assert _to_number("the answer is $42.00") == 42

def test_correct():
    assert evaluate_samples(GOOD, "#### 18", 5.0) is True

def test_fenced():
    assert evaluate_samples(FENCED, "#### 18", 5.0) is True

def test_wrong():
    assert evaluate_samples(WRONG, "#### 18", 5.0) is False

def test_missing_fn():
    assert evaluate_samples(NOFN, "#### 18", 5.0) is False

def test_blocked_import():
    # os import inside the fn is blocked -> execution fails -> False
    assert evaluate_samples(BADIMPORT, "#### 18", 5.0) is False

if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
    print("ALL SANDBOX TESTS PASSED")
