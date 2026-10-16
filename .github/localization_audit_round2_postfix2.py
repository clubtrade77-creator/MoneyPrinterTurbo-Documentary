from __future__ import annotations

from pathlib import Path

TESTS = Path("test/services/test_documentary_localization.py")
tests = TESTS.read_text(encoding="utf-8")

name = "test_unresolved_semantic_batch_does_not_retry_whole_language"
start = tests.index(f"def {name}(")
end = tests.find("\ndef ", start + 1)
if end < 0:
    end = len(tests)
block = tests[start:end]

old = '''    full_generation_calls = [
        prompt
        for prompt in generate_calls
        if "Repair ONLY this rejected documentary translation item" not in prompt
    ]
'''
new = '''    full_generation_calls = [
        prompt
        for prompt in generate_calls
        if "Repair ONLY this rejected documentary translation item" not in prompt
        and "EMERGENCY LITERAL FIDELITY REPAIR" not in prompt
    ]
'''
if block.count(old) != 1:
    raise RuntimeError(
        f"{name}: expected one generation-classification anchor, "
        f"found {block.count(old)}"
    )
block = block.replace(old, new, 1)
tests = tests[:start] + block + tests[end:]
TESTS.write_text(tests, encoding="utf-8")
