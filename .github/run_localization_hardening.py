from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATCH_RUNNER = Path(".github/localization_hardening_once.py")
LOCALIZATION = Path("app/services/documentary/localization.py")
TESTS = Path("test/services/test_documentary_localization.py")


def replace_label_block(
    text: str,
    *,
    label: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
) -> str:
    label_pos = text.index(label)
    start = text.rfind(start_marker, 0, label_pos)
    if start < 0:
        raise RuntimeError(f"start marker not found for {label}")
    end = text.index(end_marker, label_pos)
    return text[:start] + replacement + text[end:]


def repair_patch_runner() -> None:
    text = PATCH_RUNNER.read_text(encoding="utf-8")

    text = replace_label_block(
        text,
        label='    "compact payload constraints",\n)',
        start_marker="source = replace_once(",
        end_marker="\nsource = replace_once(",
        replacement='''old_payload = (
    '            "text": unit["text"],\\n'
    '            "next_source": _compact_consistency_text('
)
new_payload = (
    '            "text": unit["text"],\\n'
    '            "required_digit_tokens": sorted(\\n'
    '                _canonical_numeric_tokens(unit["text"]).elements()\\n'
    '            ),\\n'
    '            "required_number_word_spans": _source_number_word_spans(\\n'
    '                unit["text"],\\n'
    '                source_language,\\n'
    '            ),\\n'
    '            "required_protected_literals": sorted(\\n'
    '                _protected_literal_tokens(unit["text"])\\n'
    '            ),\\n'
    '            "next_source": _compact_consistency_text('
)
source = replace_once(
    source,
    old_payload,
    new_payload,
    "compact payload constraints",
)
''',
    )

    text = replace_label_block(
        text,
        label='    "compact prompt constraint explanation",\n)',
        start_marker="source = replace_once(",
        end_marker="\n\n# Cycle 3:",
        replacement='''old_prompt_context = (
    '"consistency_context" contains earlier SOURCE→TRANSLATION examples only when the\\n'
    'current item shares a repeated likely name/term with them. Use it ONLY to keep the\\n'
    'same entity/name/term spelling consistent. Never import facts from those examples.\\n\\n'
    'FIDELITY RULES — these override elegance and brevity:'
)
new_prompt_context = (
    '"consistency_context" contains earlier SOURCE→TRANSLATION examples only when the\\n'
    'current item shares a repeated likely name/term with them. Use it ONLY to keep the\\n'
    'same entity/name/term spelling consistent. Never import facts from those examples.\\n'
    '"required_digit_tokens" are hard constraints: copy every listed digit token\\n'
    'verbatim with identical multiplicity; never spell it out, convert units, round it,\\n'
    'or replace it with an equivalent value. "required_number_word_spans" must preserve\\n'
    'the exact cardinal/ordinal value and scope. "required_protected_literals" must be\\n'
    'copied exactly.\\n\\n'
    'FIDELITY RULES — these override elegance and brevity:'
)
source = replace_once(
    source,
    old_prompt_context,
    new_prompt_context,
    "compact prompt constraint explanation",
)
''',
    )

    text = replace_label_block(
        text,
        label='    "validated compact checkpoint block",\n)',
        start_marker="old_batch_block = dedent(",
        end_marker="\n\n# Cycle 4:",
        replacement='''batch_start = source.index(
    "        def on_partial(partial: dict[int, str]) -> None:\\n"
)
batch_end = source.index("        completed_weight +=", batch_start)
new_batch_block = dedent(
    """
            # Cross-item translation examples can leak facts and modifiers into the
            # current item. Initial translation therefore uses source context only;
            # global name consistency is checked after semantic review.
            enriched_missing = [
                {**unit, "consistency_context": []}
                for unit in missing
            ]
            batch_translations = _translate_compact_batch(
                source_language=source_language,
                target_language=target_language,
                batch=enriched_missing,
                generator=generator,
                on_partial=None,
            )

            # Never persist poisoned partial work. Repair hard factual violations
            # source-only before they enter the resumable checkpoint.
            for unit in enriched_missing:
                item_id = unit["i"]
                hard_issues = _hard_deterministic_review_issues(
                    [
                        {
                            "i": item_id,
                            "source": unit["text"],
                            "translation": batch_translations[item_id],
                        }
                    ]
                )
                if hard_issues:
                    batch_translations[item_id] = _literal_fidelity_repair(
                        generator=generator,
                        source_language=source_language,
                        target_language=target_language,
                        unit=unit,
                        issues=hard_issues,
                    )

            translations.update(batch_translations)
            if checkpoint is not None:
                checkpoint(translations)
    """
)
new_batch_block = "\\n".join(
    f"        {line}" if line else ""
    for line in new_batch_block.splitlines()
) + "\\n"
source = source[:batch_start] + new_batch_block + source[batch_end:]
''',
    )

    text = replace_label_block(
        text,
        label='    "repair context isolation",\n)',
        start_marker="old_constraint_block = dedent(",
        end_marker="\nsource = replace_once(",
        replacement='''repair_function_start = source.index(
    "def _build_single_item_repair_prompt("
)
constraint_start = source.index(
    "    required_literal_tokens = sorted(_protected_literal_tokens(source_text))\\n",
    repair_function_start,
)
constraint_end = source.index('    return f"""', constraint_start)
new_constraint_block = dedent(
    """
        required_literal_tokens = sorted(_protected_literal_tokens(source_text))
        required_number_word_spans = _source_number_word_spans(
            source_text,
            source_language,
        )
        required_reviewer_source_spans = _review_source_span_hints(
            source_text,
            issues,
        )
        hard_constraints = {
            "required_digit_tokens_verbatim": required_digit_tokens,
            "required_number_word_spans": required_number_word_spans,
            "required_protected_literals_verbatim": required_literal_tokens,
            "required_reviewer_source_spans": required_reviewer_source_spans,
        }

        issue_blob = " ".join(str(issue) for issue in issues).casefold()
        context_is_risky = bool(
            re.search(
                r"added|not present|identity|wrong person|"
                r"next translation|previous translation|invented|hallucinat",
                issue_blob,
            )
        )
        if context_is_risky:
            previous, following = "", ""
            translation_under_review = ""
        else:
            previous, following = _repair_context_for_unit(
                source_units,
                unit,
            )
            previous = _compact_consistency_text(previous, max_chars=480)
            following = _compact_consistency_text(following, max_chars=480)
            translation_under_review = current_translation

        consistency_only = bool(issues) and all(
            re.search(
                r"consisten|anchor item|transliteration",
                str(issue),
                re.I,
            )
            for issue in issues
        )
        consistency_context = []
        if consistency_only:
            repeated_keys = _repeated_consistency_keys(source_units)
            consistency_context = _with_consistency_context(
                [unit],
                all_units=source_units,
                translations=translations,
                repeated_keys=repeated_keys,
            )[0].get("consistency_context", [])
    """
)
new_constraint_block = "\\n".join(
    f"    {line}" if line else ""
    for line in new_constraint_block.splitlines()
) + "\\n"
source = (
    source[:constraint_start]
    + new_constraint_block
    + source[constraint_end:]
)
''',
    )

    wiring_start = text.index(
        "# Pass the independent reviewer into both production repair paths."
    )
    wiring_end = text.index("# Tests:", wiring_start)
    wiring_replacement = '''# Pass the independent reviewer into both production repair paths.
source, main_repair_wires = re.subn(
    r'(?m)^(?P<indent>[ \\t]*)issues=batch_issues,\\n'
    r'(?P=indent)status=status,',
    lambda match: (
        f'{match.group("indent")}issues=batch_issues,\\n'
        f'{match.group("indent")}reviewer=reviewer,\\n'
        f'{match.group("indent")}status=status,'
    ),
    source,
    count=1,
)
if main_repair_wires != 1:
    raise RuntimeError(
        f'main repair reviewer wiring: expected one match, found '
        f'{main_repair_wires}'
    )

source, consistency_repair_wires = re.subn(
    r'(?m)^(?P<indent>[ \\t]*)issues=consistency_issues,\\n'
    r'(?P=indent)status=status,',
    lambda match: (
        f'{match.group("indent")}issues=consistency_issues,\\n'
        f'{match.group("indent")}reviewer=reviewer,\\n'
        f'{match.group("indent")}status=status,'
    ),
    source,
    count=1,
)
if consistency_repair_wires != 1:
    raise RuntimeError(
        f'consistency repair reviewer wiring: expected one match, found '
        f'{consistency_repair_wires}'
    )

'''
    text = text[:wiring_start] + wiring_replacement + text[wiring_end:]
    PATCH_RUNNER.write_text(text, encoding="utf-8")


def replace_in_function(
    tests: str,
    name: str,
    old: str,
    new: str,
) -> str:
    start = tests.index(f"def {name}(")
    end = tests.find("\ndef ", start + 1)
    if end < 0:
        end = len(tests)
    block = tests[start:end]
    count = block.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected one test anchor, found {count}")
    block = block.replace(old, new, 1)
    return tests[:start] + block + tests[end:]


def align_legacy_tests() -> None:
    source = LOCALIZATION.read_text(encoding="utf-8")
    tests = TESTS.read_text(encoding="utf-8")

    targeted_old = '''    def generate(prompt: str) -> str:
        if "Repair ONLY this rejected documentary translation item" in prompt:
'''
    targeted_new = '''    def generate(prompt: str) -> str:
        if "EMERGENCY LITERAL FIDELITY REPAIR" in prompt:
            repair_prompts.append(prompt)
            assert "2025" in prompt
            return json.dumps(
                {
                    "items": [
                        {"i": 0, "text": "Офицер подошёл в 2025 году."}
                    ]
                },
                ensure_ascii=False,
            )
        if "Repair ONLY this rejected documentary translation item" in prompt:
'''
    tests = replace_in_function(
        tests,
        "test_numeric_drift_is_targeted_for_repair_instead_of_failing_language",
        targeted_old,
        targeted_new,
    )

    numeric_fail_new = '''    def generate(prompt: str) -> str:
        if "EMERGENCY LITERAL FIDELITY REPAIR" in prompt:
            return json.dumps(
                {"items": [{"i": 0, "text": "Поиски длились 11 часов."}]},
                ensure_ascii=False,
            )
        if "Repair ONLY this rejected documentary translation item" in prompt:
'''
    tests = replace_in_function(
        tests,
        "test_numeric_drift_cannot_be_approved_by_final_materiality",
        targeted_old,
        numeric_fail_new,
    )

    year_fail_new = '''    def generate(prompt: str) -> str:
        if "EMERGENCY LITERAL FIDELITY REPAIR" in prompt:
            return json.dumps(
                {
                    "items": [
                        {"i": 0, "text": "Событие произошло в 2024 году."}
                    ]
                },
                ensure_ascii=False,
            )
        if "Repair ONLY this rejected documentary translation item" in prompt:
'''
    tests = replace_in_function(
        tests,
        "test_hard_year_drift_cannot_be_approved_by_final_materiality",
        targeted_old,
        year_fail_new,
    )

    consistency_old = '''    assert plan.semantic_reviewed is True
    assert seen_consistency
    assert any(
        context["source"] == "Bayharr arrived at the station."
        and "Бейхарр" in context["translation"]
        for context in seen_consistency
    )
'''
    consistency_new = '''    assert plan.semantic_reviewed is True
    # Initial batches are source-isolated; global consistency review handles
    # repeated names after semantic fidelity has been established.
    assert seen_consistency == []
'''
    tests = replace_in_function(
        tests,
        "test_compact_translation_reuses_distant_name_consistency_context",
        consistency_old,
        consistency_new,
    )
    tests = tests.replace(
        "def test_compact_translation_reuses_distant_name_consistency_context(",
        "def test_compact_translation_isolates_distant_name_context(",
        1,
    )

    LOCALIZATION.write_text(source, encoding="utf-8")
    TESTS.write_text(tests, encoding="utf-8")


def main() -> None:
    repair_patch_runner()
    subprocess.run(
        [sys.executable, str(PATCH_RUNNER)],
        check=True,
    )
    align_legacy_tests()


if __name__ == "__main__":
    main()
