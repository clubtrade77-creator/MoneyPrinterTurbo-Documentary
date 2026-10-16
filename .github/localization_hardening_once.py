from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

LOCALIZATION_PATH = Path("app/services/documentary/localization.py")
TESTS_PATH = Path("test/services/test_documentary_localization.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_function(text: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"^def {re.escape(name)}\([\s\S]*?(?=^def |\Z)",
        flags=re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"function not found: {name}")
    return text[: match.start()] + dedent(replacement).strip() + "\n\n\n" + text[match.end() :]


source = LOCALIZATION_PATH.read_text(encoding="utf-8")
tests = TESTS_PATH.read_text(encoding="utf-8")

# Cycle 1: stale checkpoints and old review approvals must not survive policy changes.
source = replace_once(
    source,
    "CURRENT_LOCALIZATION_REVIEW_VERSION = 5\n"
    "_LOCALIZATION_WORK_VERSION = 1\n"
    "_LOCALIZATION_WORK_REVIEW_POLICY_VERSION = 5",
    "CURRENT_LOCALIZATION_REVIEW_VERSION = 6\n"
    "# Bump when translation/checkpoint semantics change so an interrupted\n"
    "# run cannot silently reuse text produced by an older prompt policy.\n"
    "_LOCALIZATION_WORK_VERSION = 2\n"
    "# Bump whenever deterministic or semantic review policy changes.\n"
    "_LOCALIZATION_WORK_REVIEW_POLICY_VERSION = 6",
    "localization policy versions",
)

# Cycle 2: detect number words before generation and expose hard constraints per item.
number_helper = dedent(
    r'''
    _EN_NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
        "seventy", "eighty", "ninety", "hundred", "thousand", "million",
        "billion", "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
        "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
        "seventeenth", "eighteenth", "nineteenth", "twentieth",
        "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
        "eightieth", "ninetieth", "hundredth", "thousandth", "millionth",
        "billionth", "half", "quarter", "dozen", "score", "once", "twice",
    }


    def _source_number_word_spans(text: str, source_language: str) -> list[str]:
        """Return contiguous English number/ordinal phrases for prompt constraints."""
        if _normalize_language(source_language).split("-", 1)[0] != "en":
            return []

        value = str(text or "")
        matches = list(re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", value))
        spans: list[str] = []
        seen: set[str] = set()
        index = 0
        while index < len(matches):
            token = matches[index].group(0).casefold()
            parts = token.split("-")
            if not all(part in _EN_NUMBER_WORDS for part in parts):
                index += 1
                continue

            start = matches[index].start()
            end = matches[index].end()
            cursor = index + 1
            while cursor < len(matches):
                separator = value[end : matches[cursor].start()]
                if not re.fullmatch(r"[\s,-]*", separator):
                    break
                next_token = matches[cursor].group(0).casefold()
                next_parts = next_token.split("-")
                if all(part in _EN_NUMBER_WORDS for part in next_parts):
                    end = matches[cursor].end()
                    cursor += 1
                    continue
                if (
                    next_token == "and"
                    and cursor + 1 < len(matches)
                    and re.fullmatch(
                        r"[\s,-]*",
                        value[
                            matches[cursor].end() : matches[cursor + 1].start()
                        ],
                    )
                    and all(
                        part in _EN_NUMBER_WORDS
                        for part in matches[cursor + 1]
                        .group(0)
                        .casefold()
                        .split("-")
                    )
                ):
                    end = matches[cursor + 1].end()
                    cursor += 2
                    continue
                break

            span = value[start:end].strip()
            key = span.casefold()
            if span and key not in seen:
                seen.add(key)
                spans.append(span)
            index = cursor
        return spans


    '''
)
marker = "def _with_consistency_context(\n"
if source.count(marker) != 1:
    raise RuntimeError("number helper insertion marker mismatch")
source = source.replace(marker, number_helper + marker, 1)

source = replace_once(
    source,
    dedent(
        '''
                    "text": unit["text"],
                    "next_source": _compact_consistency_text(
        '''
    ),
    dedent(
        '''
                    "text": unit["text"],
                    "required_digit_tokens": sorted(
                        _canonical_numeric_tokens(unit["text"]).elements()
                    ),
                    "required_number_word_spans": _source_number_word_spans(
                        unit["text"],
                        source_language,
                    ),
                    "required_protected_literals": sorted(
                        _protected_literal_tokens(unit["text"])
                    ),
                    "next_source": _compact_consistency_text(
        '''
    ),
    "compact payload constraints",
)
source = replace_once(
    source,
    dedent(
        '''
        "consistency_context" contains earlier SOURCE→TRANSLATION examples only when the
        current item shares a repeated likely name/term with them. Use it ONLY to keep the
        same entity/name/term spelling consistent. Never import facts from those examples.

        FIDELITY RULES
        '''
    ),
    dedent(
        '''
        "consistency_context" contains earlier SOURCE→TRANSLATION examples only when the
        current item shares a repeated likely name/term with them. Use it ONLY to keep the
        same entity/name/term spelling consistent. Never import facts from those examples.
        "required_digit_tokens" are hard constraints: copy every listed digit token
        verbatim with identical multiplicity; never spell it out, convert units, round it,
        or replace it with an equivalent value. "required_number_word_spans" must preserve
        the exact cardinal/ordinal value and scope. "required_protected_literals" must be
        copied exactly.

        FIDELITY RULES
        '''
    ),
    "compact prompt constraint explanation",
)

# Cycle 3: remove cross-item translation examples from first-pass generation and
# checkpoint only translations that already pass deterministic factual gates.
source = replace_once(
    source,
    "    repeated_consistency_keys = _repeated_consistency_keys(units)\n",
    "",
    "unused initial consistency keys",
)
old_batch_block = dedent(
    '''
            def on_partial(partial: dict[int, str]) -> None:
                translations.update(partial)
                if checkpoint is not None:
                    checkpoint(translations)

            enriched_missing = _with_consistency_context(
                missing,
                all_units=units,
                translations=translations,
                repeated_keys=repeated_consistency_keys,
            )
            translations.update(
                _translate_compact_batch(
                    source_language=source_language,
                    target_language=target_language,
                    batch=enriched_missing,
                    generator=generator,
                    on_partial=on_partial,
                )
            )
            if checkpoint is not None:
                checkpoint(translations)
    '''
)
new_batch_block = dedent(
    '''
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
    '''
)
source = replace_once(
    source,
    old_batch_block,
    new_batch_block,
    "validated compact checkpoint block",
)

# Cycle 4: strengthen deterministic acceptance gates.
source = replace_function(
    source,
    "_hard_deterministic_review_issues",
    r'''
    def _hard_deterministic_review_issues(batch: list[dict]) -> list[str]:
        issues = []
        for unit in batch:
            numeric_issue = _numeric_fidelity_issue(unit)
            if numeric_issue:
                issues.append(numeric_issue)
            year_issue = _year_fidelity_issue(unit)
            if year_issue and year_issue not in issues:
                issues.append(year_issue)
            protected_issue = _protected_literal_issue(unit)
            if protected_issue:
                issues.append(protected_issue)
            untranslated_issue = _untranslated_copy_issue(unit)
            if untranslated_issue:
                issues.append(untranslated_issue)
            copied_span_issue = _copied_source_span_issue(unit)
            if copied_span_issue:
                issues.append(copied_span_issue)
            truncation_issue = _hard_truncation_issue(unit)
            if truncation_issue:
                issues.append(truncation_issue)
            else:
                length_issue = _suspicious_length_issue(unit)
                if length_issue:
                    issues.append(length_issue)
        return issues
    ''',
)

# Cycle 5: isolate targeted repair when context itself could cause hallucinations.
old_constraint_block = dedent(
    '''
        required_literal_tokens = sorted(_protected_literal_tokens(source_text))
        hard_constraints = {
            "required_digit_tokens_verbatim": required_digit_tokens,
            "required_protected_literals_verbatim": required_literal_tokens,
        }

        previous, following = _repair_context_for_unit(
            source_units,
            unit,
        )
        previous = _compact_consistency_text(previous, max_chars=480)
        following = _compact_consistency_text(following, max_chars=480)
        repeated_keys = _repeated_consistency_keys(source_units)
        consistency_context = _with_consistency_context(
            [unit],
            all_units=source_units,
            translations=translations,
            repeated_keys=repeated_keys,
        )[0].get("consistency_context", [])
        position = next(
            (
                index
                for index, candidate in enumerate(source_units)
                if candidate["i"] == unit["i"]
            ),
            -1,
        )
    '''
)
new_constraint_block = dedent(
    '''
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
                r"added|not present|identity|wrong person|neighbor|"
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
            re.search(r"consisten|anchor item|transliteration", str(issue), re.I)
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
    '''
)
source = replace_once(
    source,
    old_constraint_block,
    new_constraint_block,
    "repair context isolation",
)
source = replace_once(
    source,
    dedent(
        '''
        If required_digit_tokens_verbatim is non-empty, EVERY listed digit token must appear
        verbatim in the corrected translation with the same value and multiplicity. Do not
        spell those digits out, round them, convert units, or substitute a nearby number.
        If required_protected_literals_verbatim is non-empty, preserve each listed literal
        exactly.
        '''
    ),
    dedent(
        '''
        If required_digit_tokens_verbatim is non-empty, EVERY listed digit token must appear
        verbatim in the corrected translation with the same value and multiplicity. Do not
        spell those digits out, round them, convert units, or substitute a nearby number.
        Preserve the exact cardinal/ordinal value and scope of every
        required_number_word_spans entry. Explicitly represent every
        required_reviewer_source_spans entry in the corrected translation. If
        required_protected_literals_verbatim is non-empty, preserve each listed literal
        exactly.
        '''
    ),
    "repair prompt hard instructions",
)
source = replace_once(
    source,
    "CURRENT TRANSLATION UNDER REVIEW:\n{current_translation}",
    "CURRENT TRANSLATION UNDER REVIEW:\n{translation_under_review}",
    "risky translation suppression",
)

# Cycle 6: literal mode must survive schema drift and use an independent semantic check.
source = replace_once(
    source,
    "    required_spans = _review_source_span_hints(source_text, issues)\n"
    "    return f\"\"\"\n",
    "    required_spans = _review_source_span_hints(source_text, issues)\n"
    "    seen_spans = {span.casefold() for span in required_spans}\n"
    "    for span in _source_number_word_spans(source_text, source_language):\n"
    "        if span.casefold() not in seen_spans:\n"
    "            seen_spans.add(span.casefold())\n"
    "            required_spans.append(span)\n"
    "    return f\"\"\"\n",
    "literal number-word constraints",
)
source = replace_function(
    source,
    "_literal_fidelity_repair",
    r'''
    def _literal_fidelity_repair(
        *,
        generator: Callable[[str], str],
        source_language: str,
        target_language: str,
        unit: dict,
        issues: list[str],
        reviewer: Callable[[str], str] | None = None,
    ) -> str:
        last_issues = list(issues)
        last_hard_valid_text = ""
        for _ in range(3):
            try:
                response = generator(
                    _build_literal_fidelity_repair_prompt(
                        source_language=source_language,
                        target_language=target_language,
                        unit=unit,
                        issues=last_issues,
                    )
                )
                repaired = _parse_compact_localization_response(response, [unit])
            except _LocalizationSchemaError as exc:
                last_issues = [*issues, f"item {unit['i']}: {exc}"]
                continue

            text = str(repaired.get(unit["i"]) or "").strip()
            if not text:
                last_issues = [
                    *issues,
                    f"item {unit['i']}: literal repair returned empty text",
                ]
                continue

            hard_issues = _hard_deterministic_review_issues(
                [
                    {
                        "i": unit["i"],
                        "source": unit["text"],
                        "translation": text,
                    }
                ]
            )
            if hard_issues:
                last_issues = [*issues, *hard_issues]
                continue

            last_hard_valid_text = text
            if reviewer is None:
                return text

            semantic_issues = _run_final_materiality_review_item(
                reviewer=reviewer,
                source_language=source_language,
                target_language=target_language,
                unit={
                    "i": unit["i"],
                    "source": unit["text"],
                    "translation": text,
                    "previous_source": "",
                    "next_source": "",
                },
            )
            if not semantic_issues:
                return text
            last_issues = [*issues, *semantic_issues]

        # Keep a hard-valid candidate if the semantic reviewer remained noisy.
        # The outer materiality gate independently checks it before final approval.
        if last_hard_valid_text:
            return last_hard_valid_text

        raise _LocalizationSemanticError(
            "literal repair could not satisfy deterministic factual constraints for "
            f"item {unit['i']}: "
            + "; ".join(last_issues[-4:])
        )
    ''',
)

# Cycle 7: one context-aware attempt, then source-only literal mode immediately.
source = replace_function(
    source,
    "_repair_single_localization_item",
    r'''
    def _repair_single_localization_item(
        *,
        generator: Callable[[str], str],
        source_language: str,
        target_language: str,
        unit: dict,
        source_units: list[dict],
        translations: dict[int, str],
        current_translation: str,
        issues: list[str],
        reviewer: Callable[[str], str] | None = None,
    ) -> str:
        response = generator(
            _build_single_item_repair_prompt(
                source_language=source_language,
                target_language=target_language,
                unit=unit,
                source_units=source_units,
                translations=translations,
                current_translation=current_translation,
                issues=issues,
            )
        )
        repaired = _parse_compact_localization_response(response, [unit])
        text = str(repaired.get(unit["i"]) or "").strip()
        if not text:
            raise _LocalizationSchemaError(
                f"localization repair returned no text for item {unit['i']}"
            )

        hard_issues = _hard_deterministic_review_issues(
            [
                {
                    "i": unit["i"],
                    "source": unit["text"],
                    "translation": text,
                }
            ]
        )
        if not hard_issues:
            return text

        return _literal_fidelity_repair(
            generator=generator,
            source_language=source_language,
            target_language=target_language,
            unit=unit,
            issues=[
                *issues,
                *hard_issues,
                (
                    f"item {unit['i']}: context-aware repair failed mandatory "
                    "machine checks; use source-only literal repair"
                ),
            ],
            reviewer=reviewer,
        )
    ''',
)
source = replace_function(
    source,
    "_repair_compact_review_batch",
    r'''
    def _repair_compact_review_batch(
        *,
        generator: Callable[[str], str],
        source_language: str,
        target_language: str,
        batch: list[dict],
        source_units: list[dict],
        translations: dict[int, str],
        issues: list[str],
        reviewer: Callable[[str], str] | None = None,
        status: LocalizationStatusCallback | None = None,
        repair_attempt: int = 1,
        repair_attempts: int = 1,
    ) -> dict[int, str]:
        repaired: dict[int, str] = {}
        for item_index, unit in enumerate(batch, start=1):
            _report_localization_status(
                status,
                "repair_item",
                item_index,
                len(batch),
                repair_attempt,
                repair_attempts,
            )
            item_issues = _issues_for_item(issues, unit["i"])
            try:
                repaired[unit["i"]] = _repair_single_localization_item(
                    generator=generator,
                    source_language=source_language,
                    target_language=target_language,
                    unit=unit,
                    source_units=source_units,
                    translations=translations,
                    current_translation=translations.get(unit["i"], ""),
                    issues=item_issues,
                    reviewer=reviewer,
                )
            except _LocalizationSchemaError as exc:
                repaired[unit["i"]] = _literal_fidelity_repair(
                    generator=generator,
                    source_language=source_language,
                    target_language=target_language,
                    unit=unit,
                    issues=[*item_issues, f"item {unit['i']}: {exc}"],
                    reviewer=reviewer,
                )
            except _NonRetryableLocalizationError:
                raise
        return repaired
    ''',
)

# Pass the independent reviewer into both production repair paths.
source = replace_once(
    source,
    "                issues=batch_issues,\n"
    "                status=status,\n",
    "                issues=batch_issues,\n"
    "                reviewer=reviewer,\n"
    "                status=status,\n",
    "main repair reviewer wiring",
)
source = replace_once(
    source,
    "                issues=consistency_issues,\n"
    "                status=status,\n",
    "                issues=consistency_issues,\n"
    "                reviewer=reviewer,\n"
    "                status=status,\n",
    "consistency repair reviewer wiring",
)

# Tests: shorter fallback plus new regression coverage for every discovered weakness.
tests = replace_once(
    tests,
    "    assert len(calls) == 4\n"
    "    assert \"EMERGENCY LITERAL FIDELITY REPAIR\" in calls[-1]\n",
    "    assert len(calls) == 2\n"
    "    assert \"EMERGENCY LITERAL FIDELITY REPAIR\" in calls[-1]\n",
    "literal fallback call count",
)

extra_tests = dedent(
    r'''


def test_review_policy_and_work_cache_versions_advance_together():
    from app.services.documentary.localization import (
        _LOCALIZATION_WORK_REVIEW_POLICY_VERSION,
        _LOCALIZATION_WORK_VERSION,
    )

    assert CURRENT_LOCALIZATION_REVIEW_VERSION >= 6
    assert (
        _LOCALIZATION_WORK_REVIEW_POLICY_VERSION
        == CURRENT_LOCALIZATION_REVIEW_VERSION
    )
    assert _LOCALIZATION_WORK_VERSION >= 2


def test_compact_prompt_exposes_per_item_machine_constraints():
    prompt = _build_compact_localization_prompt(
        source_language="en",
        target_language="ru",
        batch=[
            {
                "i": 2,
                "kind": "narration",
                "previous_text": "",
                "text": "At 7 on Christmas Eve, seven people waited.",
                "next_text": "",
                "consistency_context": [],
            }
        ],
    )

    assert '"required_digit_tokens":["7"]' in prompt
    assert '"required_number_word_spans":["seven"]' in prompt
    assert "copy every listed digit token" in prompt


def test_repair_drops_risky_neighbor_and_consistency_context():
    unit = {
        "i": 170,
        "scene_index": 1,
        "kind": "narration",
        "text": "Camila's mother continued searching.",
        "previous_text": "Rosario spoke to reporters.",
        "next_text": "Volunteers joined later.",
    }
    prompt = _build_single_item_repair_prompt(
        source_language="en",
        target_language="ru",
        unit=unit,
        source_units=[
            {
                "i": 169,
                "scene_index": 0,
                "kind": "narration",
                "text": "Rosario spoke to reporters.",
                "previous_text": "",
                "next_text": unit["text"],
            },
            unit,
        ],
        translations={169: "Росарио поговорила с журналистами."},
        current_translation="Росарио продолжила поиски.",
        issues=["item 170: added person/entity identity not present in source"],
    )

    assert "CURRENT SOURCE:\nCamila's mother continued searching." in prompt
    assert "PREVIOUS SOURCE:\n\n" in prompt
    assert "NEXT SOURCE:\n\n" in prompt
    assert "Росарио поговорила с журналистами" not in prompt
    assert "Росарио продолжила поиски" not in prompt


def test_literal_repair_retries_after_schema_drift():
    from app.services.documentary.localization import _literal_fidelity_repair

    unit = {
        "i": 2,
        "scene_index": 0,
        "kind": "narration",
        "text": "At 7 on Christmas Eve, the family waited.",
    }
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return json.dumps(
            {
                "items": [
                    {
                        "i": 2,
                        "text": "В 7 часов в канун Рождества семья ждала.",
                    }
                ]
            },
            ensure_ascii=False,
        )

    repaired = _literal_fidelity_repair(
        generator=generate,
        source_language="en",
        target_language="ru",
        unit=unit,
        issues=["item 2: omitted 7 and Christmas Eve"],
    )

    assert len(calls) == 2
    assert "7" in repaired
    assert "канун Рождества" in repaired


def test_hard_gate_rejects_long_partially_untranslated_span():
    from app.services.documentary.localization import (
        _hard_deterministic_review_issues,
    )

    issues = _hard_deterministic_review_issues(
        [
            {
                "i": 24,
                "source": (
                    "The family continued searching through the night while "
                    "volunteers checked roads and nearby fields."
                ),
                "translation": (
                    "Семья продолжала поиски: the family continued searching "
                    "through the night while volunteers checked roads and nearby fields."
                ),
            }
        ]
    )

    assert any("verbatim source-language span" in issue for issue in issues)
    '''
)
if "test_review_policy_and_work_cache_versions_advance_together" not in tests:
    tests += extra_tests

LOCALIZATION_PATH.write_text(source, encoding="utf-8")
TESTS_PATH.write_text(tests, encoding="utf-8")
