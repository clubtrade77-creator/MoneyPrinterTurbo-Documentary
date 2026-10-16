from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATCH = Path(".github/localization_audit_round2.py")

text = PATCH.read_text(encoding="utf-8")
start = text.index("old_context = dedent(")
end_marker = "\n\n# Cycle 7: reviewer issue span extraction"
end = text.index(end_marker, start)
replacement = '''context_function_start = source.index(
    "def _build_single_item_repair_prompt("
)
context_start = source.index(
    '    issue_blob = " ".join(str(issue) for issue in issues).casefold()\\n',
    context_function_start,
)
context_end = source.index(
    "    consistency_context = []\\n",
    context_start,
)
new_context = dedent(
    """
        issue_blob = " ".join(str(issue) for issue in issues).casefold()
        context_needed = bool(
            re.search(
                r"fragment|pronoun|referent|antecedent|coreference|gender|"
                r"tense|aspect|chronolog|before|after|previous|next|neighbor|"
                r"continuation|sentence boundary",
                issue_blob,
            )
        )
        if context_needed:
            previous, following = _repair_context_for_unit(
                source_units,
                unit,
            )
            previous = _compact_consistency_text(previous, max_chars=480)
            following = _compact_consistency_text(following, max_chars=480)
        else:
            previous, following = "", ""

        consistency_only = bool(issues) and all(
            re.search(
                r"consisten|anchor item|transliteration",
                str(issue),
                re.I,
            )
            for issue in issues
        )
        translation_under_review = (
            current_translation if consistency_only else ""
        )
    """
)
new_context = "\\n".join(
    f"    {line}" if line else ""
    for line in new_context.splitlines()
) + "\\n"
source = source[:context_start] + new_context + source[context_end:]
'''
text = text[:start] + replacement + text[end:]
PATCH.write_text(text, encoding="utf-8")
subprocess.run([sys.executable, str(PATCH)], check=True)
