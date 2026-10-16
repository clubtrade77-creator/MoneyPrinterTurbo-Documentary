from __future__ import annotations

from pathlib import Path

MAIN = Path("webui/Main.py")
TASK_TEST = Path("test/services/test_webui_task.py")

main = MAIN.read_text(encoding="utf-8")
tests = TASK_TEST.read_text(encoding="utf-8")

old_workspace = '''    workspace_mode = st.radio(
        tr("Workspace Mode"),
        options=("short_video", "documentary"),
        format_func=lambda value: (
            tr("Short Video Mode")
            if value == "short_video"
            else tr("Documentary Mode")
        ),
        key="workspace_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    if workspace_mode == "documentary":
'''
new_workspace = '''    # Streamlit AppTest serializes the formatted radio label, while the normal
    # runtime stores the raw option. Keeping raw values together with format_func
    # therefore makes the widget state invalid on a rerun (notably on Windows).
    # Store a translated label in the widget and keep the stable internal mode in
    # a separate session key. This also migrates existing sessions safely.
    workspace_mode_labels = {
        "short_video": tr("Short Video Mode"),
        "documentary": tr("Documentary Mode"),
    }
    workspace_mode_by_label = {
        label: mode
        for mode, label in workspace_mode_labels.items()
    }
    stored_workspace_mode = st.session_state.get(
        "workspace_mode_internal",
        st.session_state.get("workspace_mode", "short_video"),
    )
    if stored_workspace_mode not in workspace_mode_labels:
        stored_workspace_mode = "short_video"

    workspace_label_key = "workspace_mode_label"
    if st.session_state.get(workspace_label_key) not in workspace_mode_by_label:
        st.session_state[workspace_label_key] = workspace_mode_labels[
            stored_workspace_mode
        ]

    workspace_mode_label = st.radio(
        tr("Workspace Mode"),
        options=tuple(workspace_mode_by_label),
        key=workspace_label_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    workspace_mode = workspace_mode_by_label[workspace_mode_label]
    st.session_state["workspace_mode_internal"] = workspace_mode
    # Preserve the legacy non-widget key for integrations that inspect it.
    st.session_state["workspace_mode"] = workspace_mode

    if workspace_mode == "documentary":
'''
if main.count(old_workspace) != 1:
    raise RuntimeError(
        f"workspace mode anchor mismatch: {main.count(old_workspace)}"
    )
main = main.replace(old_workspace, new_workspace, 1)

old_width = '                "use_container_width": True,\n'
new_width = '                "width": "stretch",\n'
if tests.count(old_width) != 1:
    raise RuntimeError(
        f"download button width anchor mismatch: {tests.count(old_width)}"
    )
tests = tests.replace(old_width, new_width, 1)

MAIN.write_text(main, encoding="utf-8")
TASK_TEST.write_text(tests, encoding="utf-8")
