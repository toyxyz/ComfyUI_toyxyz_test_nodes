"""Local line-based wildcards; never read outside this node's wildcard folder."""
import logging
import os
import random
import re
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "wildcards"
CONFIG_PATH = Path(__file__).resolve().parents[1] / ".wildcard_settings.json"
TOKEN = re.compile(r"__([^\r\n]+?)__")
MAX_PREVIEW_CHARS = 20_000
LOG = logging.getLogger(__name__)


def active_wildcard_root():
    try:
        configured = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("wildcard_path", "")
        if isinstance(configured, str) and configured:
            return Path(configured).resolve()
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
        pass
    return ROOT.resolve()


def set_wildcard_path(value):
    if not isinstance(value, str):
        raise ValueError("Wildcard path must be text.")
    value = value.strip()
    if value:
        path = Path(value)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("Enter an existing absolute folder path, or leave it blank for the built-in folder.")
        value = str(path.resolve())
    temporary = CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({"wildcard_path": value}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(CONFIG_PATH)
    return value


def open_wildcard_folder():
    """Open the configured wildcard directory, never a client-supplied URL path."""
    root = active_wildcard_root()
    if root == ROOT.resolve():
        root.mkdir(parents=True, exist_ok=True)
    os.startfile(str(root))


def browse_wildcard_folder(description="Select wildcard folder"):
    """Ask the local Windows user to choose a folder; do not alter settings."""
    if sys.platform != "win32":
        raise OSError("Native folder browsing is available only on Windows.")
    executable = shutil.which("powershell.exe")
    if not executable:
        raise OSError("Windows PowerShell is unavailable.")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
        f"$dialog.Description = '{description}'; "
        "$dialog.ShowNewFolderButton = $false; "
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "[Console]::Write($dialog.SelectedPath) }"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-STA", "-WindowStyle", "Hidden", "-Command", script],
        capture_output=True, encoding="utf-8", errors="replace", timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise OSError("The folder picker could not be opened.")
    selected = result.stdout.strip()
    if selected and (not Path(selected).is_absolute() or not Path(selected).is_dir()):
        raise OSError("The selected folder is unavailable.")
    return selected


def wildcard_files(root=None):
    root = Path(root).resolve() if root is not None else active_wildcard_root()
    if not root.is_dir():
        return {}
    return {path.relative_to(root).with_suffix("").as_posix(): path
            for path in sorted(root.rglob("*.txt"))
            if path.is_file() and path.resolve().is_relative_to(root)}


def wildcard_preview(name, root=None):
    """Read only a configured wildcard file selected by its listed name."""
    if not isinstance(name, str) or not name or len(name) > 512:
        raise KeyError(name)
    path = wildcard_files(root).get(name)
    if path is None:
        raise KeyError(name)
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        content = source.read(MAX_PREVIEW_CHARS + 1)
    return {"content": content[:MAX_PREVIEW_CHARS],
            "truncated": len(content) > MAX_PREVIEW_CHARS}


def expand_wildcards(text, root=None, chooser=None):
    files = wildcard_files(root)
    chooser = chooser or random.SystemRandom()
    lines = {}
    replacements = 0

    def expand(value, stack=()):
        def replace(match):
            nonlocal replacements
            name = match.group(1)
            if name not in files or name in stack or len(stack) >= 16 or replacements >= 256:
                LOG.warning("Wildcard unavailable or recursive: %s", name)
                return match.group(0)
            if name not in lines:
                try:
                    lines[name] = [line.strip() for line in files[name].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                except (OSError, UnicodeError):
                    LOG.warning("Wildcard could not be read: %s", name)
                    lines[name] = []
            if not lines[name]:
                return match.group(0)
            replacements += 1
            return expand(chooser.choice(lines[name]), (*stack, name))
        return TOKEN.sub(replace, value)
    return expand(text)
