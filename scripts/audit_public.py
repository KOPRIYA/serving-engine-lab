#!/usr/bin/env python3
"""Inspect files Git would publish, without changing the project's index or files."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_NAMES = {"local-notes", "notes", "models", ".venv", "venv", ".workflow-backups",
                 ".cleanup-backups", "credentials.json", "secrets.json"}
PATTERNS = {
    "personal absolute path": re.compile(r"/(?:Users|home)/[^/\s\"']+/"),
    "GitHub token-like string": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "AWS access-key-like string": re.compile(r"AKIA[A-Z0-9]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def inspect_file(root, name):
    path = root / name
    parts = Path(name).parts
    problems = []
    if any(p in PRIVATE_NAMES or p.lower().startswith("interview") for p in parts):
        problems.append("private material in public file set")
    if path.name.startswith(".env") and path.name != ".env.example":
        problems.append("environment file")
    if path.suffix.lower() in {".pem", ".key", ".safetensors", ".gguf", ".ckpt", ".pt", ".pth"}:
        problems.append("credential or model-weight file")
    if path.is_symlink():
        return problems + ["symlink requires explicit review"]
    if not path.is_file():
        return problems + ["missing or non-regular file"]
    if path.stat().st_size > 5 * 1024 * 1024:
        return problems + ["file exceeds 5 MiB review threshold"]
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problems + ["binary file requires explicit review"]
    for label, pattern in PATTERNS.items():
        if pattern.search(content):
            problems.append(label)
    return problems


def candidate_files(root):
    """Use Git's own ignore semantics, even before a project has been git-initialized."""
    def output(command):
        return subprocess.check_output(command, stderr=subprocess.DEVNULL)
    try:
        repo = Path(output(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
                    .decode().strip())
    except (OSError, subprocess.CalledProcessError):
        repo = None
    def collect(prefix):
        names = output(prefix + ["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
        ignored = output(prefix + ["ls-files", "--cached", "--ignored", "--exclude-standard", "-z"])
        return sorted(set(filter(None, names.decode().split("\0")))), set(filter(None, ignored.decode().split("\0")))
    if repo is not None and repo.resolve() == root.resolve():
        return collect(["git", "-C", str(root)])
    with tempfile.TemporaryDirectory(prefix="serving-public-audit-") as folder:
        subprocess.run(["git", "init", "-q", folder], check=True, capture_output=True)
        return collect(["git", "--git-dir", str(Path(folder) / ".git"), "--work-tree", str(root)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        files, tracked_ignored = candidate_files(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Audit could not complete: {type(exc).__name__}. Git must be installed.")
        return 2
    findings = []
    for name in files:
        issues = inspect_file(root, name)
        if name in tracked_ignored:
            issues.append("already tracked despite ignore rules; review and untrack explicitly")
        if issues:
            findings.append((name, issues))
    for name, issues in findings:
        print(f"REVIEW {name}: {', '.join(issues)}")
    print(f"Checked {len(files)} candidate public files; {len(findings)} require review.")
    print("This audit does not replace manual review or a dedicated secret scanner.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
