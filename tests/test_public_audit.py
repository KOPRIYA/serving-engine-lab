import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "audit_public", Path(__file__).resolve().parents[1] / "scripts/audit_public.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PublicAuditTests(unittest.TestCase):
    def test_ignore_rules_without_initializing_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".gitignore").write_text("local-notes/\nmodels/\n.env\n")
            for name in ["local-notes/INTERVIEW.md", "models/model.safetensors", ".env", "README.md"]:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("fixture")
            files, tracked = audit.candidate_files(root)
            self.assertEqual(files, [".gitignore", "README.md"])
            self.assertFalse(tracked)
            self.assertFalse((root / ".git").exists())

    def test_already_tracked_ignored_file_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".env").write_text("fixture")
            subprocess.run(["git", "-C", str(root), "add", ".env"], check=True)
            (root / ".gitignore").write_text(".env\n")
            files, ignored = audit.candidate_files(root)
            self.assertIn(".env", files)
            self.assertIn(".env", ignored)

    def test_private_text_and_notes_are_flagged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / "INTERVIEW.md"
            p.write_text("/" + "Users" + "/example/private/file")
            issues = audit.inspect_file(root, p.name)
            self.assertIn("private material in public file set", issues)
            self.assertIn("personal absolute path", issues)

    def test_environment_example_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".env.example").write_text("API_KEY=replace-me\n")
            self.assertEqual(audit.inspect_file(root, ".env.example"), [])


if __name__ == "__main__":
    unittest.main()
