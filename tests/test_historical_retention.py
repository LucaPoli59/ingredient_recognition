import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.validate_legacy_experiments import verify_manifest


class HistoricalRetentionTests(unittest.TestCase):
    def test_declared_journal_may_only_grow_by_preserving_the_hashed_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "experiments" / "journal.log"
            journal.parent.mkdir()
            retained = b"historical-journal\n"
            journal.write_bytes(retained + b"new-trial\n")
            manifest = {"artifacts": [{
                "path": "experiments/journal.log",
                "size": len(retained),
                "sha256": hashlib.sha256(retained).hexdigest().upper(),
            }]}

            extensions = verify_manifest(root, manifest)

            self.assertEqual(len(extensions), 1)
            self.assertEqual(extensions[0]["retained_prefix_size"], len(retained))

    def test_mutating_a_retained_journal_prefix_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "experiments" / "journal.log"
            journal.parent.mkdir()
            retained = b"historical-journal\n"
            journal.write_bytes(b"X" + retained[1:] + b"new-trial\n")
            manifest = {"artifacts": [{
                "path": "experiments/journal.log",
                "size": len(retained),
                "sha256": hashlib.sha256(retained).hexdigest().upper(),
            }]}

            with self.assertRaises(AssertionError):
                verify_manifest(root, manifest)


if __name__ == "__main__":
    unittest.main()
