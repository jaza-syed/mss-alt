import tempfile
from pathlib import Path
from unittest import TestCase
from alt.alt_types import JamAltConfig, MusdbAltConfig, ExtractConfig
from alt.extract import run_extract


class TestRunExtractSmoke(TestCase):
    def test_run_extract_smoke(self):
        # Mock configurations
        jam_alt_cfg = JamAltConfig(revision="0e15962", languages=["en", "fr", "de", "es"])
        musdb_alt_cfg = MusdbAltConfig(revision="v1.0.0", musdb_dir=Path("musdb18hq"))
        cfg = ExtractConfig(jam_alt=jam_alt_cfg, musdb_alt=musdb_alt_cfg)

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir)

            # Run the function to ensure it doesn't crash
            try:
                run_extract(cfg=cfg, extract_dir=extract_dir)
            except Exception as e:
                self.fail(f"run_extract raised an exception: {e}")
