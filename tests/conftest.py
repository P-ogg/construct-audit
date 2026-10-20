"""Remove the app's working folders that the tests create (the app itself deletes old ones after a day)."""
import shutil, tempfile
from pathlib import Path
import pytest


@pytest.fixture(autouse=True, scope="session")
def _clean_app_folders():
    before = set(Path(tempfile.gettempdir()).glob("caudit_*"))
    yield
    for d in set(Path(tempfile.gettempdir()).glob("caudit_*")) - before:
        shutil.rmtree(d, ignore_errors=True)
