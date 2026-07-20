import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def example_cool():
    """The small, git-tracked H1-hESC chr10:19-26 Mb example (docs/data/create_example_data.py)."""
    return os.path.join(REPO_ROOT, "docs", "data", "example_chr10.cool")


@pytest.fixture(scope="session")
def example_ctcf_tsv():
    """CTCF+/- track for the same window as example_cool (docs/data/example_ctcf.tsv)."""
    return os.path.join(REPO_ROOT, "docs", "data", "example_ctcf.tsv")
