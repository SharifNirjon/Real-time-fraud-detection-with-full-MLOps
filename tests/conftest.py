import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.synthetic import make_transactions  # noqa: E402


@pytest.fixture(scope="session")
def transactions():
    return make_transactions(3000, seed=0)
