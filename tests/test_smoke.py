import tomllib
from pathlib import Path

import docs_rag_agent


def test_package_version_matches_pyproject():
    """Assert that the package version matches the version in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    expected_version = pyproject_data["project"]["version"]
    assert docs_rag_agent.__version__ == expected_version
