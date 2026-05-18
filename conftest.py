"""Root pytest configuration: --run-hardware flag + hardware marker gating."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require physical hardware (D435, SO-101 arm).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-hardware"):
        return
    skip_hw = pytest.mark.skip(reason="requires hardware (pass --run-hardware to run)")
    for item in items:
        if item.get_closest_marker("hardware"):
            item.add_marker(skip_hw)
