"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """Return a small synthetic RGB image for tests."""
    return Image.new("RGB", (64, 64), color=(128, 64, 32))


@pytest.fixture
def sample_ordinal_labels() -> list[float]:
    return [1.0, 3.0, 5.0, 7.0, 9.0]


@pytest.fixture
def sample_predicted_scores() -> list[float]:
    return [1.5, 2.8, 5.2, 6.9, 8.7]
