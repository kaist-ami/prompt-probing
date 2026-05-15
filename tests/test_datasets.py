"""Tests for dataset loading utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from prompt_probing.data.datasets import (
    OrdinalSample,
    THINGSDataset,
    WinoGroundDataset,
    load_dataset,
)


class TestOrdinalSample:
    def test_namedtuple_fields(self):
        sample = OrdinalSample(
            image=None,
            label=3.5,
            attribute="size",
            item_name="cat",
            metadata={"source": "test"},
        )
        assert sample.label == 3.5
        assert sample.attribute == "size"
        assert sample.item_name == "cat"
        assert sample.metadata["source"] == "test"


class TestTHINGSDataset:
    def test_unsupported_attribute_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            THINGSDataset(root="/tmp", attribute="nonexistent_attr")

    def test_missing_norms_file_raises(self, tmp_path):
        ds = THINGSDataset(root=tmp_path, attribute="size")
        with pytest.raises(FileNotFoundError, match="norms file not found"):
            ds.load()

    def test_load_from_csv(self, tmp_path):
        # Create minimal THINGS directory structure
        (tmp_path / "norms").mkdir()
        (tmp_path / "images").mkdir()

        norms_csv = tmp_path / "norms" / "size_norms.csv"
        norms_csv.write_text(
            "# header\n"
            "apple,3.5\n"
            "elephant,8.2\n"
            "ant,1.1\n",
            encoding="utf-8",
        )

        # Create dummy images
        for name in ("apple", "elephant", "ant"):
            Image.new("RGB", (10, 10)).save(tmp_path / "images" / f"{name}.jpg")

        ds = THINGSDataset(root=tmp_path, attribute="size")
        ds.load()

        assert len(ds) == 3
        labels = ds.labels
        assert pytest.approx(3.5) in labels
        assert pytest.approx(8.2) in labels


class TestWinoGroundDataset:
    def test_missing_data_raises(self, tmp_path):
        ds = WinoGroundDataset(root=tmp_path, attribute="semantic_order")
        with pytest.raises(FileNotFoundError, match="data file not found"):
            ds.load()

    def test_load_from_jsonl(self, tmp_path):
        (tmp_path / "images").mkdir()
        data_path = tmp_path / "data.jsonl"
        entries = [
            {"id": 1, "caption_0": "A red ball", "caption_1": "A blue ball", "tag": "color"},
            {"id": 2, "caption_0": "A big dog", "caption_1": "A small dog", "tag": "size"},
        ]
        with open(data_path, "w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")

        ds = WinoGroundDataset(root=tmp_path)
        ds.load()

        # 2 entries × 2 images each = 4 samples
        assert len(ds) == 4


class TestLoadDataset:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent", root="/tmp", attribute="size")
