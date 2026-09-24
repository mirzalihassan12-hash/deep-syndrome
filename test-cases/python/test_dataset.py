"""
Test cases for training/dataset.py's near-duplicate detection and folder
discovery logic — the mechanism that prevents train/val/test leakage.
"""
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from training.dataset import _ahash, _hamming, _group_near_duplicates, find_class_folders


def _make_image(path, color, size=(64, 64)):
    """A plain solid-colour placeholder image (fine when hash content doesn't matter)."""
    Image.new("RGB", size, color=color).save(path)


def _make_pattern_image(path, pattern, size=(64, 64)):
    """
    A non-flat image, needed for aHash tests: a solid-colour image is a
    degenerate case for average-hash (every pixel equals the average, so
    ALL solid-colour images hash identically regardless of colour). Real
    photos always have spatial structure, so tests must too.
    """
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    if pattern == "diagonal":
        draw.polygon([(0, 0), (size[0], 0), (0, size[1])], fill=(0, 0, 0))
    elif pattern == "vertical_stripe":
        draw.rectangle([size[0] // 3, 0, 2 * size[0] // 3, size[1]], fill=(0, 0, 0))
    elif pattern == "checker":
        step = size[0] // 4
        for i in range(0, size[0], step):
            for j in range(0, size[1], step):
                if (i // step + j // step) % 2 == 0:
                    draw.rectangle([i, j, i + step, j + step], fill=(0, 0, 0))
    img.save(path)


def test_ahash_is_deterministic_for_the_same_image(tmp_path):
    img_path = tmp_path / "a.png"
    _make_image(img_path, (200, 50, 50))

    h1 = _ahash(str(img_path))
    h2 = _ahash(str(img_path))
    assert h1 == h2


def test_hamming_distance_zero_for_identical_hashes():
    assert _hamming(0b1010, 0b1010) == 0


def test_hamming_distance_counts_differing_bits():
    assert _hamming(0b0000, 0b1111) == 4
    assert _hamming(0b1010, 0b0010) == 1


def test_group_near_duplicates_clusters_identical_images_only(tmp_path):
    # Two byte-identical copies of the same image (a re-saved duplicate)...
    dup_a = tmp_path / "dup_a.png"
    dup_b = tmp_path / "dup_b.png"
    _make_pattern_image(dup_a, "diagonal")
    shutil.copyfile(dup_a, dup_b)

    # ...and one genuinely different image.
    unique = tmp_path / "unique.png"
    _make_pattern_image(unique, "checker")

    groups = _group_near_duplicates([str(dup_a), str(dup_b), str(unique)])

    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2], "expected one pair grouped together and one singleton group"

    # The duplicate pair must end up in the same group.
    pair_group = next(g for g in groups if len(g) == 2)
    assert str(dup_a) in pair_group and str(dup_b) in pair_group


def test_group_near_duplicates_handles_all_unique_images(tmp_path):
    paths = []
    for i, pattern in enumerate(["diagonal", "vertical_stripe", "checker"]):
        p = tmp_path / f"img_{i}.png"
        _make_pattern_image(p, pattern)
        paths.append(str(p))

    groups = _group_near_duplicates(paths)
    assert len(groups) == 3, "distinct images must not be grouped together"


def test_find_class_folders_classifies_by_keyword(tmp_path):
    ds_dir = tmp_path / "down_syndrome"
    ctrl_dir = tmp_path / "control"
    other_dir = tmp_path / "misc"
    for d in (ds_dir, ctrl_dir, other_dir):
        d.mkdir()
    _make_image(ds_dir / "a.jpg", (1, 1, 1))
    _make_image(ctrl_dir / "b.jpg", (2, 2, 2))
    _make_image(other_dir / "c.jpg", (3, 3, 3))

    ds_folders, ctrl_folders = find_class_folders(str(tmp_path))

    ds_names = [Path(root).name for root, _ in ds_folders]
    ctrl_names = [Path(root).name for root, _ in ctrl_folders]

    assert "down_syndrome" in ds_names
    assert "control" in ctrl_names
    # A folder matching neither keyword set must not be classified as either class.
    assert "misc" not in ds_names and "misc" not in ctrl_names


def test_find_class_folders_ignores_folders_with_no_images(tmp_path):
    empty_ds_dir = tmp_path / "down_syndrome_empty"
    empty_ds_dir.mkdir()
    (empty_ds_dir / "notes.txt").write_text("not an image")

    ds_folders, ctrl_folders = find_class_folders(str(tmp_path))
    assert ds_folders == [] and ctrl_folders == []
