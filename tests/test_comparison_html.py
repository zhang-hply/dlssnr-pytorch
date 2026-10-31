"""Offline report generation and conversion without loading the model."""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image
import pytest

from dlssnr.comparison_html import generate_comparison_html, write_comparison_html


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = {}
        self.tags = []
        self.payload = ""
        self.in_payload = False
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.tags.append((tag, attrs))
        if "id" in attrs:
            self.ids[attrs["id"]] = attrs
        if tag == "script" and attrs.get("id") == "comparison-data":
            self.in_payload = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False

    def handle_data(self, data):
        if self.in_payload:
            self.payload += data


def png(root: Path, filename="000000_frame.png", *, width=7, height=5):
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    Image.new("RGB", (width * 3, height), (110, 170, 220)).save(path, format="PNG")
    return path


def manifest(filename="000000_frame.png", *, name="场景一/frame.png", **extra):
    return {"status": "completed", "directory": "/old/machine/evaluation_images",
            "side_by_side_order": ["input", "teacher", "student"],
            "images": [{"name": name, "file": filename}], **extra}


def test_relocated_legacy_manifest_derives_image_dimensions_without_model(tmp_path):
    root = tmp_path / "relocated"
    png(root)
    source = root / "comparison.json"
    source.write_text(json.dumps(manifest()))
    output = generate_comparison_html(source)
    assert output == root / "index.html"
    document = Document(output.read_text())
    data = json.loads(document.payload)
    assert data["images"] == [{"name": "场景一/frame.png", "file": "000000_frame.png",
                               "url": "000000_frame.png", "width": 7, "height": 5}]
    assert "old/machine" not in output.read_text()
    assert document.ids["split-range"]["step"] == "any"  # Allows exact center of odd-width images.
    for element in ("before-source", "frame-select", "prev-frame", "next-frame", "before-upload",
                    "after-upload", "reset-position", "compare-stage", "before-layer", "after-layer",
                    "divider", "split-position", "error-message"):
        assert element in document.ids
    assert not any(attrs.get("src", "").startswith(("http:", "https:")) for _, attrs in document.tags)
    assert not list(root.glob("*.tmp"))


def test_hostile_names_and_encoded_relative_paths_remain_data(tmp_path):
    root = tmp_path / "评估图片"
    filename = '000000_引号 " & # % <x>.png'
    hostile_name = '场景/</script><img src=x onerror="window.injected=1">.png'
    png(root, filename)
    output = write_comparison_html(manifest(filename, name=hostile_name), root, tmp_path / "reports/view.html")
    source = output.read_text()
    document = Document(source)
    data = json.loads(document.payload)
    assert data["images"][0]["name"] == hostile_name
    assert data["images"][0]["url"] == "../" + quote("评估图片/" + filename, safe="/")
    assert hostile_name not in source
    assert len([tag for tag, _ in document.tags if tag == "img"]) == 2
    assert len([tag for tag, _ in document.tags if tag == "script"]) == 2


@pytest.mark.parametrize("status", ["running", "completed", "interrupted", "error"])
def test_empty_partial_manifest_and_unvalidated_incremental_state(tmp_path, status):
    partial = manifest(status=status, width=7, height=5, images=[])
    output = write_comparison_html(partial, tmp_path, validate_images=False)
    assert json.loads(Document(output.read_text()).payload)["images"] == []
    partial["images"] = [{"name": "new.png", "file": "000000_new.png"}]
    write_comparison_html(partial, tmp_path, validate_images=False)
    data = json.loads(Document(output.read_text()).payload)
    assert data["status"] == status and data["images"][0]["width"] == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_conversion_accepts_per_frame_dimensions(tmp_path):
    png(tmp_path, "first.png", width=7, height=5)
    png(tmp_path, "second.png", width=9, height=4)
    values = manifest(images=[{"name": "first", "file": "first.png"}, {"name": "second", "file": "second.png"}])
    data = json.loads(Document(write_comparison_html(values, tmp_path).read_text()).payload)
    assert [(row["width"], row["height"]) for row in data["images"]] == [(7, 5), (9, 4)]


@pytest.mark.parametrize("filename", ["../outside.png", "/tmp/outside.png", "scene/frame.png", "..", "",
                                      "scene\\frame.png", "bad\x00.png"])
def test_unsafe_image_paths_are_rejected_without_writes(tmp_path, filename):
    with pytest.raises(ValueError, match="safe basename"):
        write_comparison_html(manifest(filename), tmp_path)
    assert not (tmp_path / "index.html").exists()


def test_symlink_cannot_escape_comparison_directory(tmp_path):
    outside = png(tmp_path / "outside")
    root = tmp_path / "images"
    root.mkdir()
    (root / outside.name).symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        write_comparison_html(manifest(), root)


@pytest.mark.parametrize("case", ["order", "not_png", "not_triptych", "wrong_dimensions", "missing", "broken"])
def test_converter_rejects_invalid_images_and_order(tmp_path, case):
    path = png(tmp_path)
    values = manifest()
    if case == "order":
        values["side_by_side_order"] = ["student", "input", "teacher"]
    elif case == "not_png":
        Image.new("RGB", (21, 5)).save(path, format="JPEG")
    elif case == "not_triptych":
        Image.new("RGB", (20, 5)).save(path, format="PNG")
    elif case == "wrong_dimensions":
        values.update(width=100, height=5)
    elif case == "missing":
        path.unlink()
    else:
        path.write_bytes(b"broken PNG")
    with pytest.raises((ValueError, OSError)):
        write_comparison_html(values, tmp_path)
    assert not (tmp_path / "index.html").exists()


@pytest.mark.parametrize("target", ["comparison.json", "000000_frame.png", "directory.html"])
def test_output_conflicts_do_not_modify_sources(tmp_path, target):
    png(tmp_path)
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(manifest()))
    (tmp_path / "directory.html").mkdir()
    before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    with pytest.raises(ValueError):
        generate_comparison_html(source, tmp_path / target)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_html_named_source_image_and_manifest_cannot_be_overwritten(tmp_path):
    image_path = png(tmp_path, "image.html")
    contents = image_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        write_comparison_html(manifest("image.html"), tmp_path, image_path)
    assert image_path.read_bytes() == contents
    manifest_path = tmp_path / "manifest.html"
    manifest_path.write_text(json.dumps(manifest("image.html")))
    with pytest.raises(ValueError, match="overwrite"):
        generate_comparison_html(manifest_path, manifest_path)


def test_incremental_renderer_requires_dimensions_when_not_reading_pngs(tmp_path):
    with pytest.raises(ValueError, match="positive image width"):
        write_comparison_html(manifest(), tmp_path, validate_images=False)


@pytest.mark.parametrize("status", [[], {}, None, 1, "unknown"])
def test_invalid_manifest_status_is_a_value_error(tmp_path, status):
    with pytest.raises(ValueError, match="invalid comparison status"):
        write_comparison_html(manifest(status=status), tmp_path)
