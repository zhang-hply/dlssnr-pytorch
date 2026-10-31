"""Offline, draggable comparisons using the existing input|teacher|student PNGs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from PIL import Image


def _output_path(images_dir: Path, out_path) -> Path:
    path = (Path(out_path).expanduser() if out_path is not None else images_dir / "index.html").resolve()
    if path.suffix.lower() != ".html" or path.is_dir():
        raise ValueError("comparison HTML output must be an .html file")
    return path


def _positive_dimension(value) -> bool:
    return type(value) is int and value > 0


def write_comparison_html(comparisons: dict, images_dir: Path, out_path=None, *, validate_images=True) -> Path:
    """Atomically render an offline viewer; all image paths are relative to images_dir.

    validate_images=False is for evaluation callbacks which just wrote PNGs and
    already know their width/height. Conversion validates existing PNG contents.
    The legacy manifest's absolute ``directory`` is deliberately ignored.
    """
    images_dir = Path(images_dir).expanduser().resolve()
    destination = _output_path(images_dir, out_path)
    if not isinstance(comparisons, dict) or comparisons.get("side_by_side_order") != ["input", "teacher", "student"]:
        raise ValueError("comparison manifest must use input | teacher | student order")
    entries = comparisons.get("images")
    if not isinstance(entries, list):
        raise ValueError("comparison manifest images must be a list")
    status = comparisons.get("status", "completed")
    if not isinstance(status, str) or status not in {"running", "completed", "interrupted", "error"}:
        raise ValueError("invalid comparison status")
    width, height = comparisons.get("width"), comparisons.get("height")
    payload = {"status": status, "width": width, "height": height, "images": []}
    protected = {images_dir / "comparison.json"}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("invalid comparison image entry")
        filename = entry.get("file")
        if not isinstance(filename, str) or not filename or filename in {".", ".."} \
                or any(char in filename for char in ("/", "\\", "\x00")):
            raise ValueError("comparison image file must be a safe basename")
        path = images_dir / filename
        if not path.resolve().is_relative_to(images_dir):
            raise ValueError("comparison image escapes its directory")
        protected.add(path.resolve())
        frame_width, frame_height = width, height
        if validate_images:
            with Image.open(path) as image:
                if image.format != "PNG" or image.width % 3 or image.width < 3 or image.height < 1:
                    raise ValueError(f"comparison image must be a triple-width PNG: {filename}")
                frame_width, frame_height = image.width // 3, image.height
                image.verify()
            if _positive_dimension(width) and width != frame_width \
                    or _positive_dimension(height) and height != frame_height:
                raise ValueError(f"comparison image dimensions differ from manifest: {filename}")
        if not _positive_dimension(frame_width) or not _positive_dimension(frame_height):
            raise ValueError("comparison HTML requires positive image width and height")
        relative = Path(os.path.relpath(path, destination.parent)).as_posix()
        payload["images"].append({"name": entry["name"], "file": filename,
                                  "url": quote(relative, safe="/"),
                                  "width": frame_width, "height": frame_height})
    if destination in protected:
        raise ValueError("comparison HTML would overwrite its manifest or source images")
    # Escaping '<' prevents hostile names from terminating the JSON script block.
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    document = _HTML.replace("__COMPARISON_DATA__", encoded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def generate_comparison_html(manifest_path, out_path=None) -> Path:
    """Convert a relocated legacy comparison.json without model inference."""
    source = Path(manifest_path).expanduser().resolve()
    destination = _output_path(source.parent, out_path)
    if destination == source:
        raise ValueError("comparison HTML would overwrite its manifest")
    with source.open(encoding="utf-8") as handle:
        comparisons = json.load(handle)
    return write_comparison_html(comparisons, source.parent, destination)


_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图像对比 · BEFORE / AFTER</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#101317;color:#edf1f5}
*{box-sizing:border-box}body{margin:0;padding:28px 20px 40px}main{max-width:1280px;margin:auto}
h1{font-size:25px;letter-spacing:-.5px;margin:0 0 8px}p{line-height:1.65;color:#aab5c3;margin:8px 0}
.eyebrow{font-size:12px;letter-spacing:2px;color:#94b9cb}.panel{background:#1a2028;border:1px solid #313b48;border-radius:14px;padding:16px;margin-top:18px}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:12px}label{display:flex;flex-direction:column;gap:7px;font-size:13px;color:#c8d2df}
.grow{flex:1;min-width:230px}select,button,input[type=file]{font:inherit;color:inherit;max-width:100%}
select,button{background:#252e3a;border:1px solid #455365;border-radius:7px;padding:9px 12px}select{width:100%}
button{cursor:pointer;white-space:nowrap}button:hover:not(:disabled){background:#344355}button:disabled{opacity:.4;cursor:default}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #9ad8f3;outline-offset:3px}
.viewer{padding:0;overflow:hidden}.stage{position:relative;width:100%;overflow:hidden;touch-action:none;user-select:none;cursor:ew-resize;background:#080a0d;--split:50%}
.layer{position:absolute;inset:0;overflow:hidden;pointer-events:none}.layer img{position:absolute;top:0;height:100%;width:300%;max-width:none;pointer-events:none}
#before-layer{clip-path:inset(0 calc(100% - var(--split)) 0 0)}
.badge{position:absolute;top:15px;padding:6px 9px;background:#111a;border:1px solid #fffa;border-radius:5px;font-size:11px;letter-spacing:1px;color:white;pointer-events:none}
#before-label{left:15px}#after-label{right:15px}
#divider{position:absolute;top:0;bottom:0;left:var(--split);width:2px;transform:translateX(-50%);background:white;box-shadow:0 0 4px #000a;pointer-events:none}
.handle{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:2px solid white;border-radius:50%;width:40px;height:40px;background:#19222ad9;display:grid;place-items:center;color:white;font-size:20px;box-shadow:0 2px 8px #0008}
.range-row{display:flex;gap:15px;align-items:center;padding:15px 18px}.range-row label{flex:1}.range-row input{width:100%;accent-color:#eaf4fa}
#split-position{font-variant-numeric:tabular-nums;min-width:130px;text-align:right;font-size:13px;color:#cfdae6}
#empty-state{text-align:center;padding:80px 20px;color:#aab5c3}.uploads{display:grid;grid-template-columns:1fr 1fr;gap:20px}.uploads input{width:100%;padding:8px 0;color:#aab5c3;font-size:12px}
.footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px;padding:0 18px 14px;font-size:12px;color:#9aaabd}
#error-message{color:#ffb6ac;background:#502b283d;border-left:3px solid #f18a7b;padding:10px 12px;margin-top:14px}
#status-message{min-height:22px;font-size:13px}.small{font-size:12px}#report-status{color:#9bb8c7}
[hidden]{display:none!important}@media(max-width:650px){body{padding:20px 12px}.uploads{grid-template-columns:1fr}.range-row{flex-wrap:wrap}.range-row label{min-width:100%}.badge{top:10px;font-size:10px}#before-label{left:10px}#after-label{right:10px}h1{font-size:22px}}
</style>
</head>
<body><main>
<div class="eyebrow">IMAGE COMPARISON</div>
<h1>拖动分割线，查看图像差异</h1>
<p>左侧 BEFORE，右侧 AFTER。两张图片保持对齐，拖动中间的白线即可逐区域对比。</p>
<p id="report-status" class="small"></p>
<section class="panel controls" aria-label="选择评估图像">
<label class="grow">图像 / 场景<select id="frame-select" aria-label="选择图像"></select></label>
<button id="prev-frame" type="button">上一张</button><button id="next-frame" type="button">下一张</button>
<label>BEFORE 来源<select id="before-source"><option value="teacher">完整教师模型输出 · Teacher</option><option value="input">原始输入图片 · Input</option></select></label>
</section>
<section class="panel viewer" aria-label="图像对比">
<div id="empty-state">尚无可用的评估图片。也可以在下方上传 BEFORE 与 AFTER 图片。</div>
<div id="compare-stage" class="stage" role="group" aria-label="左右拖动白色分割线比较图片" hidden>
<div id="after-layer" class="layer"><img id="after-image" alt="AFTER 图片" draggable="false"></div>
<div id="before-layer" class="layer"><img id="before-image" alt="BEFORE 图片" draggable="false"></div>
<span id="before-label" class="badge">BEFORE</span><span id="after-label" class="badge">AFTER</span>
<div id="divider"><span class="handle" aria-hidden="true">‹ ›</span></div>
</div>
<div class="range-row"><label for="split-range">分割线位置<input id="split-range" type="range" min="0" max="100" step="any" value="50" disabled aria-label="分割线位置"></label><output id="split-position" for="split-range">50%</output><button id="reset-position" type="button" disabled>重置分割线（50%）</button></div>
<div class="footer"><span>聚焦滑块：← / → 微调 1 像素，Shift + ← / → 微调 10 像素，Home / End 到两端。</span><span id="image-dimensions"></span></div>
</section>
<section class="panel" aria-label="上传自己的图片">
<div class="uploads"><label>上传 BEFORE 图片<input id="before-upload" type="file" accept="image/*"></label><label>上传 AFTER 图片<input id="after-upload" type="file" accept="image/*"></label></div>
<p class="small">图片仅在本地浏览器中读取。两侧尺寸相同后应用；尺寸不匹配时保留当前对比。切换图像或来源会恢复评估图片。</p>
<p id="status-message" role="status" aria-live="polite"></p><p id="error-message" role="alert" hidden></p>
</section>
</main>
<script id="comparison-data" type="application/json">__COMPARISON_DATA__</script>
<script>
"use strict";
(() => {
  const data = JSON.parse(document.getElementById("comparison-data").textContent);
  const byId = id => document.getElementById(id);
  const stage = byId("compare-stage"), range = byId("split-range"), select = byId("frame-select");
  const source = byId("before-source"), status = byId("status-message"), error = byId("error-message");
  const uploads = {before: byId("before-upload"), after: byId("after-upload")};
  const objectURLs = new Set(), decodingURLs = new Set();
  const pending = {before: null, after: null}, uploadVersion = {before: 0, after: 0};
  let pair = null, position = 0, frameVersion = 0, activeFrame = -1, pointerId = null;

  function message(text) { status.textContent = text; }
  function showError(text) { error.textContent = text; error.hidden = false; }
  function clearError() { error.textContent = ""; error.hidden = true; }
  function cleanupURLs() {
    const keep = new Set([pair?.before.url, pair?.after.url, pending.before?.url, pending.after?.url, ...decodingURLs]);
    for (const url of objectURLs) if (!keep.has(url)) { URL.revokeObjectURL(url); objectURLs.delete(url); }
  }
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve({width: image.naturalWidth, height: image.naturalHeight});
      image.onerror = () => reject(new Error("无法读取图片，请检查文件是否有效且未被移动。"));
      image.src = url;
    });
  }
  function setPosition(value) {
    if (!pair) return;
    position = Math.max(0, Math.min(pair.before.width, Number(value)));
    const percent = position / pair.before.width * 100;
    range.value = String(position);
    range.setAttribute("aria-valuetext", `${percent.toFixed(1)}%，${Number(position.toFixed(2))} 像素`);
    stage.style.setProperty("--split", `${percent}%`);
    byId("split-position").textContent = `${percent.toFixed(1)}% · ${Number(position.toFixed(2))} px`;
  }
  function paintImage(side, asset) {
    const image = byId(`${side}-image`);
    image.src = asset.url;
    image.style.width = asset.kind === "upload" ? "100%" : "300%";
    image.style.left = asset.kind === "upload" ? "0" : `${-asset.column * 100}%`;
    byId(`${side}-label`).textContent = `${side === "before" ? "BEFORE" : "AFTER"} · ${asset.label}`;
  }
  function activate(before, after, reset=false) {
    const fraction = pair && !reset ? position / pair.before.width : 0.5;
    pair = {before, after};
    stage.style.aspectRatio = `${before.width} / ${before.height}`;
    stage.dataset.width = String(before.width); stage.dataset.height = String(before.height);
    stage.dataset.beforeSource = before.kind; stage.dataset.afterSource = after.kind;
    paintImage("after", after); paintImage("before", before);
    range.max = String(before.width); range.disabled = false;
    byId("reset-position").disabled = false;
    byId("image-dimensions").textContent = `${before.width} × ${before.height}`;
    stage.hidden = false; byId("empty-state").hidden = true;
    setPosition(fraction * before.width);
    cleanupURLs();
  }
  function clearUploads() {
    for (const side of ["before", "after"]) {
      uploadVersion[side]++; pending[side] = null; uploads[side].value = "";
    }
    cleanupURLs();
  }
  function updateNavigation() {
    const index = Number(select.value);
    byId("prev-frame").disabled = !data.images.length || index <= 0;
    byId("next-frame").disabled = !data.images.length || index >= data.images.length - 1;
  }
  async function showFrame(index) {
    if (!data.images[index]) return;
    const version = ++frameVersion;
    select.value = String(index); updateNavigation(); clearUploads(); clearError();
    message("正在读取评估图片…");
    const frame = data.images[index];
    try {
      const dimensions = await loadImage(frame.url);
      if (version !== frameVersion) return;
      if (dimensions.width !== frame.width * 3 || dimensions.height !== frame.height)
        throw new Error("评估图片尺寸不符，需要 input | teacher | student 三联图。");
      activeFrame = index;
      const base = {url: frame.url, width: frame.width, height: frame.height};
      const before = source.value === "input"
        ? {...base, kind: "input", label: "原始输入", column: 0}
        : {...base, kind: "teacher", label: "教师模型", column: 1};
      activate(before, {...base, kind: "student", label: "蒸馏模型", column: 2}, true);
      message(frame.name);
    } catch (failure) {
      if (version !== frameVersion) return;
      if (activeFrame >= 0) { select.value = String(activeFrame); updateNavigation(); }
      showError(failure.message); message("保留上一次有效的对比图片。");
    }
  }
  async function upload(side) {
    const file = uploads[side].files[0];
    if (!file) return;
    const version = ++uploadVersion[side];
    const url = URL.createObjectURL(file);
    objectURLs.add(url); decodingURLs.add(url); clearError();
    try {
      const dimensions = await loadImage(url);
      if (version !== uploadVersion[side]) return;
      pending[side] = {...dimensions, url, kind: "upload", label: "本地上传"};
      const before = pending.before || pair?.before, after = pending.after || pair?.after;
      if (!before || !after) {
        message(`已读取 ${side.toUpperCase()}，请上传另一侧图片。`);
        return;
      }
      if (before.width !== after.width || before.height !== after.height) {
        showError(`图片尺寸不匹配：BEFORE ${before.width}×${before.height}，AFTER ${after.width}×${after.height}。请上传相同尺寸的另一张图片。`);
        message("待上传图片已保留，当前对比保持不变。");
        return;
      }
      pending.before = null; pending.after = null;
      activate(before, after);
      message("已应用本地上传图片。重置只会将分割线移回 50%。");
    } catch (failure) {
      if (version === uploadVersion[side]) { showError(failure.message); message("保留上一次有效的对比图片。"); }
    } finally {
      decodingURLs.delete(url); cleanupURLs();
    }
  }

  select.addEventListener("change", () => showFrame(Number(select.value)));
  source.addEventListener("change", () => { if (data.images.length) showFrame(Number(select.value)); });
  byId("prev-frame").addEventListener("click", () => showFrame(Number(select.value) - 1));
  byId("next-frame").addEventListener("click", () => showFrame(Number(select.value) + 1));
  for (const side of ["before", "after"]) uploads[side].addEventListener("change", () => upload(side));
  range.addEventListener("input", () => setPosition(range.value));
  range.addEventListener("keydown", event => {
    if (!pair) return;
    const step = event.shiftKey ? 10 : 1;
    let next;
    if (event.key === "ArrowLeft") next = position - step;
    else if (event.key === "ArrowRight") next = position + step;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = pair.before.width;
    else return;
    event.preventDefault(); setPosition(next);
  });
  byId("reset-position").addEventListener("click", () => { if (pair) setPosition(pair.before.width / 2); });
  function movePointer(event) {
    if (!pair) return;
    const rectangle = stage.getBoundingClientRect();
    if (rectangle.width) setPosition(Math.round((event.clientX - rectangle.left) / rectangle.width * pair.before.width));
  }
  stage.addEventListener("pointerdown", event => {
    if (!pair || event.button !== 0 || pointerId !== null) return;
    event.preventDefault(); pointerId = event.pointerId;
    stage.setPointerCapture(pointerId); range.focus({preventScroll: true}); movePointer(event);
  });
  stage.addEventListener("pointermove", event => { if (event.pointerId === pointerId) movePointer(event); });
  function endPointer(event) {
    if (event.pointerId !== pointerId) return;
    pointerId = null;
  }
  stage.addEventListener("pointerup", endPointer); stage.addEventListener("pointercancel", endPointer);
  stage.addEventListener("lostpointercapture", endPointer);
  window.addEventListener("beforeunload", () => { for (const url of objectURLs) URL.revokeObjectURL(url); });

  const statusLabels = {running: "评估进行中", completed: "评估已完成", interrupted: "评估已中断", error: "评估发生错误"};
  byId("report-status").textContent = `${statusLabels[data.status]} · ${data.images.length} 张可用图片 · 可离线打开`;
  for (let index = 0; index < data.images.length; index++) {
    const option = document.createElement("option");
    option.value = String(index); option.textContent = `${index + 1}. ${data.images[index].name}`; select.appendChild(option);
  }
  select.disabled = source.disabled = !data.images.length;
  updateNavigation();
  if (data.images.length) showFrame(0);
  else message("尚无评估图片，上传两张相同尺寸的图片即可开始对比。");
})();
</script>
</body>
</html>
'''
