---
name: excalidraw
description: Generate Excalidraw `.excalidraw.md` source files for DSWoK vault diagrams (charts, pipelines, conceptual sketches). Use ONLY when the user explicitly asks for an Excalidraw diagram or when the diagram-tooling order in `.claude/writing_style_guide.md` selects Excalidraw (non-flowchart visuals where Mermaid is the wrong tool).
---

# Excalidraw diagram generator (DSWoK)

Generate `.excalidraw.md` source files that the Obsidian Excalidraw plugin parses and auto-exports as `*.excalidraw.light.svg` / `*.excalidraw.dark.svg`. Notes embed via `![[name.excalidraw.light.svg]]`.

## When to use this skill

Per the diagram tooling order in `.claude/writing_style_guide.md`:

1. **Mermaid** for flowcharts, pipelines, dependency graphs, decision trees. Don't use this skill — write Mermaid inline in the note.
2. **Excalidraw** for everything else (charts, plots, conceptual diagrams, anything with non-grid layout). **Use this skill.**
3. **Hand-authored SVG** in `images/` only as a fallback when Excalidraw cannot produce the figure cleanly after a couple of iterations.

The skill is meant for vault diagrams. For codebase architecture diagrams or one-off non-vault visuals, write a `.excalidraw` JSON file directly to `/tmp/`; this skill's `.excalidraw.md` wrapper is specifically for the Obsidian plugin format.

## Where things live

- **Helpers**: `.claude/skills/excalidraw/helpers.py` — Python module with element builders, the `.excalidraw.md` wrapper, and a dark/light recovery function.
- **Source files**: write `<name>.excalidraw.md` to `Excalidraw/` in the vault root.
- **Auto-generated SVG exports**: the plugin writes `<name>.excalidraw.light.svg` and `.dark.svg` next to the source after the file is opened in Obsidian once.
- **Embeds in notes**: `![[<name>.excalidraw.light.svg]]` (use the light variant; Obsidian auto-swaps to dark in dark theme via the inverted CSS filter).

## Workflow

1. **Plan the figure.** Sketch the layout (axes, panels, curves, labels, legend) on paper or mentally. Most vault diagrams use a 900×600-ish canvas to match the existing CUPED / Multi-armed-bandits / AB-Tests illustrations. Plot areas typically run from x=110 to x=580 (470 wide) with y=80 (top) to y=510 (bottom, 430 tall).

2. **Write a Python script** that imports `helpers.py` and builds an `elements` list. Run the script to write `<name>.excalidraw.md` to `Excalidraw/`. The helper module is at `.claude/skills/excalidraw/helpers.py`. From a script, import via `sys.path.insert(0, "/Users/andreylukyanenko/Documents/DSWoK/.claude/skills/excalidraw"); from helpers import line, text, rect, build_excalidraw_md, ...`.

3. **Tell the user to open the file in Obsidian once.** The plugin parses the JSON, renders, and writes the two SVG exports. The first open is mandatory — without it, no SVGs exist and any embed in a note will be broken.

4. **Embed the light SVG in the target note.** `![[<name>.excalidraw.light.svg]]` — never embed the source `.excalidraw.md` directly.

5. **Recovery if only one variant exports.** If the plugin only wrote the dark or only the light SVG (e.g., user opened the file in only one theme), call `helpers.derive_light_from_dark(...)` or its inverse — the only difference between the two exports is a `filter="invert(93%) hue-rotate(180deg)"` attribute on the root `<svg>` element.

## Helper module API

All helpers live in `helpers.py`. The most-used:

| Helper | Returns | Purpose |
|---|---|---|
| `line(points, color, stroke_width, dashed)` | element dict | Multi-point line. Pass absolute coordinates; the helper handles the bounding-box-relative point conversion. |
| `text(x, y, content, font_size, color, text_align, font_family)` | element dict | Standalone text. For text *inside* a shape, use `bind_label()` instead. |
| `rect(x, y, w, h, color, stroke_width, bg)` | element dict | Rectangle. Useful for panel borders, legend boxes, container groupings. |
| `ellipse(x, y, w, h, color, stroke_width, bg)` | element dict | Ellipse. Users / start-end nodes / external systems in pipeline diagrams. |
| `arrow(points, color, elbowed, end_arrowhead, start_arrowhead)` | element dict | Arrow. Set `elbowed=True` for 90° corners (the helper sets `roughness=0` and `roundness=None` automatically — Excalidraw needs both for elbows). |
| `bind_label(shape, content, font_size)` | `[shape, text_el]` list | Attach a label inside a rectangle/ellipse using Excalidraw's `boundElements` + `containerId` mechanism. The plugin's `label` property does NOT work in raw JSON. |
| `rotate(element, degrees)` | element (mutated) | Rotate in place. Excalidraw uses radians; this converts. Common case: y-axis labels at `-90`. |
| `edge_point(shape, side)` | `(x, y)` tuple | Midpoint of one edge of a shape. For `side` in `'top'`, `'bottom'`, `'left'`, `'right'`. Use as arrow start/end. |
| `build_excalidraw_md(elements, background)` | string | Wrap the elements list into the `.excalidraw.md` markdown format the plugin reads. Plain JSON block, not `compressed-json`. |
| `reset_ids()` | None | Reset the element-id counter between independent diagrams in the same script. |
| `derive_light_from_dark(dark_path, light_path)` | None | Strip the dark-mode invert filter to produce a light SVG from a dark one. |

Module constants: `FONT_FAMILY = 5` (Excalifont), `ROUGHNESS = 1` (sketchy), accent colors `COLOR_RED`, `COLOR_BLUE`, `COLOR_GREEN`, `COLOR_GRAY_LIGHT`, `COLOR_TEXT_MUTED`, `COLOR_TEXT_DEFAULT`. System-architecture palette is in `helpers.PALETTE` as `(background, stroke)` tuples per role.

## Example: chart-style figure

```python
import sys, os
sys.path.insert(0, "/Users/andreylukyanenko/Documents/DSWoK/.claude/skills/excalidraw")
from helpers import (line, text, rect, rotate, build_excalidraw_md,
                     COLOR_RED, COLOR_BLUE, COLOR_GRAY_LIGHT)

elements = []

# Title
elements.append(text(280, 30, "Reliability diagram", font_size=28))

# Axes
elements.append(line([(110, 510), (580, 510)], stroke_width=2))  # x
elements.append(line([(110, 510), (110, 80)], stroke_width=2))   # y

# y=x reference (perfect calibration)
elements.append(line([(110, 510), (580, 80)],
                     color=COLOR_GRAY_LIGHT, stroke_width=2, dashed=True))

# Two miscalibration curves
over = [(110, 467), (157, 433), (204, 396), (251, 358), (298, 329),
        (345, 295), (392, 261), (439, 227), (486, 193), (533, 157), (580, 123)]
elements.append(line(over, color=COLOR_RED, stroke_width=3))

under = [(110, 510), (157, 489), (204, 470), (251, 440), (298, 395),
         (345, 295), (392, 195), (439, 150), (486, 121), (533, 101), (580, 80)]
elements.append(line(under, color=COLOR_BLUE, stroke_width=3))

# Axis labels
elements.append(text(280, 555, "Predicted probability", font_size=15))
elements.append(rotate(text(20, 280, "Empirical positive rate", font_size=15), -90))

# Legend
elements.append(rect(620, 120, 240, 130, color=COLOR_GRAY_LIGHT, stroke_width=1))
elements.append(line([(635, 175), (680, 175)],
                     color=COLOR_GRAY_LIGHT, stroke_width=2, dashed=True))
elements.append(text(690, 165, "Perfect calibration", font_size=14))
elements.append(line([(635, 205), (680, 205)], color=COLOR_RED, stroke_width=3))
elements.append(text(690, 195, "Over-confident", font_size=14))
elements.append(line([(635, 235), (680, 235)], color=COLOR_BLUE, stroke_width=3))
elements.append(text(690, 225, "Under-confident", font_size=14))

# Write source
ROOT = "/Users/andreylukyanenko/Documents/DSWoK"
out = os.path.join(ROOT, "Excalidraw/calibration_reliability_diagram.excalidraw.md")
with open(out, "w") as f:
    f.write(build_excalidraw_md(elements))
```

## Example: pipeline / system architecture

For diagrams with labeled boxes and arrows between them, use `rect`, `bind_label`, `edge_point`, and `arrow`:

```python
from helpers import rect, bind_label, edge_point, arrow, PALETTE, build_excalidraw_md

elements = []

bg, stroke = PALETTE["frontend"]
api_box = rect(100, 230, 180, 80, color=stroke, stroke_width=2, bg=bg)
elements.extend(bind_label(api_box, "API Gateway", font_size=16))

bg, stroke = PALETTE["database"]
db_box = rect(100, 530, 180, 80, color=stroke, stroke_width=2, bg=bg)
elements.extend(bind_label(db_box, "Postgres", font_size=16))

# Arrow from API bottom → DB top (with elbow if needed)
elements.append(arrow(
    [edge_point(api_box, "bottom"), edge_point(db_box, "top")],
    color="#1e1e1e", stroke_width=2, elbowed=False,
))
```

## Coordinate convention

- Top-left origin, like SVG. Larger `y` is **lower** on the canvas.
- For `line`, `arrow`, and other multi-point elements: pass absolute coordinates. The helper computes the bounding-box top-left as the element's `(x, y)` and shifts points relative to it.
- Rotation: `angle` is in radians. Use `rotate(element, degrees)` to set in degrees.

## Pitfalls (these will silently break the file)

| Issue | Fix |
|-------|-----|
| Labels don't appear inside shapes | Use `bind_label()`. The `label` property doesn't work in raw JSON; you need a separate text element with `containerId` pointing at the shape, and the shape needs `boundElements` pointing back. |
| Arrows curve at corners instead of 90° | Set `elbowed=True` on the arrow. The helper sets `roughness=0` and `roundness=None` for you (Excalidraw needs both). |
| Arrows float disconnected from shapes | Use `edge_point(shape, side)` for arrow start/end. Don't use shape centers. |
| Diagram renders, but text is in wrong font | `FONT_FAMILY` should be `5` (Excalifont). The old default `2` (Helvetica) was deprecated. |
| File parses but doesn't render | Check that every element has all of: `id`, `type`, `x`, `y`, `width`, `height`, `strokeColor`, `seed`, `version`, `versionNonce`, `isDeleted`, `updated`, `groupIds`, `frameId`, `boundElements`. The `base_props()` helper covers all the universal-required ones. |
| Diamond shape arrows broken | Don't use diamond shapes. Use `rect` with `roundness=None` and a distinctive stroke style instead. |
| `compressed-json` instead of `json` block | The plugin reads both, but `compressed-json` (LZ-String/deflate, base64-wrapped) is a pain to round-trip from CLI. Always emit plain `json` via `build_excalidraw_md()`. The plugin re-compresses on next save in Obsidian if compression is enabled in settings. |

## Validation before writing

- Every shape with a label has been wrapped via `bind_label()`.
- Multi-point arrows that should have 90° corners have `elbowed=True`.
- Arrow endpoints are computed via `edge_point(...)`, not eyeballed.
- No duplicate IDs (the global counter handles this; just don't construct elements outside the helpers).
- Every text element uses `FONT_FAMILY` (= 5) unless there's a specific reason to override.
- For curves on a chart, dense polylines (~10–15 points) approximate smoothness without bezier.

## Render and recovery

After writing the source file:

1. Tell the user: "Open `Excalidraw/<name>.excalidraw.md` in Obsidian once. The plugin will auto-export `<name>.excalidraw.light.svg` and `.dark.svg` next to the source."
2. If only one variant gets exported (the user only opened the file in one theme), call `derive_light_from_dark()` to produce the missing variant.
3. Verify the embed in the target note: `![[<name>.excalidraw.light.svg]]`.

## Output

- **Source file**: `Excalidraw/<name>.excalidraw.md` in the DSWoK vault root.
- **Generated SVGs**: `Excalidraw/<name>.excalidraw.light.svg` and `.dark.svg`, written by the plugin after first Obsidian open.
- **Note embed**: `![[<name>.excalidraw.light.svg]]` in the consuming note.
