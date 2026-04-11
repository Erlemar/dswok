---
name: excalidraw
description: Generate Excalidraw diagram files (.excalidraw) from codebase analysis. Use ONLY when the user explicitly asks for excalidraw files or diagrams.
---

# Excalidraw Diagram Generator

Generate `.excalidraw` files directly from codebase analysis.

## Quick Start

**Trigger when user explicitly asks for excalidraw:**
```
"Create an excalidraw diagram of the system"
"Generate an excalidraw file for this codebase"
"Visualize this as an excalidraw diagram"
```

**Workflow:**
1. Analyze the codebase (any language/framework)
2. Identify components, services, databases, APIs
3. Map relationships and data flows
4. Generate valid `.excalidraw` JSON with dynamic IDs and labels

**No prerequisites:** Works without existing diagrams or specific file types.

---

## Critical Rules

### 1. NEVER Use Diamond Shapes

Diamond arrow connections are broken in raw Excalidraw JSON. Use styled rectangles instead:

| Semantic Meaning | Rectangle Style |
|------------------|-----------------|
| Orchestrator/Hub | Coral (`#ffa8a8`/`#c92a2a`) + strokeWidth: 3 |
| Decision Point | Orange (`#ffd8a8`/`#e8590c`) + dashed stroke |

### 2. Labels Require TWO Elements

The `label` property does NOT work in raw JSON. Every labeled shape needs:

```json
// 1. Shape with boundElements reference
{
  "id": "my-box",
  "type": "rectangle",
  "boundElements": [{ "type": "text", "id": "my-box-text" }]
}

// 2. Separate text element with containerId
{
  "id": "my-box-text",
  "type": "text",
  "containerId": "my-box",
  "text": "My Label"
}
```

### 3. Elbow Arrows Need Three Properties

For 90-degree corners (not curved):

```json
{
  "type": "arrow",
  "roughness": 0,
  "roundness": null,
  "elbowed": true
}
```

### 4. Arrow Edge Calculations

Arrows must start/end at shape edges, not centers:

| Edge | Formula |
|------|---------|
| Top | `(x + width/2, y)` |
| Bottom | `(x + width/2, y + height)` |
| Left | `(x, y + height/2)` |
| Right | `(x + width, y + height/2)` |

---

## Element Types

| Type | Use For |
|------|---------|
| `rectangle` | Services, databases, containers, orchestrators |
| `ellipse` | Users, external systems, start/end points |
| `text` | Labels inside shapes, titles, annotations |
| `arrow` | Data flow, connections, dependencies |
| `line` | Grouping boundaries, separators |

---

## Workflow

### Step 1: Analyze Codebase

Discover components by looking for:

| Codebase Type | What to Look For |
|---------------|------------------|
| Monorepo | `packages/*/package.json`, workspace configs |
| Microservices | Service definitions, container configs |
| Infrastructure | Configuration files, deployment manifests |
| Backend API | Route definitions, controllers, DB models |
| Frontend | Component hierarchy, API calls |

**Use tools:**
- `Glob` → `**/package.json`, config files
- `Grep` → `app.get`, `@Controller`, `CREATE TABLE`
- `Read` → README, config files, entry points

### Step 2: Plan Layout

**Vertical flow (most common):**
```
Row 1: Users/Entry points (y: 100)
Row 2: Frontend/Gateway (y: 230)
Row 3: Orchestration (y: 380)
Row 4: Services (y: 530)
Row 5: Data layer (y: 680)

Columns: x = 100, 300, 500, 700, 900
Element size: 160-200px x 80-90px
```

### Step 3: Generate Elements

For each component:
1. Create shape with unique `id`
2. Add `boundElements` referencing text
3. Create text with `containerId`
4. Choose color based on type (see palette below)

### Step 4: Add Connections

For each relationship:
1. Calculate source edge point
2. Plan elbow route (avoid overlaps)
3. Create arrow with `points` array
4. Match stroke color to destination type

### Step 5: Add Grouping (Optional)

For logical groupings:
- Large transparent rectangle with `strokeStyle: "dashed"`
- Standalone text label at top-left

### Step 6: Validate and Write

Run validation checklist before writing. Save to `/tmp/` or user-specified path.

---

## Quick Arrow Reference

**Straight down:**
```json
{ "points": [[0, 0], [0, 110]], "x": 590, "y": 290 }
```

**L-shape (left then down):**
```json
{ "points": [[0, 0], [-325, 0], [-325, 125]], "x": 525, "y": 420 }
```

**U-turn (callback):**
```json
{ "points": [[0, 0], [50, 0], [50, -125], [20, -125]], "x": 710, "y": 440 }
```

**Arrow width/height** = bounding box of points:
```
points [[0,0], [-440,0], [-440,70]] → width=440, height=70
```

**Multiple arrows from same edge** — stagger positions:
```
5 arrows: 20%, 35%, 50%, 65%, 80% across edge width
```

---

## Default Color Palette

| Component | Background | Stroke |
|-----------|------------|--------|
| Frontend | `#a5d8ff` | `#1971c2` |
| Backend/API | `#d0bfff` | `#7048e8` |
| Database | `#b2f2bb` | `#2f9e44` |
| Storage | `#ffec99` | `#f08c00` |
| AI/ML | `#e599f7` | `#9c36b5` |
| External APIs | `#ffc9c9` | `#e03131` |
| Orchestration | `#ffa8a8` | `#c92a2a` |
| Message Queue | `#fff3bf` | `#fab005` |
| Cache | `#ffe8cc` | `#fd7e14` |
| Users | `#e7f5ff` | `#1971c2` |

---

## Validation Checklist

Before writing file:
- [ ] Every shape with label has boundElements + text element
- [ ] Text elements have containerId matching shape
- [ ] Multi-point arrows have `elbowed: true`, `roundness: null`
- [ ] Arrow x,y = source shape edge point
- [ ] Arrow final point offset reaches target edge
- [ ] No diamond shapes
- [ ] No duplicate IDs

---

## Common Issues

| Issue | Fix |
|-------|-----|
| Labels don't appear | Use TWO elements (shape + text), not `label` property |
| Arrows curved | Add `elbowed: true`, `roundness: null`, `roughness: 0` |
| Arrows floating | Calculate x,y from shape edge, not center |
| Arrows overlapping | Stagger start positions across edge |

---

## Output

- **Location:** `/tmp/` or user-specified path
- **Filename:** Descriptive, e.g., `system-architecture.excalidraw`
- **Testing:** Open in [excalidraw.com](https://excalidraw.com) or VS Code Excalidraw extension
