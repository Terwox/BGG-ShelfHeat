"""
Stage 6: Interactive heatmap generation.

Produces a self-contained HTML file with the shelf photo embedded as base64,
colored SVG polygon overlays showing play recency, interactive hover tooltips,
and a color legend. No external dependencies — one .html file, done.
"""

import base64
import html as html_mod
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Color system: continuous sqrt scale, colorblind-accessible
# ---------------------------------------------------------------------------
# Two palettes: "classic" (green→red) and "cb" (blue→red, colorblind-safe).
# sqrt scale with 3-year midpoint: sqrt(days / MAX) where MAX = 4× midpoint.
# All colors bright enough to read at 30% opacity over photos.
# Toggle between palettes client-side via button in the HTML.

import math

# DEFAULT palette: green → yellow → red (intuitive stoplight)
_GRADIENT_STOPS = [
    (0.00,  34, 197, 94),   # #22c55e — green (just played)
    (0.20,  74, 195, 80),   # #4ac350 — lime-green
    (0.40, 160, 190, 60),   # #a0be3c — yellow-green
    (0.50, 220, 180, 40),   # #dcb428 — gold (3yr midpoint)
    (0.65, 240, 140, 50),   # #f08c32 — amber
    (0.80, 240,  90, 60),   # #f05a3c — red-orange
    (1.00, 220,  50, 50),   # #dc3232 — red (ancient)
]

# Colorblind-safe alternate: blue → purple → red
_CB_GRADIENT_STOPS = [
    (0.00,  50, 175, 215),  # #32afd7 — bright cyan
    (0.20,  80, 150, 195),  # #5096c3 — steel blue
    (0.35, 115, 125, 175),  # #737daf — periwinkle
    (0.50, 155, 100, 155),  # #9b649b — dusty purple (3yr midpoint)
    (0.65, 185,  95, 140),  # #b95f8c — rose-mauve
    (0.80, 230,  85, 100),  # #e65564 — coral/salmon
    (1.00, 255, 107,  53),  # #ff6b35 — bright orange
]

# Non-gradient special colors (same for both palettes)
COLOR_NEVER_PLAYED = "#e839a0"       # hot pink
COLOR_NOT_IN_COLLECTION = "#7878b0"  # muted lavender
COLOR_UNIDENTIFIED = "#555555"       # neutral gray

# Scale config: sqrt(days / MAX_DAYS) where MAX = 4× midpoint.
# This places midpoint_days at exactly t = 0.5 on the gradient.
MIDPOINT_DAYS = 3 * 365   # 3 years = gradient midpoint
MAX_DAYS = 4 * MIDPOINT_DAYS  # 12 years = fully "ancient" (cyan)


def _lerp_color(t: float) -> str:
    """Interpolate the gradient at position t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))

    for i in range(len(_GRADIENT_STOPS) - 1):
        t0, r0, g0, b0 = _GRADIENT_STOPS[i]
        t1, r1, g1, b1 = _GRADIENT_STOPS[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = int(r0 + (r1 - r0) * f)
            g = int(g0 + (g1 - g0) * f)
            b = int(b0 + (b1 - b0) * f)
            return f"#{r:02x}{g:02x}{b:02x}"

    _, r, g, b = _GRADIENT_STOPS[-1]
    return f"#{r:02x}{g:02x}{b:02x}"


def days_to_color(days_since: int | None) -> str:
    """Map days-since-last-played to a hex color. sqrt scale, 3yr midpoint."""
    if days_since is None or days_since < 0:
        return COLOR_NEVER_PLAYED
    # sqrt scale: 0→0, MIDPOINT_DAYS→0.5, MAX_DAYS→1.0
    t = math.sqrt(min(days_since, MAX_DAYS) / MAX_DAYS)
    return _lerp_color(t)


# Bucket names for summary counting (used in stats/JSON export)
BUCKET_THRESHOLDS = [
    (180,  "played_recent"),
    (365,  "played_6_12mo"),
    (730,  "played_1_2yr"),
    (1095, "played_2_3yr"),
]

def _days_to_bucket(days: int | None) -> str:
    """Map days to a named bucket for summary stats."""
    if days is None:
        return "never_played"
    for threshold, name in BUCKET_THRESHOLDS:
        if days <= threshold:
            return name
    return "played_3plus"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_item(
    identification: dict | None,
    collection_match: dict | None,
) -> tuple[str, str, str]:
    """
    Classify a detected shelf item into a heatmap category.

    Returns (category_key, hex_color, human_label).
    Color is continuous (log-scale gradient) for played games,
    with standout colors for never-played / not-in-collection / unidentified.
    """
    if identification is None:
        return "unidentified", COLOR_UNIDENTIFIED, "Unidentified"

    if collection_match is None:
        return "not_in_collection", COLOR_NOT_IN_COLLECTION, "Not in collection"

    play_count = collection_match.get("play_count", 0)
    last_played = collection_match.get("last_played")

    if play_count == 0 and not last_played:
        return "never_played", COLOR_NEVER_PLAYED, "Never played"

    # Has plays but no recorded date — treat as very old
    if not last_played:
        color = days_to_color(MAX_DAYS)  # max end of gradient
        return "played_3plus", color, f"{play_count} plays (no date)"

    days = _days_since(last_played)
    bucket = _days_to_bucket(days)
    color = days_to_color(days)
    plays = f" ({play_count} plays)" if play_count else ""

    # Human-readable label
    if days == 0:
        label = f"Today{plays}"
    elif days == 1:
        label = f"Yesterday{plays}"
    elif days < 30:
        label = f"{days}d ago{plays}"
    elif days < 365:
        label = f"{days // 30}mo ago{plays}"
    else:
        label = f"{days // 365}yr {(days % 365) // 30}mo ago{plays}"

    return bucket, color, label


def _days_since(date_str: str) -> int:
    """Days between a YYYY-MM-DD string and now."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except ValueError:
        return MAX_DAYS_LOG  # unparseable → ancient
    delta = datetime.now() - d
    return max(0, delta.days)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(items: list[dict]) -> dict:
    """Count items per category."""
    by_cat: dict[str, int] = {}
    for item in items:
        cat = item["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1

    identified = sum(1 for it in items if it.get("identification"))
    matched = sum(1 for it in items if it.get("collection_match"))

    return {
        "total_detected": len(items),
        "identified": identified,
        "matched": matched,
        "never_played": by_cat.get("never_played", 0),
        "not_in_collection": by_cat.get("not_in_collection", 0),
        "unidentified": by_cat.get("unidentified", 0),
        "by_category": by_cat,
    }


# ---------------------------------------------------------------------------
# Photo encoding
# ---------------------------------------------------------------------------

def _encode_photo(photo_path: str, max_dim: int = 2048) -> tuple[str, str]:
    """Read a photo, resize if large, return (base64_string, mime_type)."""
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, "image/jpeg"


# ---------------------------------------------------------------------------
# SVG + HTML builders
# ---------------------------------------------------------------------------

def _build_polygon_svg(item: dict) -> str:
    """Build one SVG <polygon> element from an item dict."""
    points = " ".join(f"{x},{y}" for x, y in item["polygon"])

    # Tooltip data blob
    tip: dict = {"label": item.get("label", ""), "cat": item.get("category", "")}
    ident = item.get("identification")
    if ident:
        tip["name"] = html_mod.unescape(ident.get("game_name", ""))
        tip["method"] = ident.get("method", "")
        tip["id_confidence"] = ident.get("confidence", 0)
    match = item.get("collection_match")
    if match:
        tip["plays"] = match.get("play_count", 0)
        tip["last_played"] = match.get("last_played") or "Never"
        if match.get("user_rating"):
            tip["rating"] = match["user_rating"]
        if match.get("bgg_id"):
            tip["bgg_id"] = match["bgg_id"]

    # Store days_since for client-side palette toggle
    if match and match.get("last_played"):
        tip["days"] = _days_since(match["last_played"])
    elif item.get("category") in ("played_3plus",):
        tip["days"] = MAX_DAYS  # has plays but no date
    # else: days not set = special category (never_played, unidentified, etc.)

    info_attr = html_mod.escape(json.dumps(tip, ensure_ascii=True))

    return (
        f'<polygon points="{points}" '
        f'fill="{item["color"]}" fill-opacity="0.30" '
        f'stroke="{item["color"]}" stroke-width="2" stroke-opacity="0.8" '
        f'data-info="{info_attr}" class="gp"/>'
    )


def _build_legend_html(summary: dict) -> str:
    """Build the legend panel with continuous gradient bar + special colors."""
    by_cat = summary.get("by_category", {})

    # Build CSS gradient string from our stops
    grad_parts = []
    for t, r, g, b in _GRADIENT_STOPS:
        grad_parts.append(f"rgb({r},{g},{b}) {t * 100:.0f}%")
    grad_css = ", ".join(grad_parts)

    # Count played games (any bucket starting with "played_")
    played_count = sum(v for k, v in by_cat.items() if k.startswith("played_"))

    parts = ['<h4>Play Recency</h4>']

    # Gradient bar with labels
    parts.append(
        f'<div class="lg-grad-wrap">'
        f'<div class="lg-grad" style="background:linear-gradient(to right,{grad_css})"></div>'
        f'<div class="lg-grad-labels">'
        f'<span>Today</span><span>3yr</span><span>12yr+</span>'
        f'</div>'
        f'</div>'
    )
    parts.append(
        f'<div class="lg-row">'
        f'<span class="lg-sw" style="background:transparent"></span>'
        f'Played'
        f'<span class="lg-n">{played_count}</span>'
        f'</div>'
    )

    # Special categories
    special = [
        (COLOR_NEVER_PLAYED, "Never played", by_cat.get("never_played", 0)),
        (COLOR_NOT_IN_COLLECTION, "Not in collection", by_cat.get("not_in_collection", 0)),
        (COLOR_UNIDENTIFIED, "Unidentified", by_cat.get("unidentified", 0)),
    ]
    for color, label, count in special:
        opacity = "1" if count > 0 else "0.35"
        parts.append(
            f'<div class="lg-row" style="opacity:{opacity}">'
            f'<span class="lg-sw" style="background:{color}"></span>'
            f'{label}'
            f'<span class="lg-n">{count}</span>'
            f'</div>'
        )

    # Palette toggle + edit persistence buttons
    parts.append(
        '<div style="margin-top:.6rem;text-align:center;display:flex;gap:.4rem;justify-content:center;flex-wrap:wrap">'
        '<button id="palToggle" style="background:#2a2a4a;color:#999;border:1px solid #3a3a5a;'
        'border-radius:5px;padding:.3rem .7rem;cursor:pointer;font-size:.72rem">'
        'Colorblind mode</button>'
        '<button id="btnExport" style="background:#2a2a4a;color:#999;border:1px solid #3a3a5a;'
        'border-radius:5px;padding:.3rem .7rem;cursor:pointer;font-size:.72rem">'
        '💾 Export edits</button>'
        '<button id="btnImport" style="background:#2a2a4a;color:#999;border:1px solid #3a3a5a;'
        'border-radius:5px;padding:.3rem .7rem;cursor:pointer;font-size:.72rem">'
        '📂 Import edits</button>'
        '<input type="file" id="importFile" accept=".json" style="display:none">'
        '</div>'
    )

    return "\n".join(parts)


def _build_stats_html(summary: dict) -> str:
    """Build the stats bar HTML."""
    s = summary
    return (
        f'<div class="st"><span class="sv">{s["total_detected"]}</span> detected</div>'
        f'<div class="st"><span class="sv">{s["identified"]}</span> identified</div>'
        f'<div class="st"><span class="sv">{s["matched"]}</span> matched</div>'
        f'<div class="st"><span class="sv" style="color:#800080">{s["never_played"]}</span> never played</div>'
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_heatmap(
    photo_path: str,
    items: list[dict],
    detection_size: tuple[int, int],
    output_path: str,
    bgg_user: str | None = None,
    collection_names: list[str] | None = None,
    collection_games: list[dict] | None = None,
) -> str:
    """
    Generate a self-contained HTML heatmap file.

    Args:
        photo_path:        Path to the original shelf photo.
        items:             Classified items — each needs polygon, identification,
                           collection_match, category, color, label.
        detection_size:    (width, height) of the detection-res image.
        output_path:       Where to write the .html file.
        bgg_user:          Optional BGG username for the page title.
        collection_names:  (DEPRECATED) Simple list of game name strings.
        collection_games:  Full list of game dicts with name, plays, last_played.
                           Used in the edit UI for search + play data lookup.

    Returns the output file path.
    """
    photo_b64, photo_mime = _encode_photo(photo_path)
    summary = compute_summary(items)
    det_w, det_h = detection_size

    polygons_svg = "\n        ".join(_build_polygon_svg(it) for it in items)
    stats_html = _build_stats_html(summary)
    legend_html = _build_legend_html(summary)
    title_suffix = f" &mdash; {html_mod.escape(bgg_user)}" if bgg_user else ""

    # Build game data for the edit UI — rich objects with play info
    if collection_games:
        # Full collection with play data: [{name, plays, last_played}, ...]
        seen = set()
        deduped = []
        for g in sorted(collection_games, key=lambda x: x.get("name", "")):
            if g["name"] not in seen:
                seen.add(g["name"])
                deduped.append(g)
        game_list_json = json.dumps(deduped, ensure_ascii=True)
    elif collection_names:
        # Legacy: just names, wrap as objects
        game_list_json = json.dumps(
            [{"name": n, "plays": 0, "last_played": ""} for n in sorted(set(collection_names))],
            ensure_ascii=True,
        )
    else:
        names = set()
        for it in items:
            m = it.get("collection_match")
            if m and m.get("name"):
                names.add(m["name"])
            ident = it.get("identification")
            if ident and ident.get("game_name"):
                names.add(ident["game_name"])
        game_list_json = json.dumps(
            [{"name": n, "plays": 0, "last_played": ""} for n in sorted(names)],
            ensure_ascii=True,
        )

    # Build items JSON for the edit UI (so edits can write back)
    items_json = json.dumps(
        [{"id": it.get("id", i), "category": it.get("category", ""),
          "name": (it.get("identification") or {}).get("game_name", ""),
          "polygon": it.get("polygon", [])}
         for i, it in enumerate(items)],
        ensure_ascii=True,
    )

    page = _HTML_TEMPLATE.format(
        title_suffix=title_suffix,
        photo_mime=photo_mime,
        photo_b64=photo_b64,
        det_w=det_w,
        det_h=det_h,
        polygons_svg=polygons_svg,
        stats_html=stats_html,
        legend_html=legend_html,
        game_list_json=game_list_json,
        items_json=items_json,
        max_days=MAX_DAYS,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"[heatmap] Wrote {out} ({size_kb:.0f} KB)")
    return str(out)


# ---------------------------------------------------------------------------
# HTML template — self-contained, no external resources
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShelfHeat{title_suffix}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{
  background:#0f0f23;color:#ccc;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  min-height:100vh;display:flex;flex-direction:column;
}}
header{{
  padding:.8rem 1.5rem;display:flex;justify-content:space-between;
  align-items:center;border-bottom:1px solid #2a2a4a;flex-shrink:0;
}}
header h1{{font-size:1.2rem;font-weight:600;color:#e8e8f0}}
.stats{{display:flex;gap:1.4rem;font-size:.82rem;color:#999}}
.st{{display:flex;align-items:baseline;gap:.35rem}}
.sv{{font-weight:700;font-size:1rem;color:#ddd}}
main{{
  flex:1;display:flex;justify-content:center;align-items:center;
  padding:1.5rem;overflow:auto;
}}
.shelf-wrap{{
  position:relative;display:inline-block;
  max-width:95vw;max-height:82vh;
}}
.shelf-wrap img{{
  display:block;max-width:95vw;max-height:82vh;
  border-radius:6px;box-shadow:0 6px 28px rgba(0,0,0,.55);
}}
.shelf-wrap svg{{
  position:absolute;top:0;left:0;width:100%;height:100%;
  border-radius:6px;
}}
.gp{{
  cursor:pointer;
  transition:fill-opacity .15s,stroke-width .15s;
}}
.gp:hover{{
  fill-opacity:.55;stroke-width:3;
}}
.tip{{
  position:fixed;padding:.65rem .9rem;
  background:#1a1a3af0;border:1px solid #3a3a5a;
  border-radius:8px;font-size:.82rem;pointer-events:none;
  opacity:0;transition:opacity .12s;z-index:1000;
  max-width:300px;box-shadow:0 4px 18px rgba(0,0,0,.45);
  backdrop-filter:blur(6px);
}}
.tip.vis{{opacity:1}}
.tip h3{{font-size:.92rem;margin-bottom:.3rem;color:#f0f0ff}}
.tip .dt{{color:#9898b8;font-size:.78rem;line-height:1.45}}
.tip .dt span{{color:#bbb}}
.legend{{
  position:fixed;bottom:1.2rem;right:1.2rem;
  background:#16162ef0;border:1px solid #2e2e50;
  border-radius:10px;padding:.85rem 1.1rem;font-size:.78rem;
  backdrop-filter:blur(8px);z-index:900;
  box-shadow:0 3px 14px rgba(0,0,0,.35);
}}
.legend h4{{margin-bottom:.5rem;font-size:.82rem;color:#b8b8d0}}
.lg-row{{
  display:flex;align-items:center;gap:.5rem;
  margin:.25rem 0;white-space:nowrap;
}}
.lg-sw{{
  width:13px;height:13px;border-radius:3px;flex-shrink:0;
}}
.lg-n{{
  margin-left:auto;padding-left:.6rem;
  font-weight:600;color:#8888a8;min-width:1.2em;text-align:right;
}}
.lg-grad-wrap{{margin:.5rem 0 .6rem}}
.lg-grad{{
  height:14px;border-radius:4px;width:100%;
}}
.lg-grad-labels{{
  display:flex;justify-content:space-between;
  font-size:.68rem;color:#7878a0;margin-top:.15rem;
}}
.empty-msg{{
  text-align:center;color:#666;font-size:1.1rem;
  padding:3rem;
}}
/* Edit panel */
.edit-panel{{
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  background:#1a1a3af8;border:1px solid #4a4a6a;border-radius:12px;
  padding:1.4rem;width:340px;z-index:1100;
  box-shadow:0 8px 40px rgba(0,0,0,.6);backdrop-filter:blur(10px);
  display:none;
}}
.edit-panel.vis{{display:block}}
.edit-panel h3{{margin-bottom:.8rem;color:#e8e8f0;font-size:1rem}}
.edit-panel input{{
  width:100%;padding:.5rem .7rem;background:#0e0e20;color:#ddd;
  border:1px solid #3a3a5a;border-radius:6px;font-size:.88rem;
  margin-bottom:.5rem;outline:none;
}}
.edit-panel input:focus{{border-color:#6a6aff}}
.edit-results{{
  max-height:200px;overflow-y:auto;margin-bottom:.8rem;
}}
.edit-results .er{{
  padding:.4rem .6rem;cursor:pointer;border-radius:4px;
  font-size:.84rem;color:#bbb;
}}
.edit-results .er:hover,.edit-results .er.sel{{
  background:#2a2a5a;color:#fff;
}}
.edit-btns{{display:flex;gap:.5rem;justify-content:flex-end}}
.edit-btns button{{
  padding:.45rem .9rem;border:none;border-radius:5px;
  cursor:pointer;font-size:.82rem;font-weight:600;
}}
.btn-save{{background:#2a8a4a;color:#fff}}
.btn-save:hover{{background:#35a55a}}
.btn-notgame{{background:#6a2a2a;color:#fff}}
.btn-notgame:hover{{background:#8a3535}}
.btn-cancel{{background:#333;color:#aaa}}
.btn-cancel:hover{{background:#444}}
.edit-overlay{{
  position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.4);z-index:1050;display:none;
}}
.edit-overlay.vis{{display:block}}
</style>
</head>
<body>
<header>
  <h1>ShelfHeat{title_suffix}</h1>
  <nav class="stats">{stats_html}</nav>
</header>
<main>
  <div class="shelf-wrap">
    <img src="data:{photo_mime};base64,{photo_b64}" alt="Board game shelf">
    <svg viewBox="0 0 {det_w} {det_h}" preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg">
        {polygons_svg}
    </svg>
  </div>
</main>
<div class="legend">{legend_html}</div>
<div class="tip" id="tip"></div>
<div class="edit-overlay" id="editOverlay"></div>
<div class="edit-panel" id="editPanel">
  <h3 id="editTitle">Edit identification</h3>
  <input type="text" id="editSearch" placeholder="Search collection or type any name..." autocomplete="off">
  <div class="edit-results" id="editResults"></div>
  <div class="edit-btns">
    <button class="btn-notgame" id="btnNotGame">Not a game</button>
    <button class="btn-cancel" id="btnCancel">Cancel</button>
    <button class="btn-save" id="btnSave">Save</button>
  </div>
</div>
<script>
(function(){{
  const tip=document.getElementById('tip');
  const editPanel=document.getElementById('editPanel');
  const editOverlay=document.getElementById('editOverlay');
  const editSearch=document.getElementById('editSearch');
  const editResults=document.getElementById('editResults');
  const editTitle=document.getElementById('editTitle');
  const gameList={game_list_json};  // array of game objects
  const gameLookup={{}};  // name -> game data
  gameList.forEach(function(g){{gameLookup[g.name]=Object.assign({{}},{{plays:g.plays||0,last_played:g.last_played||''}});}});
  const itemsData={items_json};
  let editTarget=null;  // the polygon element being edited
  let editSelected=null;  // selected game name

  // Hover tooltips
  document.querySelectorAll('.gp').forEach((p,idx)=>{{
    p.addEventListener('mouseenter',()=>{{
      const d=JSON.parse(p.dataset.info);
      let h='';
      if(d.name){{
        h+='<h3>'+esc(d.name)+'</h3><div class="dt">';
        if(d.label) h+=esc(d.label)+'<br>';
        if(d.method) h+='ID method: <span>'+d.method+'</span><br>';
        if(d.plays!==undefined) h+='Plays: <span>'+d.plays+'</span><br>';
        if(d.last_played) h+='Last played: <span>'+d.last_played+'</span><br>';
        if(d.rating) h+='Your rating: <span>'+d.rating+'/10</span>';
        h+='</div>';
      }}else{{
        h='<h3>Unidentified</h3><div class="dt">'+esc(d.label||'Could not identify')+'<br><em>Click to identify</em></div>';
      }}
      tip.innerHTML=h;
      tip.classList.add('vis');
    }});
    p.addEventListener('mousemove',e=>{{
      let x=e.clientX+14, y=e.clientY+14;
      const r=tip.getBoundingClientRect();
      if(x+r.width>window.innerWidth) x=e.clientX-r.width-10;
      if(y+r.height>window.innerHeight) y=e.clientY-r.height-10;
      tip.style.left=x+'px';
      tip.style.top=y+'px';
    }});
    p.addEventListener('mouseleave',()=>tip.classList.remove('vis'));

    // Click to edit
    p.addEventListener('click',()=>{{
      tip.classList.remove('vis');
      editTarget=p;
      editSelected=null;
      const d=JSON.parse(p.dataset.info);
      editTitle.textContent=d.name?'Edit: '+d.name:'Identify this game';
      editSearch.value='';
      renderResults('');
      editPanel.classList.add('vis');
      editOverlay.classList.add('vis');
      editSearch.focus();
      // highlight the polygon
      p.setAttribute('stroke','#ffffff');
      p.setAttribute('stroke-width','4');
      p.setAttribute('fill-opacity','0.5');
    }});
  }});

  // Search filter
  editSearch.addEventListener('input',()=>renderResults(editSearch.value));

  function renderResults(q){{
    const lq=q.toLowerCase();
    const matches=lq?gameList.filter(g=>g.name.toLowerCase().includes(lq)).slice(0,20):gameList.slice(0,20);
    editResults.innerHTML=matches.map(g=>{{
      const extra=g.last_played?' ('+g.last_played.slice(0,10)+')':g.plays>0?' ('+g.plays+' plays)':'';
      return '<div class="er'+(g.name===editSelected?' sel':'')+'" data-name="'+esc(g.name)+'">'+esc(g.name)+'<span style="color:#888;font-size:.75rem">'+esc(extra)+'</span></div>';
    }}).join('');
    editResults.querySelectorAll('.er').forEach(el=>{{
      el.addEventListener('click',()=>{{
        if(editSelected===el.dataset.name){{
          // Double-click = save
          document.getElementById('btnSave').dispatchEvent(new Event('click'));
          return;
        }}
        editSelected=el.dataset.name;
        editResults.querySelectorAll('.er').forEach(e=>e.classList.remove('sel'));
        el.classList.add('sel');
      }});
    }});
  }}

  // Save edit — pulls in play data + recolors polygon
  document.getElementById('btnSave').addEventListener('click',()=>{{
    const name=editSelected||editSearch.value.trim();
    if(!editTarget||!name) return;
    const d=JSON.parse(editTarget.dataset.info);
    d.name=name;
    d.label='Manually identified';
    d.method='manual';
    // Pull in play data from collection lookup
    const gd=gameLookup[name];
    if(gd){{
      d.plays=gd.plays||0;
      d.last_played=gd.last_played||'Never';
      // Compute days_since for color
      if(gd.last_played){{
        const lp=new Date(gd.last_played);
        if(!isNaN(lp)) d.days=Math.floor((Date.now()-lp.getTime())/86400000);
      }}
      d.cat=d.days!==undefined?'played':'never_played';
    }}else{{
      // Not in collection — clear any stale play data
      d.cat='not_in_collection';
      delete d.days;
      delete d.plays;
      delete d.last_played;
    }}
    editTarget.dataset.info=JSON.stringify(d);
    // Recolor this polygon
    const stops=palettes[currentPal];
    let c;
    if(d.days!==undefined) c=lerpColor(Math.sqrt(Math.min(d.days,maxDays)/maxDays),stops);
    else if(d.cat==='never_played') c=specialColors.never_played;
    else if(d.cat==='not_in_collection') c=specialColors.not_in_collection;
    else c=specialColors.unidentified;
    editTarget.setAttribute('fill',c);
    editTarget.setAttribute('stroke',c);
    // Update edits log
    logEdit(editTarget, name, 'identify');
    closeEdit();
  }});

  // Not a game
  document.getElementById('btnNotGame').addEventListener('click',()=>{{
    if(!editTarget) return;
    editTarget.style.display='none';  // hide the polygon
    logEdit(editTarget, null, 'not_a_game');
    closeEdit();
  }});

  // Cancel
  document.getElementById('btnCancel').addEventListener('click',closeEdit);
  editOverlay.addEventListener('click',closeEdit);
  document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeEdit()}});

  function closeEdit(){{
    editPanel.classList.remove('vis');
    editOverlay.classList.remove('vis');
    if(editTarget){{
      // restore polygon style
      const d=JSON.parse(editTarget.dataset.info);
      const color=editTarget.getAttribute('fill');
      editTarget.setAttribute('stroke-width','2');
      editTarget.setAttribute('fill-opacity','0.30');
    }}
    editTarget=null;
    editSelected=null;
  }}

  // Track edits for export
  const edits=[];
  function logEdit(poly, name, action){{
    const idx=[...document.querySelectorAll('.gp')].indexOf(poly);
    edits.push({{index:idx, name:name, action:action, time:new Date().toISOString()}});
    // Store in localStorage for persistence
    const key='shelfheat_edits_'+window.location.pathname;
    localStorage.setItem(key, JSON.stringify(edits));
    console.log('[ShelfHeat] Edit saved:', edits[edits.length-1]);
  }}

  // --- Palette definitions (must be before applyEdits) ---
  const palettes={{
    classic:[
      [0.00,34,197,94],[0.20,74,195,80],[0.40,160,190,60],
      [0.50,220,180,40],[0.65,240,140,50],[0.80,240,90,60],[1.00,220,50,50]
    ],
    cb:[
      [0.00,50,175,215],[0.20,80,150,195],[0.35,115,125,175],
      [0.50,155,100,155],[0.65,185,95,140],[0.80,230,85,100],[1.00,255,107,53]
    ]
  }};
  const specialColors={{never_played:'#e839a0',not_in_collection:'#7878b0',unidentified:'#555555'}};
  const maxDays={max_days};
  let currentPal='classic';

  function lerpColor(t,stops){{
    t=Math.max(0,Math.min(1,t));
    for(let i=0;i<stops.length-1;i++){{
      const[t0,r0,g0,b0]=stops[i];
      const[t1,r1,g1,b1]=stops[i+1];
      if(t>=t0&&t<=t1){{
        const f=t1>t0?(t-t0)/(t1-t0):0;
        const r=Math.round(r0+(r1-r0)*f);
        const g=Math.round(g0+(g1-g0)*f);
        const b=Math.round(b0+(b1-b0)*f);
        return '#'+[r,g,b].map(v=>v.toString(16).padStart(2,'0')).join('');
      }}
    }}
    const last=stops[stops.length-1];
    return '#'+[last[1],last[2],last[3]].map(v=>v.toString(16).padStart(2,'0')).join('');
  }}

  // Load edits from localStorage on page load
  // Restore from localStorage first
  const savedKey='shelfheat_edits_'+window.location.pathname;
  const saved=localStorage.getItem(savedKey);
  if(saved){{
    try{{
      const se=JSON.parse(saved);
      if(Array.isArray(se)&&se.length){{
        const n=applyEdits(se);
        if(n) console.log('[ShelfHeat] Loaded',n,'edits from localStorage');
      }}
    }}catch(ex){{}}
  }}

  // --- Edit persistence: sidecar JSON ---
  function editsFileName(){{
    const p=window.location.pathname;
    return p.replace(/_heatmap\.html$/,'_edits.json').replace(/^.*\//,'');
  }}

  function applyEdits(se){{
    const polys=document.querySelectorAll('.gp');
    let applied=0;
    se.forEach(e=>{{
      if(e.index>=0&&e.index<polys.length){{
        const p=polys[e.index];
        if(e.action==='not_a_game'){{
          p.style.display='none';
        }}else if(e.action==='identify'&&e.name){{
          const d=JSON.parse(p.dataset.info);
          d.name=e.name;d.label='Manually identified';d.method='manual';
          const gd=gameLookup[e.name];
          if(gd){{
            d.plays=gd.plays||0;
            d.last_played=gd.last_played||'Never';
            if(gd.last_played){{
              const lp=new Date(gd.last_played);
              if(!isNaN(lp)) d.days=Math.floor((Date.now()-lp.getTime())/86400000);
            }}
            d.cat=d.days!==undefined?'played':'never_played';
          }}else{{ d.cat='not_in_collection'; delete d.days; }}
          p.dataset.info=JSON.stringify(d);
          const stops=palettes[currentPal];
          let c;
          if(d.days!==undefined) c=lerpColor(Math.sqrt(Math.min(d.days,maxDays)/maxDays),stops);
          else if(d.cat==='never_played') c=specialColors.never_played;
          else if(d.cat==='not_in_collection') c=specialColors.not_in_collection;
          else c=specialColors.unidentified;
          p.setAttribute('fill',c);p.setAttribute('stroke',c);
        }}
        if(!edits.find(x=>x.index===e.index&&x.action===e.action&&x.name===e.name)) edits.push(e);
        applied++;
      }}
    }});
    return applied;
  }}

  // Try loading sidecar JSON via fetch (works with HTTP server)
  (async()=>{{
    if(edits.length) return;  // already loaded from localStorage
    try{{
      const r=await fetch('./'+editsFileName());
      if(r.ok){{
        const se=await r.json();
        if(Array.isArray(se)&&se.length){{
          const n=applyEdits(se);
          console.log('[ShelfHeat] Loaded',n,'edits from sidecar JSON');
        }}
      }}
    }}catch(ex){{}}
  }})();

  // Export edits → download JSON
  document.getElementById('btnExport').addEventListener('click',()=>{{
    if(!edits.length){{ alert('No edits to export'); return; }}
    const blob=new Blob([JSON.stringify(edits,null,2)],{{type:'application/json'}});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download=editsFileName();
    a.click();
    URL.revokeObjectURL(a.href);
  }});

  // Import edits ← load JSON file
  const importFile=document.getElementById('importFile');
  document.getElementById('btnImport').addEventListener('click',()=>importFile.click());
  importFile.addEventListener('change',()=>{{
    const f=importFile.files[0];
    if(!f) return;
    const reader=new FileReader();
    reader.onload=()=>{{
      try{{
        const se=JSON.parse(reader.result);
        if(!Array.isArray(se)){{ alert('Invalid edits file'); return; }}
        const n=applyEdits(se);
        // Merge into localStorage too
        const key='shelfheat_edits_'+window.location.pathname;
        localStorage.setItem(key,JSON.stringify(edits));
        console.log('[ShelfHeat] Imported',n,'edits from file');
        alert('Imported '+n+' edits');
      }}catch(ex){{ alert('Failed to parse edits: '+ex.message); }}
    }};
    reader.readAsText(f);
    importFile.value='';
  }});

  function esc(s){{
    const d=document.createElement('div');
    d.textContent=s;
    return d.innerHTML;
  }}

  // --- Palette toggle ---
  function recolor(palName){{
    const stops=palettes[palName];
    document.querySelectorAll('.gp').forEach(p=>{{
      const d=JSON.parse(p.dataset.info);
      let c;
      if(d.cat&&specialColors[d.cat]) c=specialColors[d.cat];
      else if(d.days!==undefined) c=lerpColor(Math.sqrt(Math.min(d.days,maxDays)/maxDays),stops);
      else c=specialColors.unidentified;
      p.setAttribute('fill',c);
      p.setAttribute('stroke',c);
    }});
    // Update gradient bar
    const bar=document.querySelector('.lg-grad');
    if(bar){{
      const parts=stops.map(s=>'rgb('+s[1]+','+s[2]+','+s[3]+') '+(s[0]*100)+'%');
      bar.style.background='linear-gradient(to right,'+parts.join(',')+')';
    }}
  }}

  const togBtn=document.getElementById('palToggle');
  if(togBtn){{
    togBtn.addEventListener('click',()=>{{
      currentPal=currentPal==='classic'?'cb':'classic';
      togBtn.textContent=currentPal==='classic'?'Colorblind mode':'Standard mode';
      recolor(currentPal);
    }});
  }}
}})();
</script>
</body>
</html>
"""
