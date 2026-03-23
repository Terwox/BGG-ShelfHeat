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
# Color palette (from ARCHITECTURE.md)
# ---------------------------------------------------------------------------

COLORS = {
    "played_recent":    "#00ff64",   # last 6 months — green
    "played_6_12mo":    "#80ff00",   # 6-12 months — yellow-green
    "played_1_2yr":     "#ffff00",   # 1-2 years — yellow
    "played_2_3yr":     "#ffa500",   # 2-3 years — orange
    "played_3plus":     "#ff3232",   # 3+ years — red
    "never_played":     "#800080",   # zero plays — purple
    "not_in_collection": "#6464a0",  # identified, not in BGG — blue-gray
    "unidentified":     "#555555",   # couldn't identify — gray
}

LEGEND_ITEMS = [
    ("played_recent",     "Played < 6 months"),
    ("played_6_12mo",     "Played 6–12 months"),
    ("played_1_2yr",      "Played 1–2 years"),
    ("played_2_3yr",      "Played 2–3 years"),
    ("played_3plus",      "Played 3+ years"),
    ("never_played",      "Never played"),
    ("not_in_collection", "Not in collection"),
    ("unidentified",      "Unidentified"),
]


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
    """
    if identification is None:
        return "unidentified", COLORS["unidentified"], "Unidentified"

    if collection_match is None:
        return "not_in_collection", COLORS["not_in_collection"], "Not in collection"

    play_count = collection_match.get("play_count", 0)
    last_played = collection_match.get("last_played")

    if play_count == 0 and not last_played:
        return "never_played", COLORS["never_played"], "Never played"

    # Has plays but no recorded date — treat as old
    if not last_played:
        return "played_3plus", COLORS["played_3plus"], f"{play_count} plays (no date)"

    months = _months_since(last_played)
    plays = f" ({play_count} plays)" if play_count else ""

    if months <= 6:
        return "played_recent", COLORS["played_recent"], f"{months}mo ago{plays}"
    if months <= 12:
        return "played_6_12mo", COLORS["played_6_12mo"], f"{months}mo ago{plays}"
    if months <= 24:
        return "played_1_2yr", COLORS["played_1_2yr"], f"~{months // 12}yr ago{plays}"
    if months <= 36:
        return "played_2_3yr", COLORS["played_2_3yr"], f"~{months // 12}yr ago{plays}"
    return "played_3plus", COLORS["played_3plus"], f"~{months // 12}yr ago{plays}"


def _months_since(date_str: str) -> int:
    """Approximate months between a YYYY-MM-DD string and now."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except ValueError:
        return 999  # unparseable → ancient
    now = datetime.now()
    return (now.year - d.year) * 12 + (now.month - d.month)


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
    tip: dict = {"label": item.get("label", "")}
    ident = item.get("identification")
    if ident:
        tip["name"] = ident.get("game_name", "")
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

    info_attr = html_mod.escape(json.dumps(tip, ensure_ascii=True))

    return (
        f'<polygon points="{points}" '
        f'fill="{item["color"]}" fill-opacity="0.30" '
        f'stroke="{item["color"]}" stroke-width="2" stroke-opacity="0.8" '
        f'data-info="{info_attr}" class="gp"/>'
    )


def _build_legend_html(summary: dict) -> str:
    """Build the legend panel inner HTML."""
    parts = ['<h4>Legend</h4>']
    for cat_key, label in LEGEND_ITEMS:
        count = summary.get("by_category", {}).get(cat_key, 0)
        color = COLORS[cat_key]
        opacity = "1" if count > 0 else "0.35"
        parts.append(
            f'<div class="lg-row" style="opacity:{opacity}">'
            f'<span class="lg-sw" style="background:{color}"></span>'
            f'{label}'
            f'<span class="lg-n">{count}</span>'
            f'</div>'
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
) -> str:
    """
    Generate a self-contained HTML heatmap file.

    Args:
        photo_path:     Path to the original shelf photo.
        items:          Classified items — each needs polygon, identification,
                        collection_match, category, color, label.
        detection_size: (width, height) of the detection-res image.
        output_path:    Where to write the .html file.
        bgg_user:       Optional BGG username for the page title.

    Returns the output file path.
    """
    photo_b64, photo_mime = _encode_photo(photo_path)
    summary = compute_summary(items)
    det_w, det_h = detection_size

    polygons_svg = "\n        ".join(_build_polygon_svg(it) for it in items)
    stats_html = _build_stats_html(summary)
    legend_html = _build_legend_html(summary)
    title_suffix = f" &mdash; {html_mod.escape(bgg_user)}" if bgg_user else ""

    page = _HTML_TEMPLATE.format(
        title_suffix=title_suffix,
        photo_mime=photo_mime,
        photo_b64=photo_b64,
        det_w=det_w,
        det_h=det_h,
        polygons_svg=polygons_svg,
        stats_html=stats_html,
        legend_html=legend_html,
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
.empty-msg{{
  text-align:center;color:#666;font-size:1.1rem;
  padding:3rem;
}}
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
<script>
(function(){{
  const tip=document.getElementById('tip');
  document.querySelectorAll('.gp').forEach(p=>{{
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
        h='<h3>Unidentified</h3><div class="dt">'+esc(d.label||'Could not identify')+'</div>';
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
  }});
  function esc(s){{
    const d=document.createElement('div');
    d.textContent=s;
    return d.innerHTML;
  }}
}})();
</script>
</body>
</html>
"""
