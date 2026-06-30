#!/usr/bin/env python3
"""Build the final 'The Art of Debugging' book cover (labyrinth) into <repo>/images/.

The cover canvas is US Letter aspect (8.5:11) so the PDF is a true Letter page
with no cropping.

Outputs:
  - The-Art-of-Debugging-book-cover-1536x1988.png : full-resolution flat cover
  - The-Art-of-Debugging-book-cover.png           : optimized display image
  - The-Art-of-Debugging-book-cover-<wxh>.png     : small thumbnail (size in name)
  - The-Art-of-Debugging-book-cover.pdf           : vector PDF at US Letter size
                                                    (612x792pt, 8.5x11in) to match the
                                                    ML-Engineering book
  - The-Art-of-Debugging-book-cover.svg           : self-contained, each text section
                                                    is its own Inkscape layer (editable)
  - The-Art-of-Debugging-book-cover.ora           : OpenRaster; opens in GIMP/Krita with
                                                    background + each text section as
                                                    separate named layers

This directory is self-contained: the only input is sources/labyrinth.png.

Requires: ImageMagick (magick) + rsvg-convert (librsvg). zopflipng optional
(used to further shrink the PNGs when present). No GIMP needed.
"""
import base64
import pathlib
import shutil
import subprocess
import tempfile
import zipfile

HERE = pathlib.Path(__file__).parent
IMAGES = HERE.parent.parent / "images"   # <repo>/images
ART = HERE / "sources" / "labyrinth.png"
STEM = "The-Art-of-Debugging-book-cover"

# --- canvas / layout geometry -------------------------------------------------
W, H = 1536, 1988          # US Letter aspect (8.5:11) portrait cover
ARTW, ARTH = 1536, 1024    # landscape art band placed inside the portrait
ART_Y = 600                # y where the art band begins (= top dark-band height)
BOT_H = H - ART_Y - ARTH   # bottom dark-band height (452)
CX = W // 2
FONT = "Avenir Next, Helvetica Neue, Arial"
DARK = "#02060e"           # near-black used at the very top/bottom edges

GOLD = "#f6c879"
ZOOM = 1.3   # enlarge the maze (center-crop) so the central bug reads at thumb size

DEFS = """  <defs>
    <linearGradient id="titleGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffe1a3"/>
      <stop offset="1" stop-color="#f0b347"/>
    </linearGradient>
    <linearGradient id="topScrim" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#02060e" stop-opacity="0.55"/>
      <stop offset="1" stop-color="#02060e" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="botScrim" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#02060e" stop-opacity="0"/>
      <stop offset="1" stop-color="#02060e" stop-opacity="0.6"/>
    </linearGradient>
  </defs>"""

# Each text section: label -> inner SVG markup on the full WxH canvas.
LAYERS = {
    "title-line-1": (
        f'<text x="{CX}" y="276" text-anchor="middle" font-family="{FONT}" '
        f'font-weight="600" font-size="142" fill="{GOLD}" letter-spacing="2">The Art of</text>'
    ),
    "title-line-2": (
        f'<text x="{CX}" y="473" text-anchor="middle" font-family="{FONT}" '
        f'font-weight="800" font-size="214" fill="url(#titleGrad)" letter-spacing="1">Debugging</text>'
    ),
    "subtitle": (
        f'<text x="{CX}" y="693" text-anchor="middle" font-family="{FONT}" '
        f'font-weight="500" font-size="86" fill="#ffffff" letter-spacing="17">An Open Book</text>'
    ),
    "author": (
        f'<text x="{CX}" y="1812" text-anchor="middle" font-family="{FONT}" '
        f'font-weight="500" font-size="86" fill="#ededed" letter-spacing="17">by Stas Bekman</text>'
    ),
}
ORDER = ["background", "title-line-1", "title-line-2", "subtitle", "author"]


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def edge_color(art: pathlib.Path, gravity: str) -> str:
    """Average color of a thin strip at the top (North) or bottom (South)."""
    out = subprocess.check_output([
        "magick", str(art), "-gravity", gravity, "-crop", f"{W}x8+0+0", "+repage",
        "-resize", "1x1!", "-format", "%[pixel:u]", "info:",
    ]).decode().strip()
    return out  # e.g. "srgb(8,12,20)"


def compose_art(out: pathlib.Path):
    """Pad zoomed landscape art into the portrait canvas: a top dark band of
    height ART_Y, the art band, then a bottom dark band filling the rest."""
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        norm = d / "art.png"
        zw, zh = int(ARTW * ZOOM), int(ARTH * ZOOM)
        run("magick", ART, "-resize", f"{zw}x{zh}^", "-gravity", "center",
            "-extent", f"{ARTW}x{ARTH}", norm)
        topc = edge_color(norm, "North")
        botc = edge_color(norm, "South")
        top = d / "top.png"
        bot = d / "bot.png"
        run("magick", "-size", f"{W}x{ART_Y}", f"gradient:{DARK}-{topc}", top)
        run("magick", "-size", f"{W}x{BOT_H}", f"gradient:{botc}-{DARK}", bot)
        run("magick", top, norm, bot, "-append", "-depth", "8", "-strip", out)


def render(svg_text: str, out_png: pathlib.Path):
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg_text)
        tmp = f.name
    subprocess.run(["rsvg-convert", "-o", str(out_png), tmp], check=True)
    pathlib.Path(tmp).unlink()


# US Letter page in px so rsvg-convert (96dpi: 1px=0.75pt) yields 612x792pt.
LETTER_W, LETTER_H = 816, 1056   # -> 612 x 792 pt (8.5 x 11 in)


def letter_pdf(svg_path: pathlib.Path, out_pdf: pathlib.Path):
    """Vector PDF at exact US Letter size (612x792pt). The canvas is already
    Letter aspect, so this is a clean uniform downscale with no cropping; text
    stays vector and the maze art is embedded as raster."""
    subprocess.run(
        ["rsvg-convert", "-f", "pdf", "-w", str(LETTER_W), "-h", str(LETTER_H),
         "-o", str(out_pdf), str(svg_path)],
        check=True,
    )


def svg_open() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n{DEFS}\n'
    )


def bg_layer_markup(bg_png: pathlib.Path) -> str:
    """Background image + the legibility scrims, as one 'background' layer."""
    return (
        f'<image x="0" y="0" width="{W}" height="{H}" '
        f'xlink:href="data:image/png;base64,{b64(bg_png)}"/>\n'
        f'    <rect x="0" y="0" width="{W}" height="760" fill="url(#topScrim)"/>\n'
        f'    <rect x="0" y="{H - 500}" width="{W}" height="500" fill="url(#botScrim)"/>'
    )


def write_master_svg(bg_png: pathlib.Path) -> pathlib.Path:
    parts = [svg_open()]
    parts.append(
        '  <g inkscape:groupmode="layer" inkscape:label="background">\n'
        f'    {bg_layer_markup(bg_png)}\n  </g>\n'
    )
    for label, inner in LAYERS.items():
        parts.append(
            f'  <g inkscape:groupmode="layer" inkscape:label="{label}">\n'
            f'    {inner}\n  </g>\n'
        )
    parts.append("</svg>\n")
    out = IMAGES / f"{STEM}.svg"
    out.write_text("".join(parts))
    return out


def layer_svg(inner: str) -> str:
    return svg_open() + "  " + inner + "\n</svg>\n"


def build_ora(bg_png: pathlib.Path, flat_png: pathlib.Path) -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp())
    data = tmp / "data"
    data.mkdir()

    # background layer = art + scrims (render so it matches the flat cover)
    render(svg_open() + "  " + bg_layer_markup(bg_png) + "\n</svg>\n",
           data / "background.png")
    for label, inner in LAYERS.items():
        render(layer_svg(inner), data / f"{label}.png")

    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f'<image version="0.0.3" w="{W}" h="{H}" xres="96" yres="96">',
        "  <stack>",
    ]
    for label in reversed(ORDER):  # first <layer> = topmost
        lines.append(
            f'    <layer name="{label}" src="data/{label}.png" '
            'x="0" y="0" opacity="1.0" visibility="visible" composite-op="svg:src-over"/>'
        )
    lines += ["  </stack>", "</image>", ""]
    (tmp / "stack.xml").write_text("\n".join(lines))

    shutil.copy(flat_png, tmp / "mergedimage.png")
    (tmp / "Thumbnails").mkdir()
    run("magick", flat_png, "-resize", "256x384", tmp / "Thumbnails" / "thumbnail.png")

    out = IMAGES / f"{STEM}.ora"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "image/openraster", compress_type=zipfile.ZIP_STORED)
        for p in sorted(tmp.rglob("*")):
            if p.is_file() and p.name != "mimetype":
                z.write(p, p.relative_to(tmp), compress_type=zipfile.ZIP_DEFLATED)
    shutil.rmtree(tmp)
    return out


def optimize_small(full_png: pathlib.Path, out_png: pathlib.Path, box: str):
    """Resize within `box` (WxH, aspect preserved) + strip + zopfli-optimize."""
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d) / "small.png"
        run("magick", full_png, "-resize", box, "-strip", tmp)
        if out_png.exists():
            out_png.unlink()
        if shutil.which("zopflipng"):
            run("zopflipng", "-y", tmp, out_png)
        else:
            shutil.copy(tmp, out_png)


def main():
    IMAGES.mkdir(exist_ok=True)
    # drop stale dimensioned PNGs from previous (differently sized) builds
    for stale in IMAGES.glob(f"{STEM}-*x*.png"):
        stale.unlink()
    with tempfile.TemporaryDirectory() as d:
        bg = pathlib.Path(d) / "bg.png"
        compose_art(bg)
        svg = write_master_svg(bg)
        png_full = IMAGES / f"{STEM}-{W}x{H}.png"
        render(svg.read_text(), png_full)
        pdf = IMAGES / f"{STEM}.pdf"
        letter_pdf(svg, pdf)
        ora = build_ora(bg, png_full)
        png_small = IMAGES / f"{STEM}.png"
        optimize_small(png_full, png_small, "548x754")

        # much smaller thumbnail, with its size in the filename
        thumb_tmp = IMAGES / f"{STEM}-thumb.png"
        optimize_small(png_full, thumb_tmp, "200x300")
        dims = subprocess.check_output(
            ["magick", "identify", "-format", "%wx%h", str(thumb_tmp)]
        ).decode().strip()
        png_thumb = IMAGES / f"{STEM}-{dims}.png"
        if png_thumb.exists():
            png_thumb.unlink()
        thumb_tmp.rename(png_thumb)

    for p in (png_small, png_thumb, png_full, pdf, svg, ora):
        print(f"wrote images/{p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
