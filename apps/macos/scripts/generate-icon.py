#!/usr/bin/env python3
"""
Generate the LatticeStudio.icns app icon.

Design: dark rounded-rect (#0A0B0D), teal (#00E5C7) lattice grid glyph.
Renders a 5x5 node grid connected by edges, with corner nodes highlighted.

Usage:
    uv run scripts/generate-icon.py [--out <path>]
    # Outputs: Resources/LatticeStudio.icns
"""
import argparse, os, subprocess, sys, tempfile, shutil, math
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"], stdout=subprocess.DEVNULL)
    from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]  # lattice/
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR.parent / "Resources" / "LatticeStudio.icns"

BG = (10, 11, 13)          # #0A0B0D
TEAL = (0, 229, 199)       # #00E5C7
TEAL_DIM = (0, 140, 122)   # dimmed edges


def render_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded-rect background — corner radius ~22% of size
    r = int(size * 0.22)
    draw.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=r, fill=BG + (255,))

    # Grid parameters: 4x4 grid of lines (5 nodes per axis) occupying ~55% of canvas
    pad = size * 0.20
    grid_size = size - 2 * pad
    cols = rows = 4  # 5 nodes = 4 intervals

    nodes = []
    for gy in range(rows + 1):
        for gx in range(cols + 1):
            x = pad + gx * (grid_size / cols)
            y = pad + gy * (grid_size / rows)
            nodes.append((x, y))

    # Draw edges first (behind nodes)
    edge_w = max(1, int(size * 0.018))
    for gy in range(rows + 1):
        for gx in range(cols + 1):
            idx = gy * (cols + 1) + gx
            nx, ny = nodes[idx]
            # Horizontal edge
            if gx < cols:
                rx, ry = nodes[idx + 1]
                draw.line([(nx, ny), (rx, ry)], fill=TEAL_DIM + (200,), width=edge_w)
            # Vertical edge
            if gy < rows:
                bx, by = nodes[idx + (cols + 1)]
                draw.line([(nx, ny), (bx, by)], fill=TEAL_DIM + (200,), width=edge_w)

    # Draw nodes — larger at corners, medium at edges, small inside
    for gy in range(rows + 1):
        for gx in range(cols + 1):
            is_corner = (gx in (0, cols)) and (gy in (0, rows))
            is_edge   = (gx in (0, cols)) or (gy in (0, rows))
            if is_corner:
                base_r = size * 0.048
                alpha = 255
            elif is_edge:
                base_r = size * 0.030
                alpha = 220
            else:
                base_r = size * 0.020
                alpha = 180
            nr = int(base_r)
            x, y = nodes[gy * (cols + 1) + gx]
            draw.ellipse(
                [(x - nr, y - nr), (x + nr, y + nr)],
                fill=TEAL + (alpha,)
            )

    return img


ICON_SIZES = [16, 32, 128, 256, 512]


def build_iconset(iconset_dir: Path):
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for sz in ICON_SIZES:
        img = render_icon(sz)
        img.save(iconset_dir / f"icon_{sz}x{sz}.png")
        img2 = render_icon(sz * 2).resize((sz * 2, sz * 2), Image.LANCZOS)
        img2.save(iconset_dir / f"icon_{sz}x{sz}@2x.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "LatticeStudio.iconset"
        build_iconset(iconset)
        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"iconutil failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)

    print(f"Icon written: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
