"""Resize AI-generated isometric sprites to fit normGym tile cells.

Usage:
    python runner/sprites/resize_iso_sprite.py <input_image> <output_path>

The script automatically determines the target dimensions based on the output
path (e.g., floor/ -> 80x40, wall -> 80x80, counter -> 80x65, etc.).

You can also specify dimensions explicitly:
    python runner/sprites/resize_iso_sprite.py <input_image> <output_path> --size 80x65

What it does:
1. Crops transparent padding from the input image
2. Scales uniformly so the content fills 80px width (no distortion, no clipping)
3. Bottom-aligns on the target canvas (so the isometric base sits at the bottom)
4. Saves as RGBA PNG with transparent background
"""

import argparse
import os
import sys
from PIL import Image


# Target dimensions by sprite category
# Maps path keywords to (width, height)
SIZE_MAP = {
    # Flat sprites (floor-level)
    'floor/': (80, 40),
    'empty': (80, 40),
    'entrance': (80, 40),
    'queue_zone': (80, 40),
    'rope_barrier': (80, 40),
    'rope_stanchion': (80, 40),
    'cart_corral': (80, 40),
    'snack_stand': (80, 40),
    'atm_room': (80, 40),

    # Medium-height sprites
    'table_2top': (80, 55),
    'table_4top': (80, 55),
    'theater_seat': (80, 55),
    'product_shelf': (80, 55),
    'medicine_shelf': (80, 55),
    'display_case': (80, 55),
    'bp_station': (80, 55),

    # Tall-medium sprites
    'counter': (80, 65),
    'pickup_shelf': (80, 65),
    'laptop_bar': (80, 65),
    'chair': (80, 65),
    'concession': (80, 65),
    'writing_desk': (80, 65),
    'deposit_slips': (80, 65),
    'waiting_chair': (80, 65),
    'bagging_area': (80, 65),
    'rx_pickup': (80, 65),

    # Full-block sprites
    'wall': (80, 80),
    'kitchen': (80, 80),
    'screen_room': (80, 80),
    'vault': (80, 80),
    'stockroom': (80, 80),
    'storage_room': (80, 80),
}

# Agent sprites
AGENT_SIZE = (40, 40)
LLM_STAR_SIZE = (20, 20)


def detect_target_size(output_path):
    """Detect target dimensions from the output path."""
    path_lower = output_path.lower()

    if 'agents/llm_star' in path_lower:
        return LLM_STAR_SIZE
    if 'agents/' in path_lower:
        return AGENT_SIZE

    for keyword, size in SIZE_MAP.items():
        if keyword in path_lower:
            return size

    # Default to 80x40 (flat tile)
    print(f"Warning: Could not detect size from path '{output_path}', defaulting to 80x40")
    return (80, 40)


def resize_sprite(input_path, output_path, target_w, target_h):
    """Resize a sprite to fill the target dimensions properly.

    For flat sprites (target height <= 40), resizes to exact target dimensions
    since isometric diamonds all share the standard 2:1 ratio and any slight
    stretch is imperceptible.

    For 3D sprites (target height > 40), scales to fill width first, then
    fits height if needed, preserving aspect ratio so extruded objects don't
    get distorted. Bottom-aligns on the canvas so the base diamond sits correctly.
    """
    img = Image.open(input_path).convert('RGBA')

    # Step 1: Crop transparent padding
    bbox = img.getbbox()
    if bbox is None:
        print(f"Warning: {input_path} is fully transparent")
        return
    cropped = img.crop(bbox)
    cw, ch = cropped.size

    is_flat = target_h <= 40

    if is_flat:
        # Flat tiles: resize to exact target (diamond fills the cell)
        scaled = cropped.resize((target_w, target_h), Image.LANCZOS)
        result = scaled
    else:
        # 3D sprites: scale to fill width, preserve aspect ratio
        scale = target_w / cw
        new_w = target_w
        new_h = int(ch * scale)

        # If scaled height exceeds target, scale to fit height instead
        if new_h > target_h:
            scale = target_h / ch
            new_w = int(cw * scale)
            new_h = target_h

        scaled = cropped.resize((new_w, new_h), Image.LANCZOS)

        # Create target canvas and bottom-align + center horizontally
        result = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = target_h - new_h  # bottom-align
        result.paste(scaled, (paste_x, paste_y), scaled)

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result.save(output_path, 'PNG')

    print(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)} "
          f"({cw}x{ch} -> {target_w}x{target_h} canvas)")


def main():
    parser = argparse.ArgumentParser(description='Resize isometric sprites for normGym')
    parser.add_argument('input', help='Input image path (or directory for batch mode)')
    parser.add_argument('output', help='Output image path (or directory for batch mode)')
    parser.add_argument('--size', help='Target size as WxH (e.g., 80x65). Overrides auto-detection.')
    args = parser.parse_args()

    # Parse explicit size if provided
    explicit_size = None
    if args.size:
        w, h = args.size.split('x')
        explicit_size = (int(w), int(h))

    # Batch mode: input is a directory
    if os.path.isdir(args.input):
        files = [f for f in os.listdir(args.input) if f.lower().endswith('.png')]
        if not files:
            print("No PNG files found in input directory")
            return
        os.makedirs(args.output, exist_ok=True)
        print(f"Batch resizing {len(files)} sprites:")
        for fname in sorted(files):
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.output, fname)
            target_w, target_h = explicit_size or detect_target_size(out_path)
            resize_sprite(in_path, out_path, target_w, target_h)
    else:
        # Single file mode
        target_w, target_h = explicit_size or detect_target_size(args.output)
        print(f"Resizing to {target_w}x{target_h}:")
        resize_sprite(args.input, args.output, target_w, target_h)


if __name__ == '__main__':
    main()
