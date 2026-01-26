import argparse
from pathlib import Path
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Create a small GIF from PNGs.")
    p.add_argument("--input", help="Path to a single predicted_vs_actual.png")
    p.add_argument("--batch-dir", help="Directory containing results/iter_XX_plots")
    p.add_argument("--pattern", default="iter_*_plots/predicted_vs_actual.png")
    p.add_argument("--output", default="predicted_vs_actual.gif")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--duration", type=int, default=800, help="Frame duration in ms")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)

    frames = []
    if args.batch_dir:
        base = Path(args.batch_dir)
        images = sorted(base.glob(args.pattern))
        if not images:
            raise SystemExit("No images found for batch mode.")
        for path in images:
            img = Image.open(path).convert("RGBA")
            img = img.resize((args.width, args.height), resample=Image.LANCZOS)
            frames.append(img)
    else:
        if not args.input:
            raise SystemExit("Provide --input or --batch-dir.")
        img = Image.open(Path(args.input)).convert("RGBA")
        img = img.resize((args.width, args.height), resample=Image.LANCZOS)
        frames = [img, img.copy()]

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
    )

    print(f"Saved GIF: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
