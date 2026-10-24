"""ASCII webcam: reads your webcam, renders it as ASCII art, and publishes it
as a virtual camera that Zoom / Meet / Teams can select.

Requires the OBS Virtual Camera driver (install OBS Studio once).
"""
import argparse

import cv2
import numpy as np
import pyvirtualcam
from PIL import Image, ImageDraw, ImageFont

# dark -> bright; thin, varied glyphs (no solid blocks like # % @) so it reads as text
RAMP = " .'`,:;-_~i!l|/\\()1{}[]?+<>tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*"
RAIN_CHARS = "0123456789:.=*+-<>|/\\ZzXYT"  # falling glyphs for --rain (Consolas has no katakana)


def build_atlas(chars, font_path, font_size):
    """Pre-render every ramp character to a small bitmap so frames are built
    with numpy indexing instead of per-character draw calls."""
    font = ImageFont.truetype(font_path, font_size)
    left, top, right, bottom = font.getbbox("M")
    cw, ch = right - left, int(font_size * 1.25)  # room for descenders (j, p, q)
    atlas = np.zeros((len(chars), ch, cw), np.uint8)
    for i, c in enumerate(chars):
        img = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(img).text((0, 0), c, fill=255, font=font)
        atlas[i] = np.asarray(img)
    # small fonts never reach full brightness; rescale so the hard-edge cutoff works at any size
    atlas = np.minimum(atlas.astype(np.uint16) * 255 // max(int(atlas.max()), 1), 255).astype(np.uint8)
    return atlas, cw, ch


class FaceCropper:
    """Crops the camera frame around your face so the character grid is spent on
    the face instead of the whole room. Position is smoothed so it doesn't jitter."""

    def __init__(self, aspect, zoom):
        self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.aspect, self.zoom = aspect, zoom
        self.box = None  # smoothed (centre x, centre y, face height)
        self.target = None  # last accepted detection
        self.n = 0

    def crop(self, frame):
        H, W = frame.shape[:2]
        if self.n % 3 == 0:  # detect every few frames to save CPU
            g = cv2.cvtColor(cv2.resize(frame, (W // 2, H // 2)), cv2.COLOR_BGR2GRAY)
            faces = self.det.detectMultiScale(g, 1.2, 5, minSize=(40, 40))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                det = np.array([(x + w / 2) * 2, (y + h / 2) * 2, h * 2], np.float32)
                # deadzone: ignore small detector wobble so the crop holds still
                if self.target is None or np.abs(det - self.target).max() > 0.12 * self.target[2]:
                    self.target = det
        self.n += 1
        if self.target is None:
            return frame
        # glide toward the target a little each frame instead of jumping
        self.box = self.target.copy() if self.box is None else self.box + (self.target - self.box) * 0.08
        cx, cy, fh = self.box
        ch_ = min(fh * self.zoom, H)
        cw_ = ch_ * self.aspect
        if cw_ > W:
            cw_, ch_ = W, W / self.aspect
        x0 = int(np.clip(cx - cw_ / 2, 0, W - cw_))
        y0 = int(np.clip(cy - ch_ / 2, 0, H - ch_))
        return frame[y0:y0 + int(ch_), x0:x0 + int(cw_)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--font-size", type=int, default=12, help="smaller = more detail")
    p.add_argument("--font", default="C:/Windows/Fonts/consolab.ttf", help="bold Consolas by default")
    p.add_argument("--speed", type=float, default=0.6, help="rain speed multiplier (lower = easier to compress)")
    p.add_argument("--color", action="store_true", help="tint characters with the source colour")
    p.add_argument("--matrix", action="store_true", help="green-on-black Matrix style")
    p.add_argument("--smooth", type=float, default=0.35,
                   help="weight of the newest frame, 0-1 (lower = steadier but laggier)")
    p.add_argument("--solo", action="store_true", help="remove the background so only you are drawn")
    p.add_argument("--face", action="store_true", help="crop to your face so it fills the frame")
    p.add_argument("--zoom", type=float, default=1.7, help="--face: crop height as a multiple of face height")
    p.add_argument("--out-size", help="output size WxH, e.g. 1920x1080 (camera capture stays --width/--height)")
    p.add_argument("--fullscreen", action="store_true", help="start the preview window fullscreen")
    p.add_argument("--preview", action="store_true",
                   help="open a window showing the output (share this window in a call for sharper video)")
    p.add_argument("--dump", help="save frame 60 to this PNG path and exit (for debugging)")
    p.add_argument("--rain", action="store_true", help="Matrix digital rain with your image in it")
    p.add_argument("--gain", type=float, default=2.2, help="rain mode brightness multiplier")
    p.add_argument("--invert-amount", type=float, default=0.8,
                   help="rain mode: 0 = you are bright, 1 = you are a dark silhouette")
    p.add_argument("--invert", action="store_true", help="dark characters on light background")
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (is another app using it?)")

    chars = RAIN_CHARS if args.rain else (RAMP[::-1] if args.invert else RAMP)
    atlas, cw, ch = build_atlas(chars, args.font, args.font_size)
    # output size can differ from the camera size, e.g. render 1080p so fullscreen stays sharp
    out_W, out_H = (int(v) for v in args.out_size.lower().split("x")) if args.out_size else (args.width, args.height)
    cols, rows = out_W // cw, out_H // ch
    out_w, out_h = cols * cw, rows * ch
    lut = np.linspace(0, len(chars) - 1, 256).astype(np.int32)

    # Browsers can reject odd camera sizes, so always publish exactly width x height.
    bg = 255 if args.invert else 0
    canvas = np.full((out_H, out_W, 3), bg, np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    # digital-rain state: one falling streak per column
    rng = np.random.default_rng()
    trails = rng.integers(8, 22, cols).astype(np.float32)
    period = rows + trails
    heads = rng.random(cols).astype(np.float32) * period
    speeds = (rng.uniform(0.25, 0.7, cols) * args.speed).astype(np.float32)
    glyph_idx = rng.integers(0, len(chars), (rows, cols))
    row_ids = np.arange(rows, dtype=np.float32)[:, None]
    frames = 0
    s = args.smooth  # weight of the newest frame in temporal smoothing
    prev_gray = prev_person = stretch = None
    cropper = FaceCropper(out_w / out_h, args.zoom) if args.face else None
    segmenter = None
    if args.solo:
        import mediapipe as mp  # imported lazily: heavy, and only needed for --solo
        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    with pyvirtualcam.Camera(out_W, out_H, args.fps, fmt=pyvirtualcam.PixelFormat.RGB) as cam:
        print(f"Virtual camera: {cam.device}  ({out_W}x{out_H}, {cols}x{rows} chars)")
        fullscreen = args.fullscreen
        if args.preview:
            cv2.namedWindow("ASCII Cam", cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty("ASCII Cam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Preview: press f to toggle fullscreen, q or Esc to quit")
        print("Select it in Zoom/Meet. Ctrl+C to stop.")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            if cropper:
                frame = cropper.crop(frame)
            person = None
            if segmenter:
                seg = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
                seg = cv2.resize(seg, (cols, rows), interpolation=cv2.INTER_AREA)
                person = np.clip((seg - 0.3) / 0.4, 0, 1)  # soft edge between you and the background
                prev_person = person if prev_person is None else prev_person * (1 - s) + person * s
                person = prev_person
            small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # boost contrast so faces don't wash out into a few characters
            gray = clahe.apply(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))
            # temporal smoothing so cells near a character threshold don't flicker
            gray_f = gray.astype(np.float32)
            prev_gray = gray_f if prev_gray is None else prev_gray * (1 - s) + gray_f * s
            gray = prev_gray.astype(np.uint8)
            if person is not None and not args.rain:
                gray = (gray * person).astype(np.uint8)  # background cells become blank
            if args.rain:
                a = args.invert_amount
                w = gray.astype(np.float32) / 255
                if person is not None and (person > 0.5).sum() > 50:
                    # spread your own tones across the full range (ignores the removed background)
                    lo_hi = np.percentile(w[person > 0.5], (5, 95)).astype(np.float32)
                    stretch = lo_hi if stretch is None else stretch * 0.9 + lo_hi * 0.1
                    w = np.clip((w - stretch[0]) / max(stretch[1] - stretch[0], 0.1), 0, 1)
                w = (1 - a) * w + a * (1 - w)
                if person is not None:
                    w = np.maximum(w, 0.12 * person) * person  # small floor so dark hair/shadow still shows
                heads = (heads + speeds) % period
                d = (heads[None, :] - row_ids) % period[None, :]  # distance behind the streak head
                streak = np.where(d < trails[None, :], 1 - d / trails[None, :], 0)
                # image shows through as streak brightness plus a faint glow of its own
                # your image decides where it's bright; the streaks only animate it
                val = w * (0.55 + 0.45 * streak)
                val = np.clip(val ** 0.65 * args.gain, 0, 1)
                flip = rng.random((rows, cols)) < 0.015
                glyph_idx[flip] = rng.integers(0, len(chars), int(flip.sum()))
                glyphs = atlas[glyph_idx]
            else:
                glyphs = atlas[lut[gray]]  # (rows, cols, ch, cw)
            mask = glyphs.transpose(0, 2, 1, 3).reshape(out_h, out_w)

            # thin glyphs like '.' light few pixels; make them stronger
            # hard edges: soft anti-aliased strokes get smeared by video compression
            mask = np.where(mask > 50, 255, 0).astype(np.uint8)

            if args.rain:
                glow = np.clip((val - 0.7) / 0.3, 0, 1)  # streak heads go white-green
                cell = np.stack([glow * 200, 255 * val, glow * 200], axis=2).astype(np.uint8)
            elif args.matrix:
                lum = (gray.astype(np.float32) / 255) ** 0.8
                glow = np.clip((lum - 0.8) / 0.2, 0, 1)  # brightest cells go white-green
                cell = np.stack([glow * 200, 12 + 243 * lum, glow * 200], axis=2).astype(np.uint8)
            elif args.color:
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                cell = (255 * (rgb.astype(np.float32) / 255) ** 0.6).astype(np.uint8)
            else:
                cell = None

            if cell is not None:
                cell = np.repeat(np.repeat(cell, ch, axis=0), cw, axis=1)
                img = (cell.astype(np.uint16) * mask[..., None] // 255).astype(np.uint8)
            else:
                img = np.repeat(mask[..., None], 3, axis=2)
            if args.invert:
                img = 255 - img

            canvas[:out_h, :out_w] = img
            cam.send(canvas)
            if args.preview:
                cv2.imshow("ASCII Cam", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc closes it
                    return
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty("ASCII Cam", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            frames += 1
            if args.dump and frames == 60:
                cv2.imwrite(args.dump, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                print(f"Saved frame to {args.dump}")
                return
            cam.sleep_until_next_frame()


if __name__ == "__main__":
    main()
