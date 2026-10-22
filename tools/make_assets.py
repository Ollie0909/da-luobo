"""把 ./photos 的照片去背，輸出頭部到 assets/heads、全身到 assets/bodies。

座標是手動標的（照片轉正後的像素座標 / scale），之後換照片要重標。
"""
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter, ImageOps
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "photos"
HEADS = ROOT / "assets" / "heads"
BODIES = ROOT / "assets" / "bodies"

# name: (photo, scale, (left, hair_top, right, chin)[, 下半橢圓寬度倍率])
HEAD_SPECS = {
    "normal-1": ("7.jpg", 1.0, (205, 735, 600, 1225)),
    "normal-2": ("4.jpg", 1.0, (262, 95, 592, 532), 0.8),
    "normal-3": ("3.jpg", 1.0, (420, 702, 592, 928)),
    "normal-4": ("1.jpg", 2.016, (640, 280, 860, 548)),
    "hit-1": ("2.jpg", 2.016, (632, 80, 1212, 765), 0.9),
    "hit-2": ("9.jpg", 1.0, (378, 648, 672, 938)),
}

# name: (photo, scale, (left, top, right, bottom))
BODY_SPECS = {
    "barefoot": ("1.jpg", 2.016, (250, 270, 1400, 1965)),
    "squat": ("5.jpg", 1.0, (340, 520, 745, 1095)),
    "bridge": ("10.jpg", 1.0, (405, 740, 595, 1295)),
}

HEAD_HEIGHT = 360
BODY_HEIGHT = 520


def load(name):
    return ImageOps.exif_transpose(Image.open(PHOTOS / name)).convert("RGB")


def scaled(box, s):
    return tuple(round(v * s) for v in box)


def head_mask(size, box, lower_rx=1.0):
    """上半部保留整塊（頭髮），下半部用橢圓收在下巴，去掉脖子和肩膀。"""
    w, h = size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pad = (x1 - x0) * 0.06
    rx, ry = ((x1 - x0) / 2 + pad) * lower_rx, (y1 - cy) + pad * 0.6
    yy, xx = np.mgrid[0:h, 0:w]
    upper = (yy < cy) & (xx > x0 - pad) & (xx < x1 + pad) & (yy > y0 - pad * 2)
    lower = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    m = Image.fromarray(((upper | lower) * 255).astype(np.uint8))
    return m.filter(ImageFilter.GaussianBlur(max(2, (x1 - x0) / 120)))


def keep_largest(alpha):
    """只留最大一塊，清掉背景殘留的小碎片。"""
    labels, n = ndimage.label(alpha > 40)
    if n <= 1:
        return alpha
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
    return np.where(labels == 1 + int(np.argmax(sizes)), alpha, 0).astype(np.uint8)


def trim_and_resize(img, height):
    img = img.crop(img.getchannel("A").point(lambda a: 255 if a > 12 else 0).getbbox())
    if img.height > height:
        img = img.resize((round(img.width * height / img.height), height), Image.LANCZOS)
    return img


def make_heads(session):
    HEADS.mkdir(parents=True, exist_ok=True)
    for name, (photo, s, box, *rest) in HEAD_SPECS.items():
        img = load(photo)
        x0, y0, x1, y1 = scaled(box, s)
        mx, my = (x1 - x0) * 0.35, (y1 - y0) * 0.35
        crop_box = (max(0, round(x0 - mx)), max(0, round(y0 - my)),
                    min(img.width, round(x1 + mx)), min(img.height, round(y1 + my)))
        crop = img.crop(crop_box)
        cut = remove(crop, session=session)
        local = (x0 - crop_box[0], y0 - crop_box[1], x1 - crop_box[0], y1 - crop_box[1])
        alpha = np.minimum(np.array(cut.getchannel("A")), np.array(head_mask(cut.size, local, *rest)))
        cut.putalpha(Image.fromarray(keep_largest(alpha)))
        out = trim_and_resize(cut, HEAD_HEIGHT)
        out.save(HEADS / f"{name}.png", optimize=True)
        print("head", name, out.size)


def make_bodies(session):
    BODIES.mkdir(parents=True, exist_ok=True)
    for name, (photo, s, box) in BODY_SPECS.items():
        img = load(photo).crop(scaled(box, s))
        cut = remove(img, session=session, post_process_mask=True)
        out = trim_and_resize(cut, BODY_HEIGHT)
        out.save(BODIES / f"{name}.png", optimize=True)
        print("body", name, out.size)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["heads", "bodies"]
    if "heads" in which:
        make_heads(new_session("isnet-general-use"))
    if "bodies" in which:
        make_bodies(new_session("isnet-general-use"))
