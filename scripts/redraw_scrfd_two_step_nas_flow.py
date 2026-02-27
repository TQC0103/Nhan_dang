from PIL import Image, ImageDraw, ImageFont


W, H = 2450, 520
BG = (244, 244, 247)
INK = (42, 42, 72)
ARROW = (20, 20, 20)


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill=INK, spacing=4):
    x0, y0, x1, y1 = box
    lft, top, rgt, bot = draw.multiline_textbbox(
        (0, 0), text, font=fnt, align="center", spacing=spacing
    )
    tw, th = rgt - lft, bot - top
    draw.multiline_text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
        text,
        font=fnt,
        fill=fill,
        align="center",
        spacing=spacing,
    )


def arrow(draw: ImageDraw.ImageDraw, p0, p1, width=5, head_len=18, head_w=12):
    draw.line([p0, p1], fill=ARROW, width=width)
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / nrm, dy / nrm
    bx, by = x1 - head_len * ux, y1 - head_len * uy
    px, py = -uy, ux
    p2 = (bx + head_w * px, by + head_w * py)
    p3 = (bx - head_w * px, by - head_w * py)
    draw.polygon([p2, (x1, y1), p3], fill=ARROW)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_main = load_font(60)
    f_mid = load_font(54)
    f_small = load_font(36)

    # Start/end are true circles (equal width and height) for better readability.
    start = (20, 80, 380, 440)
    b1 = (390, 150, 910, 370)
    bmid = (940, 150, 1460, 370)
    b2 = (1490, 150, 2030, 370)
    out = (2050, 80, 2410, 440)

    d.ellipse(start, outline=INK, width=6, fill=BG)
    d.rounded_rectangle(b1, radius=28, outline=(36, 160, 72), width=6, fill=(194, 231, 206))
    d.rectangle(bmid, outline=INK, width=6, fill=(245, 245, 248))
    d.rounded_rectangle(b2, radius=28, outline=(28, 120, 230), width=6, fill=(172, 199, 227))
    d.ellipse(out, outline=INK, width=6, fill=BG)

    centered_text(d, start, "Start NAS", f_main, spacing=6)
    centered_text(d, b1, "Bước 1: Giới hạn FLOPs\nBootstrap quét Backbone", f_small)
    centered_text(d, bmid, "Chốt tỉ lệ Depth/Width\nƯu tiên tầng cạn C2, C3", f_small)
    centered_text(d, b2, "Bước 2: Quét số lượng\nkênh", f_small)
    centered_text(d, out, "Mô hình\nSCRFD\nTối ưu", f_mid, spacing=4)

    y = 260
    arrow(d, (380, y), (390, y), width=6, head_len=22, head_w=14)
    arrow(d, (910, y), (940, y), width=6, head_len=22, head_w=14)
    arrow(d, (1460, y), (1490, y), width=6, head_len=22, head_w=14)
    arrow(d, (2030, y), (2050, y), width=6, head_len=22, head_w=14)

    out_path = "figures/custom/scrfd_two_step_nas_flow.png"
    img.save(out_path, quality=95)
    print(out_path)


if __name__ == "__main__":
    main()
