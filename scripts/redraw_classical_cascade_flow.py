from PIL import Image, ImageDraw, ImageFont


W, H = 2048, 768
BG = (242, 242, 245)
STROKE = (55, 62, 98)
TEXT = (35, 40, 65)
ARROW = (20, 20, 20)


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=TEXT, spacing=3):
    x0, y0, x1, y1 = box
    lft, top, rgt, bot = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=spacing)
    tw, th = rgt - lft, bot - top
    px = x0 + (x1 - x0 - tw) / 2
    py = y0 + (y1 - y0 - th) / 2
    draw.multiline_text((px, py), text, font=font, fill=fill, align="center", spacing=spacing)


def rr(draw: ImageDraw.ImageDraw, box, fill, radius=24, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=STROKE, width=width)


def rect(draw: ImageDraw.ImageDraw, box, fill, width=3):
    draw.rectangle(box, fill=fill, outline=STROKE, width=width)


def diamond(draw: ImageDraw.ImageDraw, box, fill, width=3):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    draw.polygon(pts, fill=fill, outline=STROKE, width=width)


def arrow(draw: ImageDraw.ImageDraw, p0, p1, width=4, ah=14, aw=9, color=ARROW):
    draw.line([p0, p1], fill=color, width=width)
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / nrm, dy / nrm
    bx, by = x1 - ah * ux, y1 - ah * uy
    px, py = -uy, ux
    p2 = (bx + aw * px, by + aw * py)
    p3 = (bx - aw * px, by - aw * py)
    draw.polygon([p2, (x1, y1), p3], fill=color)


def poly_arrow(draw: ImageDraw.ImageDraw, pts, width=4, color=ARROW):
    for i in range(len(pts) - 2):
        draw.line([pts[i], pts[i + 1]], fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], width=width, color=color)


def dashed_rect(draw: ImageDraw.ImageDraw, box, dash=10, gap=8, color=(255, 45, 45), width=3):
    x0, y0, x1, y1 = box
    x = x0
    while x < x1:
        xe = min(x + dash, x1)
        draw.line([(x, y0), (xe, y0)], fill=color, width=width)
        draw.line([(x, y1), (xe, y1)], fill=color, width=width)
        x += dash + gap
    y = y0
    while y < y1:
        ye = min(y + dash, y1)
        draw.line([(x0, y), (x0, ye)], fill=color, width=width)
        draw.line([(x1, y), (x1, ye)], fill=color, width=width)
        y += dash + gap


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    title_font = load_font(16)
    node_font = load_font(16)
    small_font = load_font(14)
    tiny_font = load_font(12)

    # Main and bottom frames
    rect(d, (300, 10, 1930, 540), fill=BG, width=4)
    rect(d, (940, 550, 1820, 760), fill=BG, width=4)

    # Nodes
    in_box = (8, 235, 205, 340)
    pyr_box = (238, 235, 405, 335)
    scan_box = (500, 235, 667, 335)
    d1_box = (720, 170, 950, 395)
    d2_box = (1040, 170, 1260, 395)
    dn_box = (1370, 255, 1600, 475)
    out_box = (1765, 255, 2040, 355)

    rr(d, in_box, fill=(209, 142, 220))
    rect(d, pyr_box, fill=(246, 246, 247))
    rect(d, scan_box, fill=(246, 246, 247))
    diamond(d, d1_box, fill=(246, 246, 247))
    diamond(d, d2_box, fill=(246, 246, 247))
    diamond(d, dn_box, fill=(246, 246, 247))
    rr(d, out_box, fill=(174, 175, 235))

    # Title
    draw_centered_text(
        d,
        (980, 16, 1570, 100),
        "Quy trình Phân loại Phân\ntầng Cascade Classifier",
        title_font,
    )

    # Node labels
    draw_centered_text(d, in_box, "Ảnh đầu vào", small_font)
    draw_centered_text(d, pyr_box, "Tạo Tháp Hình\nẢnh", small_font)
    draw_centered_text(d, scan_box, "Quét Cửa Sổ\nTrượt", small_font)
    draw_centered_text(d, d1_box, "Trạm 1: Bộ phận\nloại\nRất đơn giản", small_font)
    draw_centered_text(d, d2_box, "Trạm 2: Bộ phận\nloại\nPhức tạp hơn", small_font)
    draw_centered_text(d, dn_box, "... Trạm n: Bộ phận\nloại\nPhức tạp nhất ...", small_font)
    draw_centered_text(d, out_box, "Đầu ra: Tọa độ khuôn mặt", small_font)

    # Right helper boxes
    merge_box = (1630, 95, 1830, 190)
    rej_top_box = (1495, 185, 1590, 275)
    rej_bot_box = (1490, 430, 1585, 515)
    rect(d, merge_box, fill=(246, 246, 247))
    rect(d, rej_top_box, fill=(246, 246, 247))
    rect(d, rej_bot_box, fill=(246, 246, 247))
    draw_centered_text(d, merge_box, "Gom cụm Bounding\nBox", small_font)
    draw_centered_text(d, rej_top_box, "Loại", small_font)
    draw_centered_text(d, rej_bot_box, "Loại", small_font)

    # State labels and notes
    draw_centered_text(d, (1020, 210, 1100, 250), "Non-face", tiny_font)
    draw_centered_text(d, (1010, 258, 1068, 298), "Face", tiny_font)
    draw_centered_text(d, (1320, 210, 1400, 250), "Non-face", tiny_font)
    draw_centered_text(d, (1320, 303, 1380, 343), "Face", tiny_font)
    draw_centered_text(d, (1462, 215, 1592, 255), "Vượt qua tất", tiny_font)
    draw_centered_text(d, (1145, 395, 1255, 435), "Chi tiết xử", tiny_font)

    dashed = (1060, 94, 1235, 148)
    dashed_rect(d, dashed)
    draw_centered_text(d, dashed, "Loại bỏ nền siêu", tiny_font)

    # Bottom module
    b1 = (970, 590, 1190, 720)
    b2 = (1250, 603, 1480, 707)
    b3 = (1540, 603, 1740, 707)
    rect(d, b1, fill=(246, 246, 247))
    rect(d, b2, fill=(246, 246, 247))
    rect(d, b3, fill=(246, 246, 247))
    draw_centered_text(d, (1210, 550, 1640, 600), "Bên trong 1 Trạm Bộ phận\nloại Mạnh", small_font)
    draw_centered_text(d, b1, "1. Trích xuất Đặc trưng\nHaar-like / MB-LBP +\nIntegral Image", tiny_font)
    draw_centered_text(d, b2, "2. Các Bộ phận loại\nYếu", small_font)
    draw_centered_text(d, b3, "3. Kết hợp AdaBoost\nThành Strong", small_font)

    # Main horizontal flow
    arrow(d, (205, 286), (238, 286), width=4)
    arrow(d, (405, 286), (500, 286), width=4)
    arrow(d, (667, 286), (720, 286), width=4)
    arrow(d, (950, 286), (1040, 286), width=4)

    # d2 -> trạm n
    arrow(d, (1260, 286), (1370, 341), width=4)

    # d1/d2 side logic to top reject and dashed box
    poly_arrow(d, [(950, 232), (1040, 232)], width=4)
    poly_arrow(d, [(1100, 232), (1100, 150)], width=4)
    poly_arrow(d, [(1260, 232), (1320, 232), (1495, 232)], width=4)
    poly_arrow(d, [(1495, 232), (1495, 175)], width=4)

    # d2 face to trạm n and trạm n to outputs
    poly_arrow(d, [(1260, 318), (1320, 318), (1370, 340)], width=4)
    poly_arrow(d, [(1600, 340), (1765, 305)], width=4)
    poly_arrow(d, [(1585, 232), (1630, 140)], width=4)
    poly_arrow(d, [(1830, 140), (1905, 255)], width=4)
    poly_arrow(d, [(1570, 430), (1585, 472)], width=4)

    # pass-all -> merge
    poly_arrow(d, [(1592, 235), (1698, 235), (1698, 190)], width=4)

    # bottom internal and pointers
    poly_arrow(d, [(1190, 655), (1250, 655)], width=4)
    poly_arrow(d, [(1480, 655), (1540, 655)], width=4)
    d.line([(1245, 414), (1245, 582)], fill=ARROW, width=3)
    arrow(d, (1245, 582), (1245, 590), width=3)
    d.line([(1260, 414), (970, 414)], fill=ARROW, width=3)

    out_path = "figures/custom/classical_cascade_detailed_flow.png"
    img.save(out_path, quality=95)
    print(out_path)


if __name__ == "__main__":
    main()
