import json
import re

# Đọc các cell từ Train notebook
with open('spos-supernet-for-scrf-dynamic-block.ipynb', 'r', encoding='utf-8') as f:
    train_nb = json.load(f)

# Đọc các cell từ Search notebook
with open('scrfd-spos-nas-and-inference-dynamic-block (2).ipynb', 'r', encoding='utf-8') as f:
    search_nb = json.load(f)

new_cells = []

# Danh sách các mã cell đã duyệt để tránh bị trùng (đặc biệt là import và class khai báo lại)
seen_code = set()

# Process train cells
for cell in train_nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        # Sửa prior_prob
        source = source.replace("prior_prob = 0.01", "prior_prob = 0.05")
        
        # Sửa _create_heatmap
        if "def _create_heatmap" in source:
            # Dùng Regex hoặc replace khúc code đó
            old_func = """    def _create_heatmap(self, target_size, bboxes, stride, orig_w, orig_h):
        \"\"\"Hàm vẽ điểm tâm của khuôn mặt lên bản đồ nhiệt\"\"\"
        # Khởi tạo heatmap toàn số 0
        heatmap = torch.zeros((1, target_size[0], target_size[1]), dtype=torch.float32)

        # Tỷ lệ thu phóng ảnh gốc so với ảnh 640x640
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h

        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            # Bỏ qua các khuôn mặt quá nhỏ hoặc lỗi
            if w < 2 or h < 2: continue

            # Tính tọa độ tâm trên ảnh 640x640
            center_x = (x_min + w / 2) * scale_x
            center_y = (y_min + h / 2) * scale_y

            # Chiếu tọa độ tâm xuống kích thước lưới của Stride hiện tại
            grid_x = int(center_x / stride)
            grid_y = int(center_y / stride)

            # Đảm bảo tọa độ không bị vượt quá giới hạn (out of bounds)
            if 0 <= grid_x < target_size[1] and 0 <= grid_y < target_size[0]:
                # Gán nhãn 1 (Có mặt người)
                heatmap[0, grid_y, grid_x] = 1.0

                # Mẹo nhỏ: Đánh dấu các ô kế cận (Blur nhẹ) để mạng dễ hội tụ hơn
                # (Tạo một vùng 3x3 quanh tâm với giá trị 0.5)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if (dx != 0 or dy != 0) and 0 <= nx < target_size[1] and 0 <= ny < target_size[0]:
                            # Chỉ ghi đè nếu ô đó đang là 0 (để không xóa mất tâm 1.0)
                            if heatmap[0, ny, nx] == 0:
                                heatmap[0, ny, nx] = 0.5

        return heatmap"""
            
            new_func = """    def _create_heatmap(self, target_size, bboxes, stride, orig_w, orig_h):
        \"\"\"Hiệu chỉnh: Vẽ Viền (Edge) của khuôn mặt thay vì Tâm\"\"\"
        heatmap = torch.zeros((1, target_size[0], target_size[1]), dtype=torch.float32)
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h

        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            if w < 2 or h < 2: continue

            x1 = x_min * scale_x
            y1 = y_min * scale_y
            x2 = (x_min + w) * scale_x
            y2 = (y_min + h) * scale_y

            gx1, gy1 = int(x1 / stride), int(y1 / stride)
            gx2, gy2 = int(x2 / stride), int(y2 / stride)
            
            gx1 = max(0, min(gx1, target_size[1]-1))
            gy1 = max(0, min(gy1, target_size[0]-1))
            gx2 = max(0, min(gx2, target_size[1]-1))
            gy2 = max(0, min(gy2, target_size[0]-1))
            
            # Ưu tiên Viền (1.0)
            heatmap[0, gy1, gx1:(gx2+1)] = 1.0
            heatmap[0, gy2, gx1:(gx2+1)] = 1.0
            heatmap[0, gy1:(gy2+1), gx1] = 1.0
            heatmap[0, gy1:(gy2+1), gx2] = 1.0
            
            # Tâm là Focus cấp 2 (0.5)
            cx = (gx1 + gx2) // 2
            cy = (gy1 + gy2) // 2
            # Ghi đè tâm là 0.5 (Chỉ khi width/height đủ để tạo khoảng rỗng bên trong)
            if gx2 - gx1 > 1 and gy2 - gy1 > 1:
                heatmap[0, cy, cx] = 0.5
                
        return heatmap"""
            source = source.replace(old_func, new_func)

        # Cắt bớt phần save model cuống của Train nếu có để Search cell lồng vào
        cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.split('\n')][:-1] if '\n' not in source[-1:] else [line + '\n' for line in source.split('\n')[:-1]]
        # Better:
        cell['source'] = [line + '\n' for line in source.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n') # Xóa newline thừa ở dòng cuối
        
        seen_code.add(source.strip())
        
    new_cells.append(cell)


# Thêm Markdown header chuyển tiếp sang Search
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## PHASE 2: THỰC THI KIẾM TRÚC NAS\n",
        "Tự động rà soát Không gian kiến trúc (Architecture Search Space) ngay sau khi huấn luyện xong."
    ]
})

# Thêm các logic Search của NAS notebook (Bỏ qua các class bị trùng)
for cell in search_nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        # Nếu là import bị trùng hoặc khai báo class bị trùng -> bỏ qua
        if "SEARCH_SPACE =" in source and "class DynamicConv2d" in source:
            continue
        if "class DynamicInvertedResidual" in source:
            continue
        if "class SPOS_Backbone" in source:
            continue
        if "class SharedPAFPN" in source:
            continue
        if "class WiderFaceHeatmapDataset" in source:
            continue
        if "class ProxyFocalLossWithLogits" in source:
            continue
        if "parse_wider_face_with_sr" in source:
            continue
        if "KHÔI PHỤC MẠNG SUPERNET" in source:
            # Thay vì khôi phục từ file, chúng ta đã có Supernet đang ở RAM do cell train, bỏ qua
            continue
            
        cell['source'] = [line + '\n' for line in source.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
        new_cells.append(cell)
    else:
        new_cells.append(cell)


final_nb = {
    "cells": new_cells,
    "metadata": train_nb.get('metadata', {}),
    "nbformat": train_nb.get('nbformat', 4),
    "nbformat_minor": train_nb.get('nbformat_minor', 5)
}

with open('scrfd_nas_edge_heatmap.ipynb', 'w', encoding='utf-8') as f:
    json.dump(final_nb, f, indent=2, ensure_ascii=False)

print("Tạo Notebook kết hợp thành công!")
