# LaTeX Midterm Report Workspace

Repository này đã được sắp xếp lại theo hướng tách rõ:
- nguồn LaTeX đang biên soạn
- dữ liệu thực nghiệm
- tài liệu tham khảo gốc
- mã script hỗ trợ
- phần template cũ để lưu trữ

## Cấu trúc chính

- `main.tex`, `cover.tex`, `preface.tex`, `info.tex`, `Glossary.tex`, `Week1.tex`, `Week2.tex`: bộ file LaTeX đang dùng để build báo cáo.
- `references.bib`: dữ liệu trích dẫn dùng với `biblatex + biber`.
- `assets/`: ảnh dùng trực tiếp trong tài liệu.
- `figures/`: ảnh kết quả phục vụ nội dung báo cáo.
  - `figures/ch1_classical/`: ảnh minh họa Viola-Jones và metadata chọn ảnh đại diện.
- `data/`: dữ liệu chạy thử nghiệm.
  - `data/widerface/`: bộ WIDER FACE val, annotation split, và file thống kê quét.
- `scripts/`: script hỗ trợ tự động hóa.
  - `scripts/select_classical_demo_image.py`: quét WIDER FACE + Viola-Jones để chọn ảnh có đủ TP/FP/FN.
- `resources/`: tài liệu nguồn để đọc và trích ý.
  - `resources/books/`: PDF sách/chương sách.
  - `resources/papers/`: PDF/txt của paper.
  - `resources/ocr/`: text OCR/phụ trợ.
  - `resources/book_pages/`: ảnh trang sách đã tách.
- `archive/template_sections/`: phần template cũ không còn dùng trực tiếp.
- `build/`: file trung gian sinh ra khi biên dịch.

## Build

Pipeline khuyến nghị:

1. `xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex`
2. `biber main`
3. `xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex`

VS Code LaTeX Workshop recipe:

`xelatex synctex -> biber -> xelatex synctex`

Chi tiết cài môi trường và xử lý lỗi build:
- `COMPILE_REPORT.md`

## Ghi chú

- Hình phục vụ phân tích nên đặt trong `figures/`.
- Dữ liệu thô và file tách nhãn nên đặt trong `data/`.
- Tài liệu đọc tham khảo nên đặt trong `resources/`.
