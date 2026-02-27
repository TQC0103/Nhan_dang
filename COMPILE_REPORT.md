# Hướng Dẫn Cài Môi Trường Và Compile Report

Tài liệu này mô tả cách cài đủ công cụ để build report LaTeX của repo này.

## 1. Yêu cầu

- `XeLaTeX`
- `Biber`
- Font Unicode (khuyến nghị: Times New Roman)
- (Tùy chọn) VS Code + extension `LaTeX Workshop`

## 2. Cài đặt nhanh theo hệ điều hành

### Windows (khuyến nghị)

1. Cài MiKTeX:
   - Qua winget:
     - `winget install MiKTeX.MiKTeX`
2. Mở terminal mới và kiểm tra:
   - `xelatex --version`
   - `biber --version`

Nếu thiếu `biber`, mở MiKTeX Console và cập nhật package.

### macOS

1. Cài MacTeX:
   - `brew install --cask mactex`
2. Kiểm tra:
   - `xelatex --version`
   - `biber --version`

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y texlive-full biber
xelatex --version
biber --version
```

## 3. Compile report từ terminal

Tại thư mục gốc repo, chạy đúng thứ tự:

```bash
xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
biber main
xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
```

`main.pdf` sẽ được tạo ở thư mục gốc.

## 4. Compile trong VS Code

1. Cài extension `LaTeX Workshop`.
2. Chọn recipe:
   - `xelatex synctex -> biber -> xelatex synctex`
3. Build bằng `Ctrl+Alt+B`.

## 5. Lỗi thường gặp

- `biber: command not found`
  - Chưa cài `biber` hoặc PATH chưa cập nhật.
  - Cài lại theo mục 2, rồi mở terminal mới.

- Thiếu font Unicode hoặc lỗi dấu tiếng Việt
  - Cài font hệ thống như Times New Roman.
  - Đảm bảo compile bằng `xelatex`, không dùng `pdflatex`.

- Citation chưa lên số
  - Chạy lại đủ chuỗi:
    - `xelatex -> biber -> xelatex -> xelatex`

