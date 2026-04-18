# Phân Tích Kết Quả SCRFD 2.5G Cosine: Default Scale Set vs Paper SR12

## 1. Mục tiêu

Báo cáo này phân tích hai câu hỏi:

1. Vì sao `ASR+JSAR` đều tốt hơn `baseline` ở cả hai scale set.
2. Vì sao scale set `paper_sr12_scale_set` lại cho kết quả tuyệt đối kém hơn scale set mặc định trong source code, dù đây là scale set được mô tả là tốt trong phần SR search của paper.

Phạm vi báo cáo chỉ dùng các kết quả đã train/eval trong thư mục này. Phần giải thích tập trung vào:

- metric WIDERFace,
- phân phối scale được dùng trong train,
- phân phối kích thước mặt sau SR simulation,
- mật độ positive assignment mà `JSAR` bổ sung cho tiny faces,
- hard-subset analysis ở run `default_source_scale_set`.

## 2. Phương pháp cải tiến

### 2.1. Adaptive Sample Redistribution (ASR)

`ASR` thay việc lấy mẫu scale augmentation tĩnh bằng một vòng lặp phản hồi theo thống kê train. Ở cuối mỗi epoch, hook thu các thống kê theo size bin, tính độ khó của từng bin, rồi cập nhật xác suất chọn các crop ratio cho epoch kế tiếp.

![ASR schematic](report_assets/asr_schematic.png)

Trong pipeline SCRFD này:

- `scale nhỏ` tương ứng với crop nhỏ hơn, tức `zoom-in`, nên mặt sau resize sẽ lớn hơn.
- `scale lớn` tương ứng với crop lớn hơn, tức `zoom-out`, nên mặt sau resize sẽ nhỏ hơn.

Vì vậy, khi `ASR` tăng xác suất ở các scale lớn hơn, phân phối train sẽ dịch về phía nhiều tiny faces hơn.

### 2.2. Joint SampleAssignment Redistribution (JSAR)

`JSAR` không đổi backbone, neck, head hay inference. Nó chỉ sửa phần target assignment khi train để tiny faces nhận được nhiều positive anchors hơn nếu assigner gốc đang under-supervise chúng.

![JSAR schematic](report_assets/jsar_schematic.png)

Trong implementation hiện tại, hiệu ứng quan trọng nhất là:

- tiny faces có số positives trên mỗi GT tăng rõ rệt,
- small/medium/large gần như giữ nguyên,
- do đó mật độ supervision tăng đúng vào vùng khó nhất của WIDERFace Hard.

## 3. Tổng quan kết quả

### 3.1. Metric tổng hợp

![WIDERFace metrics across scale sets](report_assets/metrics_by_scale_set.png)

| Scale set | Model | easy_AP | medium_AP | hard_AP | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| Default source | Baseline | 0.9140 | 0.8969 | 0.7249 | 0.8453 |
| Default source | ASR+JSAR | 0.9039 | 0.8925 | 0.7690 | 0.8551 |
| Paper SR12 | Baseline | 0.9016 | 0.8842 | 0.7073 | 0.8310 |
| Paper SR12 | ASR+JSAR | 0.8845 | 0.8720 | 0.7457 | 0.8341 |

### 3.2. Delta của ASR+JSAR so với baseline

![ASR delta by scale set](report_assets/asr_delta_by_scale_set.png)

Điểm nhất quán nhất giữa hai scale set là:

- `hard_AP` đều tăng mạnh,
- `easy_AP` và `medium_AP` đều giảm nhẹ,
- `mAP` tăng nhưng mức tăng nhỏ hơn lợi ích riêng trên Hard.

Điều này cho thấy `ASR+JSAR` không phải cải tiến đồng đều cho mọi kích thước mặt. Nó chủ yếu chuyển năng lực học sang nhóm tiny/hard faces.

## 4. Vì sao ASR+JSAR tốt hơn baseline ở cả hai scale set

### 4.1. JSAR luôn tăng supervision density cho tiny faces

![JSAR tiny supervision](report_assets/jsar_tiny_supervision.png)

Ở cả hai scale set, `JSAR` gần như giữ nguyên mức boost cho tiny faces:

- default scale set:
  - tiny positives / GT: `1.73 -> 2.67`
  - boost ratio: `1.542x`
- paper SR12 scale set:
  - tiny positives / GT: `1.73 -> 2.66`
  - boost ratio: `1.538x`

Điều này rất quan trọng: `JSAR` vẫn hoạt động gần như giống hệt nhau ở cả hai scale pool. Nói cách khác, phần assignment redistribution của phương pháp là ổn định và không phải nguyên nhân gây tụt hiệu năng tuyệt đối ở `paper_sr12_scale_set`.

### 4.2. Gain trên hard tập trung vào tiny faces

Với `default_source_scale_set`, hard-subset analysis đầy đủ cho thấy lợi ích của `ASR+JSAR` tập trung gần như hoàn toàn vào các mặt rất nhỏ.

![Default hard size gain](report_assets/default_hard_size_gain.png)

Các delta recall proxy theo hard-size bin:

- `[0, 8)`: `+0.138`
- `[8, 16)`: `+0.119`
- `[16, 32)`: `+0.009`
- từ `32` px trở lên: gần như bằng `0` hoặc âm nhẹ

Điều này khớp trực tiếp với mục tiêu của `JSAR`: tăng mật độ positive assignment cho tiny faces vốn rất dễ bị under-supervised.

Ngoài ra, ở run mặc định:

- hard recall proxy tăng `0.8243 -> 0.8659`
- precision proxy cũng tăng `0.0153 -> 0.0182`

Tức là `ASR+JSAR` không chỉ bắt được thêm hard faces mà còn không phải đánh đổi bằng việc phun ra quá nhiều false positives ở ngưỡng đang đo.

### 4.3. ASR thay đổi sample distribution, còn JSAR biến thay đổi đó thành supervision hữu ích

Hai thành phần đóng vai trò khác nhau:

- `ASR` quyết định model nhìn thấy loại crop nào nhiều hơn.
- `JSAR` đảm bảo tiny faces trong các crop đó thật sự nhận được đủ positive anchors.

Nếu chỉ có `ASR` mà assignment vẫn quá chặt, nhiều tiny faces vẫn bị under-supervise. Nếu chỉ có `JSAR` mà SR không thay đổi, model vẫn chưa nhìn thấy đủ nhiều tình huống tiny/hard để tận dụng toàn bộ lợi ích của assignment expansion. Hai phần này bổ sung cho nhau.

## 5. Vì sao paper SR12 scale set cho kết quả tuyệt đối kém hơn

Đây là điểm cần phân biệt rõ:

- `ASR+JSAR` vẫn cải thiện so với baseline ngay cả trên `paper_sr12_scale_set`.
- Nhưng cả baseline lẫn ASR+JSAR trên `paper_sr12_scale_set` đều kém hơn run dùng scale set mặc định.

Nói cách khác, vấn đề nằm nhiều hơn ở **candidate scale pool** của run `paper_sr12_scale_set`, không phải vì `ASR+JSAR` ngừng hoạt động.

### 5.1. Hình học của hai scale pool rất khác nhau

![Scale pool geometry](report_assets/scale_pool_geometry.png)

Từ thống kê scale pool:

- default baseline:
  - expected magnification `E[1/scale] = 1.270`
  - extreme zoom-in mass (`scale <= 0.6`) = `0.300`
  - extreme zoom-out mass (`scale >= 2.0`) = `0.100`
- paper SR12 baseline:
  - expected magnification `E[1/scale] = 0.890`
  - extreme zoom-in mass (`scale <= 0.6`) = `0.083`
  - extreme zoom-out mass (`scale >= 2.0`) = `0.250`

Nghĩa là scale set của paper, khi áp dụng trong setting hiện tại, có xu hướng:

- ít khả năng `zoom-in` mạnh hơn,
- nhiều khả năng `zoom-out` mạnh hơn,
- và nhìn chung đẩy phân phối train sang một bài toán khó hơn.

### 5.2. Phân phối kích thước mặt sau SR bị đẩy quá mạnh sang tiny regime

![Distribution pressure](report_assets/distribution_pressure.png)

So với default scale set, `paper_sr12_scale_set` tạo áp lực mạnh hơn lên vùng tiny:

- default baseline:
  - tiny ratio = `0.475`
  - median face size = `17.12`
  - tiny -> `>=16px` promotion ratio = `0.223`
- paper SR12 baseline:
  - tiny ratio = `0.582`
  - median face size = `12.97`
  - tiny -> `>=16px` promotion ratio = `0.092`

Điều này có hai hệ quả:

1. Model phải học trên nhiều tiny faces hơn đáng kể.
2. Nhưng các face đó lại ít được “đẩy” sang vùng kích thước dễ học hơn.

Vì vậy, độ khó của train distribution tăng lên mạnh hơn mức mà JSAR có thể bù hết.

### 5.3. ASR trên paper SR12 không cứu được vì không gian lựa chọn bản thân nó đã bất lợi hơn

Trên `paper_sr12_scale_set`, `ASR` vẫn tự điều chỉnh xác suất để tránh bớt các mức quá cực đoan, nhưng candidate pool vẫn bị ràng buộc bởi hai đặc điểm:

- thiếu các lựa chọn zoom-in mạnh kiểu default source scale set,
- có thêm các lựa chọn zoom-out mạnh như `2.3`, `2.6`.

Kết quả là:

- `ASR+JSAR` vẫn thắng baseline trong cùng scale set,
- nhưng trần hiệu năng của toàn bộ run thấp hơn, vì không gian scale cho SR đã đẩy bài toán sang phía quá nhiều tiny faces.

### 5.4. Giải thích đúng phạm vi

Từ các số đo hiện có, cách diễn giải đúng là:

- scale set `paper_sr12_scale_set` không nhất thiết “xấu” trong mọi thiết lập,
- nhưng trong **public codebase + 80 epochs + cosine schedule + cấu hình train hiện tại**, nó tạo ra một train distribution khó hơn default scale set,
- nên hiệu năng tuyệt đối thấp hơn ở cả baseline lẫn ASR+JSAR.

## 6. Ghi chú về dữ liệu phân tích hard-subset của paper SR12

Hard-subset per-image analysis chỉ được dùng mạnh cho `default_source_scale_set`. Với `paper_sr12_scale_set`, bundle predictions đã tải về không đầy đủ cho improved run, nên phần hard-subset per-image của scale set này không đủ tin cậy để dùng làm bằng chứng chính cho báo cáo.

Vì vậy, các kết luận về `paper_sr12_scale_set` trong báo cáo này chủ yếu dựa trên:

- WIDERFace metric chính thức,
- scale probability history,
- face-size distribution simulation,
- JSAR assignment summary.

## 7. Kết luận

### 7.1. Về ASR+JSAR

`ASR+JSAR` tốt hơn baseline ở cả hai scale set vì:

- `ASR` thay đổi distribution của các sample được nhìn thấy trong train,
- `JSAR` tăng mật độ positive supervision đúng ở tiny faces,
- hard gain tập trung mạnh ở các bin `[0, 8)` và `[8, 16)`, tức đúng vùng khó nhất của WIDERFace Hard.

### 7.2. Về default source scale set vs paper SR12 scale set

Trong setting đang thử nghiệm:

- default source scale set cho kết quả tốt hơn tuyệt đối,
- vì nó giữ cân bằng tốt hơn giữa `zoom-in` và `zoom-out`,
- còn `paper_sr12_scale_set` đẩy quá nhiều samples về tiny regime và làm giảm khả năng “promote” tiny faces sang vùng kích thước dễ học hơn.

Nói ngắn gọn:

- `ASR+JSAR` là cải tiến có hiệu quả ổn định theo hướng tiny/hard faces.
- Nhưng chất lượng cuối cùng vẫn phụ thuộc mạnh vào candidate scale pool mà `ASR` được phép redistributes trên đó.
