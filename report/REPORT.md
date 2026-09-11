# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU – Tesla T4 (device = cuda)

**Python / PyTorch / Ultralytics:** 3.13.15 / 2.11.0+cu128 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không. Giữ nguyên checkpoint, mã nguồn và ngưỡng của bài lab (detection 0.35, segmentation 0.35); ba ảnh mẫu đều qua kiểm tra SHA-256.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic` (COCO image id 210273, 640 x 428 px).

- **Record hạng 1:** `class_id = 468`, `class_name = cab`, `rank = 1`, `score = 0.510915`, `taxonomy_name = ImageNet-1K`.
- **Record này mô tả toàn ảnh như thế nào?** Record không có toạ độ và không trỏ tới vùng ảnh nào; nó gán một nhãn duy nhất cho cả bức ảnh. Ảnh `traffic` thực tế có nhiều xe buýt, nhiều ô tô và một người đi bộ, nhưng đầu ra phân loại chỉ nói cả ảnh giống lớp `cab` nhất. Bốn record còn lại (`minibus` 0.164284, `police_van` 0.085848, `recreational_vehicle` 0.054110, `streetcar` 0.048193) cho thấy model đang phân vân trong nhóm phương tiện.
- **Ai định nghĩa class list?** Taxonomy ImageNet-1K đi kèm checkpoint `yolo11n-cls.pt`, tức danh sách 1000 lớp do bộ dữ liệu huấn luyện quy định. Model không tự nghĩ ra lớp mới; muốn có lớp theo nghiệp vụ thì phải định nghĩa trong guideline rồi gán nhãn và huấn luyện lại.
- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?** Tên lớp có thể trùng hoặc lệch nghĩa giữa các taxonomy (`cab` của ImageNet-1K khác `car` của COCO-80), còn `class_id` chỉ có nghĩa bên trong đúng một taxonomy. Giữ đủ ba trường mới truy ngược được nhãn về đúng bảng lớp.
- **Nếu ảnh có nhiều chủ thể, guideline cần quy định gì?** Phải nói rõ single-label hay multi-label, quy tắc chọn chủ thể chính (diện tích lớn nhất, ở trung tâm, hay theo mục đích nghiệp vụ), và ca không xác định thì chuyển escalation. Đây là rủi ro thật: sample `kitchen` có rank 1 là `gong` với score 0.420192 – ảnh gian bếp treo nhiều xoong chảo tròn nên model gán nhầm thành cái cồng.
- **Vì sao model score không phải ground truth?** `score` chỉ là mức tin cậy nội bộ của model dùng để xếp hạng, không phải điểm chất lượng nhãn. Sample `dining` có rank 1 `restaurant` 0.792161 hợp lý, trong khi `kitchen` có rank 1 `gong` 0.420192 sai hẳn: score không tự chứng minh prediction đúng.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen` (COCO image id 397133, 640 x 427 px), `score_threshold = 0.35`.

- **Một record:** `class_name = person`, `score = 0.912625`, `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`, `bbox_width = 113.58`, `bbox_height = 279.68`.
- **Diễn giải vị trí box:** gốc toạ độ (0, 0) ở góc trên bên trái. Hộp bắt đầu tại x = 385.33 (khoảng 60 phần trăm chiều ngang ảnh rộng 640 px) và y = 69.24 (gần mép trên), kết thúc tại x = 498.92, y = 348.92. Vật thể nằm ở nửa phải ảnh, cao 279.68 px tức khoảng 65 phần trăm chiều cao ảnh 427 px – đúng với người đầu bếp đang đứng quay lưng trong hình minh hoạ.
- **So sánh số prediction ở các threshold (cùng ảnh `kitchen`):** conf 0.20 giữ 17 prediction; conf 0.35 giữ 11 prediction; conf 0.60 chỉ còn 6 prediction.
- **Điều gì thay đổi với độ bao phủ và khối lượng review?** Hạ xuống 0.20 giữ thêm 6 prediction (thêm `spoon`, `potted plant`, `dining table` và vài `bowl`/`cup` nhỏ): bao phủ tốt hơn nhưng reviewer phải xem nhiều box nhiễu hơn. Nâng lên 0.60 chỉ còn `person`, `bowl`, `oven`: danh sách gọn nhưng mất hẳn các `cup` và `bowl` nhỏ ở xa, đó là bỏ sót thật chứ không phải vật thể không tồn tại. Threshold là cấu hình lọc prediction, không phải quy tắc cho phép bỏ qua vật thể khi tạo ground truth.
- **Một quy tắc box chặt:** box phải ôm sát các pixel ngoài cùng còn nhìn thấy của vật thể, sai số không quá 2 px mỗi cạnh, không bao bóng đổ hay phần phản chiếu; vật bị che thì chỉ khoanh phần nhìn thấy được.
- **Object bị che khuất hoặc cắt mép:** guideline phải quy định box được kẹp về trong khung ảnh. Trong output có record `person` thứ hai với `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]` và `bowl` với x_min = 0.32, tức dính sát mép trái. Cần thêm ngưỡng phần trăm vật thể còn nhìn thấy tối thiểu để vẫn gán nhãn, và quy tắc cho vật bị che thành nhiều mảnh. Nếu không xác định được vật thể là gì thì escalation chứ không đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- **Một record:** `instance_id = kitchen-002`, `class_name = bowl`, `score = 0.735744`, `bbox_xyxy = [31.44, 343.77, 100.89, 385.55]`, `polygon_point_count = 67`, 5 điểm đầu của `polygon_xy` là [[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]].
- **Polygon bổ sung chi tiết gì so với box?** Box chỉ là hình chữ nhật bao ngoài nên luôn chứa cả phần nền; polygon bám theo biên thật của instance nên giữ được hình dạng, tính được diện tích và tách vật thể khỏi nền. Mức chi tiết thay đổi theo vật thể: `kitchen-005` (`dining table`) cần 598 điểm vì biên phức tạp và bị đồ vật che, còn `kitchen-010` (`bowl`, ở xa) chỉ có 12 điểm.
- **`instance_id` dùng để làm gì và không phải loại ID nào?** Nó định danh từng đối tượng riêng trong bộ output của bài lab: `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010` đều thuộc lớp `bowl` nhưng là bốn instance khác nhau, mỗi instance một polygon. Đây không phải class ID (cả bốn dùng chung một `class_id` của lớp `bowl`) và không phải tracking ID (không theo dõi cùng một vật qua nhiều khung hình).
- **Một quy tắc biên mask:** bám đúng biên nhìn thấy, không lấn sang nền quá 1 đến 2 px; khi vật bị che chia thành nhiều mảnh thì không nối liền qua vùng bị che trừ khi guideline cho phép.
- **Vùng mờ, tiếp xúc hoặc bị che:** các `bowl` đặt sát nhau trên mặt bàn và `kitchen-005` (`dining table`, `bbox_xyxy = [0.97, 234.74, 345.86, 423.87]`) chạm mép dưới ảnh là những ca cần guideline quy định trước: tách hai vật chạm nhau ở đâu, có vẽ tiếp phần ngoài khung hình không, xử lý vùng mờ nhoè thế nào. Chưa rõ thì escalation chứ annotator không tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: `class_id` + `class_name` + tên taxonomy | `kitchen` rank 1 = `gong` (0.420192) trong khi ảnh là gian bếp; ảnh nhiều chủ thể nên nhãn đơn dễ gây tranh cãi | Gán nhãn theo quy tắc chủ thể chính trong guideline, không chép prediction; đánh dấu ca mơ hồ để hỏi lại | Nhãn có nằm trong taxonomy đã chốt không, các ảnh nhiều chủ thể có được xử lý nhất quán không |
| Phát hiện vật thể | Một `class_name` kèm một `bbox_xyxy` theo pixel cho mỗi vật thể | Ở ngưỡng 0.60 mất các `cup` và `bowl` nhỏ; ở 0.20 xuất hiện thêm box chồng lấn như `potted plant`, `spoon` | Khoanh đủ mọi vật thể thuộc phạm vi kể cả vật nhỏ, bị che hoặc cắt mép; không lấy threshold làm cớ bỏ sót | Box có bám sát không, có bỏ sót vật thể không, có hai box cho cùng một vật không |
| Instance segmentation | Một `class_name` kèm một polygon cho mỗi instance, có `instance_id` riêng | `kitchen-010` (`bowl`) chỉ có 12 điểm nên biên rất thô; nhiều `bowl` nằm sát nhau khó tách | Tách đúng từng instance và bám biên; ghi chú vùng bị che | Số instance có khớp số vật thể không, biên có lấn nền không, các vật cùng lớp chạm nhau có bị gộp không |

## 5. An toàn dữ liệu

- **Một quy tắc bảo vệ dữ liệu:** chỉ dùng ba ảnh công khai đã được notebook cố định và kiểm tra SHA-256 (COCO val2017, kèm `IMAGE_ATTRIBUTION.md` ghi tác giả và giấy phép CC BY 2.0); không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab và repository công khai. Họ tên và MSSV chỉ xuất hiện trong tên repository, không nằm trong báo cáo hay output.
- **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:** GV/Lab Coach qua kênh hỗ trợ của lớp; không tiếp tục xử lý và không chia sẻ lại dữ liệu đó.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
