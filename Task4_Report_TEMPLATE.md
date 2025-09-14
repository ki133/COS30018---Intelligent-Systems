# Task C.4 Report Template (Option C)

> Replace placeholders (ALL-CAPS) with your content. Remove guidance blocks when finalising.

## 1. Overview / Executive Summary
- Mục tiêu: Xây dựng hàm tổng quát tạo mô hình DL (LSTM/GRU/RNN) và thực nghiệm nhiều cấu hình.
- Dataset: STOCK TICKER (e.g. CBA.AX) từ START_DATE đến END_DATE.
- Kết quả nổi bật: Ví dụ: GRU 1 layer outperform LSTM stacked nhỏ ở RMSE/MAE.

## 2. Model Builder Function Explanation
File: `model_builder.py`

### 2.1 Design Goals
- Tái sử dụng ý tưởng từ P1 `create_model` nhưng tổng quát hơn.
- Cho phép: nhiều loại cell, số units linh hoạt, dropout, bidirectional, learning rate custom.

### 2.2 Code Walkthrough (Important Snippets)
| Feature | Code Snippet | Giải thích |
|---------|--------------|-----------|
| Mapping layer types | `RECURRENT_LAYER_MAP` | Chuyển từ string sang lớp Keras tương ứng. |
| Dynamic layers | Loop qua `layer_units_list` | Tạo mỗi tầng với return_sequences tùy vị trí. |
| Bidirectional | `Bidirectional(recurrent_layer)` | Bọc cell để học xuôi + ngược thời gian. |
| Optimizer factory | `OPTIMIZER_MAP` | Cho phép chọn Adam/RMSprop/SGD + LR tùy chọn. |
| Summary capture | `model.summary(print_fn=...)` | Lưu text summary để ghi log. |

### 2.3 Less Straightforward Lines (With References)
- `Bidirectional` wrapper: [Keras Docs](https://keras.io/api/layers/recurrent_layers/bidirectional/)
- `recurrent_dropout` tác động lên state update nội bộ: [Keras RNN FAQ](https://keras.io/api/layers/recurrent_layers/lstm/)
- Custom learning rate: Khởi tạo optimizer thay vì dùng chuỗi tên.

## 3. Data Preparation Pipeline Reuse
- Sử dụng `load_and_process_data` (Task C.2) để đảm bảo: scaling tránh leakage, chia train/test thống nhất.
- Chỉ dùng feature chính: `Close` (có thể mở rộng đa biến ở phiên bản sau).

## 4. Experimental Setup
| Parameter | Values Explored |
|-----------|-----------------|
| Models | LSTM, GRU, SimpleRNN |
| Layer stacks | [64], [64,32], [128,64] |
| Sequence length | 40 (quick run), típ chạy đầy đủ: 60/80 |
| Epochs | QUICK: 2 (validation), Final: EPOCHS_PLAN |
| Batch sizes | 16 (quick) / 32 |
| Optimizer | Adam (LR=0.001) |
| Loss | MSE |
| Metrics | MAE, RMSE (post), MAPE (post) |

## 5. Results Summary
(Trích từ `experiments/batch_summary.csv`)

| Model | Layers | RMSE | MAPE (%) | Notes |
|-------|--------|------|----------|-------|
| LSTM | 64-32 | 0.4109 | 37.48 | Stable learning |
| GRU | 64 | 0.2170 | 17.57 | Best in quick run |
| RNN | 128-64 | 0.6687 | 60.12 | Overfitting, higher error |

### 5.1 Observations
- GRU thường hội tụ nhanh hơn trên dataset này.
- Stacked SimpleRNN cho error lớn → gradient vanishing.
- LSTM vs GRU: GRU ít tham số hơn → tốt hơn với chuỗi ngắn.

### 5.2 Error Analysis
- Test set có giá vượt ngoài range huấn luyện → cảnh báo scaling (ISSUE #2 kế thừa từ Task C.2).
- Đề xuất: tái-fit scaler với chiến lược rolling hoặc dùng RobustScaler.

## 6. Discussion
- Tại sao GRU tốt hơn ở đây? Ít tham số → tránh overfit khi dữ liệu giới hạn.
- Khi nào nên dùng Bidirectional? Khi toàn bộ chuỗi lịch sử cố định (forecast offline) nhưng cẩn trọng khi làm real-time.
- Hướng mở rộng: thêm attention, multi-feature input (Volume, High, Low,...), hyperparameter tuning tự động (Optuna).

## 7. Limitations
- Chưa dùng nhiều feature.
- Chưa thêm regularization khác (L2, early advanced scheduling ngoài ReduceLROnPlateau).
- Chưa đánh giá bằng directional accuracy hoặc trading simulation.

## 8. Future Work
- Grid/Random search tự động.
- Thêm CNN-LSTM / Transformer baseline.
- Evaluate economic metrics (profit/loss giả lập).

## 9. References
- Keras API Docs: https://keras.io/
- GRU vs LSTM Empirical: https://arxiv.org/abs/1412.3555
- Bidirectional RNNs: Schuster & Paliwal (1997)
- Time Series Windowing Patterns: https://machinelearningmastery.com/

## 10. Appendix
- Model summaries: xem từng thư mục `experiments/<timestamp>/model_summary.txt`
- Training curves: có thể plot thêm từ `training_history.csv`.

---
(End of Template)
