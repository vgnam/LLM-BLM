# NVIDIA NIM Walk-Forward Backtest 2025 (GPT-OSS-20B)

Walk-forward rebalance backtest chạy **hoàn toàn trong năm 2025**, dùng **GPT-OSS-20B** (qua NVIDIA NIM) để tạo views cho các phương pháp LLM-BLM. Portfolio được tái cân bằng mỗi **30 hoặc 60 phiên giao dịch**: tại mỗi rebalance, LLM và optimizer được chạy lại với cửa sổ formation là **~252 phiên gần nhất (khoảng 1 năm)** trước điểm cắt, rồi giữ nguyên bộ trọng số cho N phiên kế tiếp.

Kết quả là một đường NAV liên tục (investable) cho mỗi method ở mỗi chu kỳ holding period. Trong các bảng, **in đậm** là giá trị tốt nhất cho metric đó và <u>gạch chân</u> là tốt thứ hai.

## Cấu trúc

- `config.json` / `config_used.json` — cấu hình thử nghiệm đã dùng
- `summary/all_method_metrics.csv` — bảng metrics tổng hợp
- `us_technology/`, `us_financials/`, `cross_asset_etfs/` — kết quả từng dataset
- `run_errors.json` — log lỗi (rỗng = chạy thành công toàn bộ)

## Phương pháp

| Method | Loại |
|---|---|
| **BL** | Baseline (Black-Litterman equilibrium prior) |
| **MVO** | Baseline (Mean-Variance Optimization) |
| **EW** | Baseline (Equal Weight) |
| **PairBL (GPT-OSS-20B)** | LLM (RelView-BL) |

## Dữ liệu & tham số

| Tham số | Giá trị |
|---|---|
| Năm backtest | 2025 (01-02 → 12-31) |
| Holding periods | **30 ngày** và **60 ngày** |
| Formation look-back | ~252 phiên giao dịch (~1 năm) |
| Rebalance đầu tiên | sau ≥40 phiên formation |
| Model | `openai/gpt-oss-20b` (NVIDIA NIM) |
| Repeats / min calls | 2 / 2 |
| Max pairs (views) | 15 |
| Temperature | 0.3 |
| Calibration | none |
| Tau | 0.025 |
| Risk aversion | 0.1 |
| Max weight | 15% |
| Transaction cost | 10 bps |
| Reasoning effort | `low` (ổn định output) |

## Kết quả

### Holding period 30 ngày

**us_technology**

| Method                | cumulative_return   | sharpe      | sortino     | max_drawdown   | final_nav   |
|:----------------------|:--------------------|:------------|:------------|:---------------|:------------|
| BL                    | **81.6%**           | <u>1.93</u> | <u>3.05</u> | -23.3%         | **1.82**    |
| MVO                   | 50.4%               | 1.67        | 2.66        | -18.6%         | 1.50        |
| EW                    | 39.1%               | 1.51        | 2.33        | -19.5%         | 1.39        |
| LLM-BLM (GPT-OSS-20B) | 44.3%               | 1.59        | 2.47        | <u>-18.6%</u>  | 1.44        |
| PairBL (GPT-OSS-20B)  | <u>78.6%</u>        | **2.24**    | **3.63**    | **-17.1%**     | <u>1.79</u> |

**us_financials**

| Method                | cumulative_return   | sharpe      | sortino     | max_drawdown   | final_nav   |
|:----------------------|:--------------------|:------------|:------------|:---------------|:------------|
| BL                    | **36.2%**           | <u>1.42</u> | <u>2.01</u> | -18.6%         | **1.36**    |
| MVO                   | 21.5%               | 1.05        | 1.43        | -17.5%         | 1.21        |
| EW                    | 23.9%               | 1.27        | 1.74        | <u>-15.1%</u>  | 1.24        |
| LLM-BLM (GPT-OSS-20B) | 30.4%               | **1.61**    | **2.36**    | **-12.9%**     | 1.30        |
| PairBL (GPT-OSS-20B)  | <u>32.9%</u>        | 1.35        | 1.88        | -18.6%         | <u>1.33</u> |

**cross_asset_etfs**

| Method                | cumulative_return   | sharpe      | sortino     | max_drawdown   | final_nav   |
|:----------------------|:--------------------|:------------|:------------|:---------------|:------------|
| BL                    | 24.6%               | 1.58        | 2.36        | -13.6%         | 1.25        |
| MVO                   | <u>31.3%</u>        | **2.73**    | **4.17**    | **-6.9%**      | <u>1.31</u> |
| EW                    | 18.7%               | 1.82        | 2.71        | -9.5%          | 1.19        |
| LLM-BLM (GPT-OSS-20B) | 21.5%               | 2.02        | 3.01        | <u>-8.4%</u>   | 1.22        |
| PairBL (GPT-OSS-20B)  | **33.1%**           | <u>2.13</u> | <u>3.17</u> | -12.1%         | **1.33**    |


### Holding period 60 ngày

**us_technology**

| Method                | cumulative_return   | sharpe      | sortino     | max_drawdown   | final_nav   |
|:----------------------|:--------------------|:------------|:------------|:---------------|:------------|
| BL                    | **80.8%**           | **2.17**    | **3.54**    | -23.3%         | **1.81**    |
| MVO                   | 44.3%               | 1.75        | 2.86        | **-18.6%**     | 1.44        |
| EW                    | 34.9%               | 1.54        | 2.42        | <u>-19.5%</u>  | 1.35        |
| LLM-BLM (GPT-OSS-20B) | <u>73.6%</u>        | <u>2.16</u> | <u>3.52</u> | -21.6%         | <u>1.74</u> |
| PairBL (GPT-OSS-20B)  | 69.0%               | 1.99        | 3.25        | -22.2%         | 1.69        |

**us_financials**

| Method                | cumulative_return   | sharpe      | sortino     | max_drawdown   | final_nav   |
|:----------------------|:--------------------|:------------|:------------|:---------------|:------------|
| BL                    | <u>19.1%</u>        | <u>0.96</u> | <u>1.33</u> | -18.6%         | <u>1.19</u> |
| MVO                   | 12.2%               | 0.73        | 0.99        | -17.5%         | 1.12        |
| EW                    | 12.0%               | 0.79        | 1.07        | **-15.1%**     | 1.12        |
| LLM-BLM (GPT-OSS-20B) | 8.9%                | 0.58        | 0.80        | <u>-16.8%</u>  | 1.09        |
| PairBL (GPT-OSS-20B)  | **20.6%**           | **1.04**    | **1.44**    | -17.9%         | **1.21**    |

**cross_asset_etfs**

| Method                | cumulative_return   | sharpe      | sortino     | max_drawdown   | final_nav   |
|:----------------------|:--------------------|:------------|:------------|:---------------|:------------|
| BL                    | 15.6%               | 1.21        | 1.80        | -13.6%         | 1.16        |
| MVO                   | <u>20.8%</u>        | **2.32**    | **3.52**    | **-6.9%**      | <u>1.21</u> |
| EW                    | 13.9%               | 1.55        | 2.31        | <u>-9.5%</u>   | 1.14        |
| LLM-BLM (GPT-OSS-20B) | 20.7%               | <u>1.90</u> | <u>2.91</u> | -9.7%          | 1.21        |
| PairBL (GPT-OSS-20B)  | **24.7%**           | 1.85        | 2.79        | -12.0%         | **1.25**    |


## Biểu đồ NAV

### us_technology

**30 ngày**

![us_technology 30d NAV](us_technology/holding_30/nav.png)

**60 ngày**

![us_technology 60d NAV](us_technology/holding_60/nav.png)

### us_financials

**30 ngày**

![us_financials 30d NAV](us_financials/holding_30/nav.png)

**60 ngày**

![us_financials 60d NAV](us_financials/holding_60/nav.png)

### cross_asset_etfs

**30 ngày**

![cross_asset_etfs 30d NAV](cross_asset_etfs/holding_30/nav.png)

**60 ngày**

![cross_asset_etfs 60d NAV](cross_asset_etfs/holding_60/nav.png)

## Biểu đồ Drawdown

### us_technology

![us_technology drawdown](us_technology/holding_30/drawdown.png)

### us_financials

![us_financials drawdown](us_financials/holding_30/drawdown.png)

### cross_asset_etfs

![cross_asset_etfs drawdown](cross_asset_etfs/holding_30/drawdown.png)

## Biểu đồ Metrics

![us_technology metrics](us_technology/holding_30/metrics.png)

![us_financials metrics](us_financials/holding_30/metrics.png)

![cross_asset_etfs metrics](cross_asset_etfs/holding_30/metrics.png)

## Biểu đồ trọng số (Weight Heatmap)

![us_technology weights](us_technology/holding_30/weights_heatmap.png)

![us_financials weights](us_financials/holding_30/weights_heatmap.png)

![cross_asset_etfs weights](cross_asset_etfs/holding_30/weights_heatmap.png)

## Nhận xét chính

- **30 ngày tốt hơn 60 ngày** ở hầu hết các dataset: tái cân bằng thường xuyên hơn giúp bám sát views và giảm drawdown.
- **PairBL (GPT-OSS-20B)** đạt Sharpe cao nhất ở us_technology 30d (2.24) và dẫn đầu cumulative return ở us_financials / cross_asset_etfs.
- **LLM-BLM** vượt baselines về Sharpe ở us_financials 30d (1.61) với drawdown thấp nhất (-12.9%).
- **BL** là baseline mạnh nhờ prior equilibrium — các method LLM cần vượt qua điểm này.
- MVO rất mạnh ở cross_asset_etfs (Sharpe 2.73, Max DD chỉ -6.9%).

## Guardrails

- Các LLM calls được thực hiện sau cửa sổ test với tên asset nhận diện được, nên kết quả có thể chịu ảnh hưởng của training-cutoff hoặc leakage phía nhà cung cấp dù pipeline số là point-in-time.
- Đây là kết quả nghiên cứu, không phải bằng chứng cho hiệu suất live.

## Chạy lại

```powershell
$env:NVIDIA_API_KEY = "<key>"
py run_nvidia_nim_walkforward_2025.py --dry-run   # ước lượng số calls
py run_nvidia_nim_walkforward_2025.py             # chạy đầy đủ
```
