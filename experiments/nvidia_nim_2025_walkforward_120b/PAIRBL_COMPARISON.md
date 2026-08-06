# Walk-Forward 2025: PairBL Model Comparison (GPT-OSS-20B vs GPT-OSS-120B)

Backtest hoàn toàn trong 2025, tái cân bằng 30/60 ngày với formation ~252 phiên. Bảng dưới so sánh riêng method **PairBL (RelView-BL)** giữa hai model qua NVIDIA NIM. **In đậm** là model tốt hơn cho metric đó, <u>gạch chân</u> là tốt thứ hai.

## Holding period 30 ngày

### US Technology Equities

| Method                | Cum. return   | Ann. return   | Ann. vol   | Sharpe   | Sortino   | Calmar   | Max DD     |   Final NAV |
|:----------------------|:--------------|:--------------|:-----------|:---------|:----------|:---------|:-----------|------------:|
| PairBL (GPT-OSS-20B)  | 78.6%         | 100.6%        | 33.6%      | **2.24** | **3.63**  | **5.88** | **-17.1%** |        1.79 |
| PairBL (GPT-OSS-120B) | 67.8%         | 86.1%         | 36.3%      | 1.89     | 2.97      | 4.14     | -20.8%     |        1.68 |

### US Financial Equities

| Method                | Cum. return   | Ann. return   | Ann. vol   |   Sharpe |   Sortino |   Calmar | Max DD   |   Final NAV |
|:----------------------|:--------------|:--------------|:-----------|---------:|----------:|---------:|:---------|------------:|
| PairBL (GPT-OSS-20B)  | 32.9%         | 40.7%         | 28.3%      |     1.35 |      1.88 |     2.19 | -18.6%   |        1.33 |
| PairBL (GPT-OSS-120B) | 20.6%         | 25.1%         | 26.7%      |     0.98 |      1.32 |     1.37 | -18.4%   |        1.21 |

### Cross-Asset ETFs

| Method                | Cum. return   | Ann. return   | Ann. vol   | Sharpe   | Sortino   | Calmar   | Max DD    | Final NAV   |
|:----------------------|:--------------|:--------------|:-----------|:---------|:----------|:---------|:----------|:------------|
| PairBL (GPT-OSS-20B)  | 33.1%         | 40.9%         | 16.7%      | 2.13     | 3.17      | 3.38     | -12.1%    | 1.33        |
| PairBL (GPT-OSS-120B) | **35.4%**     | **43.9%**     | 12.0%      | **3.09** | **4.89**  | **7.89** | **-5.6%** | **1.35**    |

## Holding period 60 ngày

### US Technology Equities

| Method                | Cum. return   | Ann. return   | Ann. vol   | Sharpe   | Sortino   | Calmar   | Max DD   |   Final NAV |
|:----------------------|:--------------|:--------------|:-----------|:---------|:----------|:---------|:---------|------------:|
| PairBL (GPT-OSS-20B)  | 69.0%         | 108.4%        | 41.0%      | 1.99     | 3.25      | 4.88     | -22.2%   |        1.69 |
| PairBL (GPT-OSS-120B) | 78.3%         | 124.8%        | 39.1%      | **2.27** | **3.67**  | **5.86** | -21.3%   |        1.78 |

### US Financial Equities

| Method                | Cum. return   | Ann. return   | Ann. vol   | Sharpe   | Sortino   | Calmar   | Max DD   | Final NAV   |
|:----------------------|:--------------|:--------------|:-----------|:---------|:----------|:---------|:---------|:------------|
| PairBL (GPT-OSS-20B)  | **20.6%**     | **30.0%**     | 29.4%      | **1.04** | **1.44**  | **1.68** | -17.9%   | **1.21**    |
| PairBL (GPT-OSS-120B) | 11.7%         | 16.8%         | 29.6%      | 0.67     | 0.93      | 0.92     | -18.2%   | 1.12        |

### Cross-Asset ETFs

| Method                | Cum. return   | Ann. return   | Ann. vol   |   Sharpe |   Sortino |   Calmar | Max DD   | Final NAV   |
|:----------------------|:--------------|:--------------|:-----------|---------:|----------:|---------:|:---------|:------------|
| PairBL (GPT-OSS-20B)  | **24.7%**     | **36.1%**     | 17.5%      |     1.85 |      2.79 |     3.01 | -12.0%   | **1.25**    |
| PairBL (GPT-OSS-120B) | 19.8%         | 28.7%         | 12.7%      |     2.06 |      3.08 |     3.66 | -7.9%    | 1.20        |

## Biểu đồ NAV

### US Technology Equities

**GPT-OSS-20B**

![us_technology 20b NAV](experiments/nvidia_nim_2025_walkforward/us_technology/holding_30/nav.png)

**GPT-OSS-120B**

![us_technology 120b NAV](experiments/nvidia_nim_2025_walkforward_120b/us_technology/holding_30/nav.png)

### US Financial Equities

**GPT-OSS-20B**

![us_financials 20b NAV](experiments/nvidia_nim_2025_walkforward/us_financials/holding_30/nav.png)

**GPT-OSS-120B**

![us_financials 120b NAV](experiments/nvidia_nim_2025_walkforward_120b/us_financials/holding_30/nav.png)

### Cross-Asset ETFs

**GPT-OSS-20B**

![cross_asset_etfs 20b NAV](experiments/nvidia_nim_2025_walkforward/cross_asset_etfs/holding_30/nav.png)

**GPT-OSS-120B**

![cross_asset_etfs 120b NAV](experiments/nvidia_nim_2025_walkforward_120b/cross_asset_etfs/holding_30/nav.png)
