# NVIDIA NIM portfolio experiment results

## Data windows

Views and portfolio weights use only 2025 observations. Fixed weights are evaluated on the available 2026 sessions; no 2026 return enters a view or optimizer input.

- `us_technology`: 250 formation rows (2025-01-02 to 2025-12-31), 144 test rows (2026-01-02 to 2026-07-30).
- `us_financials`: 250 formation rows (2025-01-02 to 2025-12-31), 144 test rows (2026-01-02 to 2026-07-30).
- `cross_asset_etfs`: 250 formation rows (2025-01-02 to 2025-12-31), 144 test rows (2026-01-02 to 2026-07-30).

## NVIDIA NIM model status

| Requested model | API model | Status | HTTP |
|---|---|---|---:|
| `qwen3-next-80b-a3b-instruct:latest` | `qwen/qwen3-next-80b-a3b-instruct` | unavailable | 410 |
| `mistralai/mixtral-8x7b-instruct-v0-1` | `mistralai/mixtral-8x7b-instruct-v0.1` | unavailable | 410 |
| `openai/gpt-oss-20b` | `openai/gpt-oss-20b` | available | 200 |

Unavailable models were not substituted.

## Results

Per-dataset metrics are saved in full under `summary/`.

```text
         Dataset                 Method  cumulative_return   sharpe  max_drawdown
   us_technology                     BL           0.495514 1.724112     -0.257546
   us_technology                    MVO           0.496649 1.840696     -0.231825
   us_technology                     EW           0.189810 1.274387     -0.157717
   us_technology   BLM_LLM__gpt_oss_20b           0.495514 1.724112     -0.257546
   us_technology RelViewBL__gpt_oss_20b           0.584813 1.963260     -0.256762
   us_financials                     BL           0.049939 0.457685     -0.201017
   us_financials                    MVO           0.075497 0.633562     -0.182571
   us_financials                     EW           0.080842 0.818481     -0.131619
   us_financials   BLM_LLM__gpt_oss_20b           0.066490 0.590883     -0.188021
   us_financials RelViewBL__gpt_oss_20b           0.091102 0.722240     -0.181288
cross_asset_etfs                     BL           0.209090 1.785184     -0.090032
cross_asset_etfs                    MVO           0.066124 0.578467     -0.144855
cross_asset_etfs                     EW           0.126747 1.729588     -0.054472
cross_asset_etfs   BLM_LLM__gpt_oss_20b           0.172005 1.349211     -0.120341
cross_asset_etfs RelViewBL__gpt_oss_20b           0.111893 1.003006     -0.118416
```

### Equal-dataset aggregate

```text
                Method  cumulative_return   sharpe  max_drawdown
                    BL           0.254618 1.688426     -0.110593
  BLM_LLM__gpt_oss_20b           0.249326 1.693200     -0.112005
                    EW           0.137404 1.571659     -0.088512
                   MVO           0.210785 1.439429     -0.138417
RelViewBL__gpt_oss_20b           0.257699 1.656719     -0.131745
```

## Interpretation guardrails

The LLM calls were made after the 2026 test window, with recognizable asset names. Therefore the LLM methods can be affected by training-cutoff or provider-side temporal leakage even though the numeric pipeline is point-in-time. Results are research outputs, not evidence of deployable live performance.
