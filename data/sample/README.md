# 示例数据包 / Sample Data Package

固定种子生成的示例输入数据（`seed=42/3/5/9`），用于在运行仿真前查看数据形态与字段含义。
Generated with fixed seeds so you can inspect the input shapes before running the pipeline.

## advertisers_sample.csv

| 字段 | 说明 |
|---|---|
| advertiser_id | 广告主 ID |
| name | 名称 |
| category / category_idx | 品类（8 类） |
| tier | S / M / L（中小/中型/大型） |
| daily_budget | 日预算 |
| initial_bid | 初始出价 |
| quality | 质量分 |
| audience_tags | 定向人群标签（`\|` 分隔） |
| creatives | 广告文案（`\|` 分隔） |
| objective | 投放目标 |
| value_per_conversion | 单转化价值 |
| base_ctr | 基础点击率 |

## creative_corpus_sample.csv

`advertiser_id`、`creative_index`、`text`：每条广告文案一行。

## operation_logs_sample.csv

`advertiser_id`、`event_type`（raise_budget / raise_bid / change_creative / expand_targeting / exit）、`magnitude`、`day`、`text`（自由文本备注）。

## feedback_corpus_sample.csv

`id`、`kind`（user_complaint / advertiser_ticket / audit_record / experiment_review / industry_news）、`opportunity`（对应策略机会）、`text`。

## behavior_sample.csv

行为特征基线：`pre_spend`、`pre_conversions`、`pre_ctr`、`budget`、`quality`、`category`、`tier_S`、`tier_L`、`y_click_next`（点击标签，用于预测消融）。可用 Criteo 数据替换（见 README）。

## 加载 / Loading

```python
from semadexp.data.sample import load_sample
data = load_sample()  # -> dict[str, pd.DataFrame]
```

重新生成：`python scripts/export_sample_data.py`。

