# Trial-Activation-Conversion-Analysis
Trial Activation &amp; Conversion Analysis for a Workforce Management SaaS Platform

---
## Project Overview

Splendor Analytics offers a 30-day free trial for organisations using its workforce management platform. However, the company lacks clarity on what defines a “successful” trial experience and which user behaviors drive conversion to paid plans.

This project analyzes behavioral event data to:

- Identify key engagement patterns

- Define trial activation criteria

- Build data models for scalable tracking

- Generate product insights for decision-making

---
## Project Objectives
1. Clean and prepare raw event data for analysis

2. Identify behaviors that correlate with conversion

3. Define measurable Trial Goals (Activation Criteria)

4. Build SQL models:

    - Trial Goals Table

    - Trial Activation Table

5. Generate product insights:

    - Conversion drivers

    - Engagement patterns

    - Feature adoption

---
## Dataset Description

The dataset contains raw behavioural events from organisations that started their trial between January and March 2024.

Raw dataset stats:
- 170,526 total rows
- 966 unique organisations
- 28 unique activity types

| Column          | Description                          |
| --------------- | ------------------------------------ |
| organization_id | Unique organisation identifier       |
| activity_name   | Product action performed             |
| timestamp       | Event timestamp                      |
| converted       | Whether organisation converted       |
| converted_at    | Conversion timestamp (if applicable) |
| trial_start     | Trial start date                     |
| trial_end       | Trial end date                       |


[ insert raw data screenshot ]
---
##  Setup & Installation
Prerequisites
- Python 3.8+ (Pandas, NumPy, Seaborn, Matplotlib)
- MySQL 8.0+ (Data Modeling)
- Jupyter Notebook

---
## Project Structure
```
├── data/
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── eda_analysis.ipynb
│   └── product_metrics.ipynb
├── sql/
│   ├── trial_goals.sql
│   └── trial_activation.sql
├── requirements.txt
└── README.md

```
---
## Task 1 — Data Cleaning, Transformation, EDA & Conversion Driver Analysis

The raw dataset required several cleaning steps and transformation before analysis:

| Data Issue           |Action Taken                       |
| ---------------------| ------------------------------------ |
| UPPERCASE column names      | Normalised to lowercase       
| 67,631 exact duplicate rows (~40%)        | Removed duplicates on org + activity + timestamp             |
| Datetime columns stored as strings          | Parsed with pd.to_datetime()                     |
| 136,291 null values in converted_at field           |Validated as expected. Non-converted orgs have no conversion date       |
| Events outside trial window         | Filtered with ±1 minute tolerance |
| Created derived features         | days_from_start, trial_length, before_conversion                     |

After cleaning: 102,895 rows remain across 966 organisations.

```python
# Remove exact duplicate rows
df = df.drop_duplicates(subset=["organization_id", "activity_name", "timestamp"])

# Filter events within trial window
df = df[(df["timestamp"] >= df["trial_start"] - pd.Timedelta("1min")) &
        (df["timestamp"] <= df["trial_end"]   + pd.Timedelta("1min"))]

# Create derived features:
# Days since trial start
df['days_from_start'] = (df['timestamp'] - df['trial_start']).dt.days

# Trial duration
df['trial_length'] = (df['trial_end'] - df['trial_start']).dt.days

# Before/after conversion
df['before_conversion'] = df['timestamp'] <= df['converted_at']
```
📎 See the [python notebook](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/Splendor%20Data%20Challenge%20-%20Trial%20Activation.ipynb) for full cleaning code


