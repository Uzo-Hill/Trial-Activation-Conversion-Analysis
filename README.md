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

The Raw Data Preview
*![raw data](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/raw_data.PNG)*

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

1.1 Data Cleaning

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
See the [python notebook](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/Splendor%20Data%20Challenge%20-%20Trial%20Activation.ipynb) for full cleaning code
---

1.2 Activity Name Transformation

The raw activity_name column was transformed in three stages:
- Canonical Name Mapping — Each raw dot-notation activity was mapped to a clean, human-readable label. For example:
```python
  activity_canonical = {
    "Scheduling.Template.ApplyModal.Applied" : "Template Applied",
    "PunchClock.PunchedIn"                   : "Punched In/Out",
    "PunchClock.PunchedOut"                  : "Punched In/Out",   # merged pair
    ...
}
df["activity_canonical"] = df["activity_name"].map(activity_canonical)
```
- Paired Activity Merging — Activities described as two sides of the same action were merged under a single canonical name:

| Raw Values         |Canonical Name                     |
| ---------------------| ------------------------------------ |
| PunchClock.PunchedIn + PunchedOut      | Punched In/Out       
| Break.Activate.Started + Finished        | Break Started/Finished            |
| Absence.Request.Approved + Rejected         | Absence Approved/Rejected                   |
| Shift.View.Opened + ShiftDetails.View.Opened           |Shift Details Viewed       |

- Activity Categorisation — A corrected activity_category column maps all activities to six product pillars:

```python
activity_map = {
    "Scheduling":          "Scheduling",
    "Mobile":              "Scheduling",        # mobile schedule = scheduling feature
    "Shift":               "Scheduling",
    "ShiftDetails":        "Scheduling",
    "PunchClock":          "Time Tracking",
    "PunchClockStartNote": "Time Tracking",     # corrected grouping
    "PunchClockEndNote":   "Time Tracking",     # corrected grouping
    "Break":               "Time Tracking",
    "Timesheets":          "Payroll",
    "Integration":         "Payroll",
    "Absence":             "Absence",
    "Revenue":             "Revenue & Budgeting",
    "Communication":       "Communication",
}
df["activity_category"] = df["activity_name"].str.split(".").str[0].map(activity_map)
```
---
1.3 Exploratory Data Analysis

- What percentage of organisations convert?

*![conversion rate](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/conversion_rate.png)*

21.3% of the 966 trialling organisations converted to paid — consistent with the product team's estimate of roughly 1 in 5.


- How long does it take organisations to convert?

*![time to convert](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/time_to_convert.png)*


- Which features are most used during trials?

*![Mostfeature](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/most_frequent_trial_activities.png)*


- Daily activity timeline

*![activity timeline](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/engagement_trend.png)*

---

1.4 Conversion Driver Analysis

Three complementary methods were applied to identify which in-app behaviours are associated with conversion:

- Method 1 : Chi-Square Test

Tests whether the usage of each activity is statistically associated with conversion.

```python
from scipy.stats import chi2_contingency

chi_results = []
for act in activity_cols:
    contingency = pd.crosstab(orgs[act] > 0, orgs["converted"])
    chi2, p, _, _ = chi2_contingency(contingency)
    chi_results.append({"activity": act, "chi2": chi2, "p_value": p})
```
Chi-Square Tests — Activity Usage vs Conversion

*![Chi2 test](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/Chi2_test.PNG)*

Key Findings:
- No activity is statistically significant. All p-values are above **0.05**, meaning we cannot conclude that the usage of any single activity is strongly associated with conversion

- Communication.Message.Created has the highest chi-square score (1.99) but still falls short of significance, suggesting it has the weakest association with conversion among all activities

- This is an honest null finding. Conversion is likely driven by the volume and breadth of engagement across multiple activities rather than any one specific feature trigger.

---
- Method 2 : Random Forest Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(X, y)
# CV ROC-AUC: 0.533
```
*![feature importance](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/feature_importance_viz.PNG)*

Key Findings:
- Overall engagement volume matters most — total_events and active_days are the top predictors, confirming that how much an organisation engages matters more than which specific feature they use

- Scheduling is the dominant activity — three of the top ten features are scheduling activities (Shift.Created, AssignmentChanged, Shift.Approved), making it the clearest behavioural signal during the trial

- The model's ROC-AUC of 0.533 is barely above chance (0.5), indicating that individual activity usage alone is a weak predictor of conversion, consistent with the chi-square null findings.
---
- Method 3 : Logistic Regression Odds Ratios

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_scaled, y)
# CV ROC-AUC: 0.513
```

*![odd ratio](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/odd_ratio_bar.PNG)*

Key Findings:
- Scheduling.ShiftHandover.Created has the highest odds ratio (1.37), meaning organisations that requested shift handovers were *37%* more likely to convert, suggesting deeper scheduling workflow adoption is a positive conversion signal

- All odds ratios are close to 1.0, indicating that no single activity dramatically increases or decreases the probability of conversion, reinforcing the weak individual-feature findings from the previous two methods

- Timesheets.BulkApprove.Confirmed has the lowest odds ratio (0.59), which is counterintuitive, organisations that bulk-approved timesheets were actually less likely to convert, possibly because they completed the trial workflow without feeling the need to pay for continued access.














