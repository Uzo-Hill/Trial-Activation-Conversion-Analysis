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

---
1.5 Engagement Segmentation

Organisations were segmented by number of distinct activities used during the trial:

*![engagement Seg](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/level_trial_engagement.png)*

- Conversion rates are narrow across all segments (19–25%), suggesting that feature breadth alone does not strongly drive conversion decisions.
- Power users (organizations with 8+ activities) shows the highest conversion rate **(~25%)**, suggesting strong product interaction drives value realization.
- Low engagement still converts **(~20%)**, implying that some organisations may convert with minimal usage.

---
1.6 Trial Goal Definition

Based on the conversion driver analysis, five trial goals were defined representing the platform's core value pillars. Goals were selected using three signals: statistical association, odds ratio strength, and product-value workflow logic.

| Goal          | Definition                          | Completion Rate   |
| --------------- | ------------------------------------ |----------------|
| goal_scheduling_core | Created ≥ 3 shifts    | 58% |
| goal_schedule_viewed   | Viewed schedule on mobile ≥ 1 time | 47% |
| goal_punchclock_used       | Clocked in or out at least once| 22% |
| goal_timesheet_approved       | Approved ≥ 1 shift or bulk timesheet| 21% |
| goal_team_comms    | Sent ≥ 1 team communication | 15% |


*![goal completion](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/orgs_completed_trial_goal.png)*

Key Findings:

- Completion rates are nearly identical between converters and non-converters across all five goals, reinforcing the earlier finding that no single activity or goal is a strong standalone predictor of conversion


- Scheduling Core and Schedule Viewed are the most completed goals, **58%** and **47%** of all organisations respectively, confirming that shift creation and mobile schedule viewing are the most natural entry points into the platform.

- Team Communication is the least completed goal **(15%)** and is actually slightly higher among non-converters **(16%)** than converters **(12%)**, suggesting that sending messages during the trial does not drive conversion and may not belong in the activation definition without further validation

---
1.7 Trial Activation Results

An organisation achieves Trial Activation when it completes all 5 goals

*![Activation milestone](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/activation_milestone.png)*

- Fully Activated organisations have the lowest conversion rate (16%). Surprisingly, completing all 5 goals does not lead to the highest conversion, suggesting the current activation definition may not be well-calibrated to predict conversion

- Partially activated organisations (≥3 goals) have the highest conversion rate (27%) — indicating that a 3-goal threshold may be a more meaningful activation benchmark than requiring all 5 goals

- Even organisations with no activation convert at 25% — this challenges the entire activation framework and suggests that conversion at Splendor Analytics may be driven by factors outside of in-app behaviour, such as sales outreach, pricing, or organisational decision-making timelines.

---

## Task 2 : SQL Data Models

The SQL models follow a three-layer data warehouse architecture:
```
┌──────────────────────────────────────────────────────────┐
│  STAGING LAYER                                           │
│  stg_trial_events  (102,895 rows)                        │
│  Raw cleaned event stream — one row per event            │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│  INTERMEDIATE LAYER                                      │
│  int_org_activity_summary  (966 rows)                    │
│  Pivoted activity counts — one row per organisation      │
└───────────┬──────────────────────────────┬───────────────┘
            │                              │
┌───────────▼────────────┐   ┌────────────▼───────────────┐
│  MARTS LAYER           │   │  MARTS LAYER               │
│  mart_trial_goals      │   │  mart_trial_activation     │
│  (966 rows)            │   │  (966 rows)                │
│  5 goal flags per org  │   │  Activation flag + tier    │
└────────────────────────┘   └────────────────────────────┘
```
---
The SQL models were built using **mysql-connector-python** rather than MySQL Workbench's import wizard, because the large dataset (102,895 rows) consistently failed or imported partially through the wizard. Python batch insertion was used instead:

```python
import mysql.connector

conn = mysql.connector.connect(
    host="localhost", user="root",
    password="***", database="splendor_analytics"
)
cursor = conn.cursor()

# Insert all rows in batches of 5,000
for i in range(0, len(rows), 5000):
    cursor.executemany(insert_query, rows[i:i+5000])
    conn.commit()
    print(f"Inserted rows {i+1} to {min(i+5000, len(rows))}")
```

- Staging Table — stg_trial_events

One row per event. Stores the cleaned dataset exactly as produced from Task 1.

```sql
CREATE TABLE stg_trial_events (
    organization_id    VARCHAR(50),
    activity_name      VARCHAR(100),
    timestamp          VARCHAR(25),
    converted          INT,           -- 0 or 1
    converted_at       VARCHAR(25),   -- NULL for non-converted orgs
    trial_start        VARCHAR(25),
    trial_end          VARCHAR(25),
    days_from_start    INT,
    trial_length       INT,
    before_conversion  INT,
    activity_canonical VARCHAR(100),
    activity_category  VARCHAR(50)
);
```

- Intermediate View — int_org_activity_summary

Collapses 102,895 event rows into 966 org-level rows by pivoting activity counts using SUM(activity_name = '...'):

```sql
CREATE VIEW int_org_activity_summary AS
SELECT
    organization_id,
    MAX(converted)                                          AS converted,
    COUNT(*)                                                AS total_events,
    COUNT(DISTINCT activity_name)                           AS distinct_activities,
    COUNT(DISTINCT DATE(timestamp))                         AS active_days,
    SUM(activity_name = 'Scheduling.Shift.Created')         AS cnt_shift_created,
    SUM(activity_name = 'Mobile.Schedule.Loaded')           AS cnt_schedule_loaded,
    SUM(activity_name = 'PunchClock.PunchedIn')             AS cnt_punched_in,
    -- ... all 28 activities
FROM stg_trial_events
GROUP BY organization_id;
```

- Mart Table 1 : mart_trial_goals
Grain: One row per organisation

Purpose: Tracks whether each trialist has completed each of the five trial goals

```python
cursor.execute("DROP TABLE IF EXISTS mart_trial_goals")
cursor.execute("""
    CREATE TABLE mart_trial_goals AS
    SELECT
        organization_id,
        converted,
        trial_start,
        trial_end,
        CASE WHEN cnt_shift_created >= 3
             THEN 1 ELSE 0 END                  AS goal_scheduling_core,
        CASE WHEN cnt_schedule_loaded >= 1
             THEN 1 ELSE 0 END                  AS goal_schedule_viewed,
        CASE WHEN (cnt_punched_in +
                   cnt_punched_out) >= 1
             THEN 1 ELSE 0 END                  AS goal_punchclock_used,
        CASE WHEN (cnt_shift_approved +
                   cnt_timesheet_bulk_approved) >= 1
             THEN 1 ELSE 0 END                  AS goal_timesheet_approved,
        CASE WHEN cnt_message_created >= 1
             THEN 1 ELSE 0 END                  AS goal_team_comms,
        cnt_shift_created,
        cnt_schedule_loaded,
        cnt_punched_in,
        cnt_punched_out,
        cnt_shift_approved,
        cnt_timesheet_bulk_approved,
        cnt_message_created,
        total_events,
        distinct_activities,
        active_days
    FROM int_org_activity_summary
""")
conn.commit()

cursor.execute("SELECT COUNT(*) FROM mart_trial_goals")
```
- Mart Table 2 : mart_trial_activation

```
cursor.execute("DROP TABLE IF EXISTS mart_trial_activation")
cursor.execute("""
    CREATE TABLE mart_trial_activation AS
    SELECT
        organization_id,
        converted,
        trial_start,
        trial_end,
        goal_scheduling_core,
        goal_schedule_viewed,
        goal_punchclock_used,
        goal_timesheet_approved,
        goal_team_comms,
        (goal_scheduling_core    +
         goal_schedule_viewed    +
         goal_punchclock_used    +
         goal_timesheet_approved +
         goal_team_comms)                        AS goals_completed,
        CASE WHEN (goal_scheduling_core    +
                   goal_schedule_viewed    +
                   goal_punchclock_used    +
                   goal_timesheet_approved +
                   goal_team_comms) = 5
             THEN 1 ELSE 0 END                   AS is_activated,
        CASE
            WHEN (goal_scheduling_core    +
                  goal_schedule_viewed    +
                  goal_punchclock_used    +
                  goal_timesheet_approved +
                  goal_team_comms) = 5    THEN 'Fully Activated'
            WHEN (goal_scheduling_core    +
                  goal_schedule_viewed    +
                  goal_punchclock_used    +
                  goal_timesheet_approved +
                  goal_team_comms) >= 3   THEN 'Partially Activated (>=3 goals)'
            WHEN (goal_scheduling_core    +
                  goal_schedule_viewed    +
                  goal_punchclock_used    +
                  goal_timesheet_approved +
                  goal_team_comms) >= 1   THEN 'Early Exploration (1-2 goals)'
            ELSE                               'No Activation'
        END                                      AS activation_tier,
        CASE WHEN converted = 1
              AND (goal_scheduling_core    +
                   goal_schedule_viewed    +
                   goal_punchclock_used    +
                   goal_timesheet_approved +
                   goal_team_comms) = 0
             THEN 1 ELSE 0 END                   AS converted_without_activation,
        CASE WHEN (goal_scheduling_core    +
                   goal_schedule_viewed    +
                   goal_punchclock_used    +
                   goal_timesheet_approved +
                   goal_team_comms) = 5
              AND converted = 0
             THEN 1 ELSE 0 END                   AS activated_without_converting
    FROM mart_trial_goals
""")
conn.commit()

cursor.execute("SELECT COUNT(*) FROM mart_trial_activation")
```
---
### EER Diagram

*![EER](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/EER_Diagram.PNG)*

See the SQL file in the repo for full SQL code including 8 business insight queries

---

## Task 3 — Descriptive Analytics & Product Metrics

- Key Product Metrics

*![product metrics](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/product_metrics.png)*

---

- Feature Adoption Rate

*![feature adoption](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/feature_adoption.png)*

- Scheduling is by far the most adopted feature.

- Payroll and Revenue features are severely underutilised, representing a significant onboarding gap.
---

- Monthly Cohort Performance
 
*![monthly](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/monthly_trial_volume.png)*


---

- Retention Curve

*![engagement drop](https://github.com/Uzo-Hill/Trial-Activation-Conversion-Analysis/blob/main/project_image/Engagement_drop.PNG)*

- Engagement drops sharply after Day 1 — only **38.9%** of organisations show any activity beyond their first day.

- By Day 7, this falls to approximately 28%, and continues declining gradually through to Day 30.

---

### Actionable recommendations

1. Only 39% of organizations have any activity after Day 1. An automated Day-2 nudge targeting inactive organizations could significantly improve trial engagement.

2. Scheduling is the most adopted module  and the strongest engagement signal. Onboarding should guide organizations to create their first 3 shifts quickly.

3. Fully activated organizations (all 5 goals) have the LOWEST conversion rate at 16.2%. Consider a 3-goal threshold as a more predictive activation benchmark.

4. Median conversion happens at Day 30 (trial expiry). A Day-25 prompt reminding organizations of trial expiry could accelerate conversion decisions.

5. Target high-intent non-converters with direct sales outreach.

---

### Limitations
- Conversion may be influenced by external factors (sales, pricing, timing)

- Event data does not capture user intent directly

-Trial goals are hypotheses, not guaranteed drivers

---
### Conclusion
This analysis shows that:

- Engagement matters, but quality > quantity

- Current activation definition is not predictive

- Product improvements should focus on:
    - Early engagement
    - Core feature adoption
    - Better activation metrics


---
## Acknowledgement

This project was completed as part of the Splendor Analytics Data Analyst Community Challenge. The dataset and problem statement were provided by Splendor Analytics.


---
Solution by :

Uzoh C. Hillary

Data Scientist / Data Analyst












