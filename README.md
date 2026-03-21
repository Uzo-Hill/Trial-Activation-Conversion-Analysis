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

---
##  Tech Stack & Tools Used

- Python 3.9+ - Primary programming language
- MySQL - Data model
- Jupyterlab - Interactive development and analysis
- **Statistics, Data Processing & Analysis** - Pandas, Numpy, Chi Square.
- **Machine Learning**: Scikit-learn, Machine learning models - Logistic Regression, RandomForestRegressor
- Matplotlib, Seaborn - visualization

---
## Data Cleaning & Preparation










