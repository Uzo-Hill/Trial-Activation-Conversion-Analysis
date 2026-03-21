#!/usr/bin/env python
# coding: utf-8

# ### Defining Trial Activation & Conversion Drivers for a Workforce Management SaaS Platform

# ### Introduction

# Splendor Analytics offers a 30-day free trial for organisations using its workforce management platform. However, the company lacks clarity on what defines a “successful” trial experience and which user behaviors drive conversion to paid plans.

# This project analyzes behavioral event data to:
# 
# - Identify key engagement patterns
# 
# - Define trial activation criteria
# 
# - Build data models for scalable tracking
# 
# - Generate product insights for decision-making

# ### Project Objectives

# 1. Clean and prepare raw event data for analysis
# 
# 2. Identify behaviors that correlate with conversion
# 
# 3. Define measurable Trial Goals (Activation Criteria)
# 
# 4. Build SQL models:
# 
#     - Trial Goals Table
# 
#     - Trial Activation Table
# 
# 5. Generate product insights:
# 
#     - Conversion drivers
# 
#     - Engagement patterns
# 
#     - Feature adoption

# In[ ]:





# ### Importing Libraries and Loading the Raw CSV dataset of behavioural events

# In[3]:


import subprocess
subprocess.run(["pip", "install", "scikit-learn", "scipy", "statsmodels", "-q"], check=True)


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# In[2]:


# Loading the raw dataset and initial inspection

df = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Splendor Data Challenge\DA task.csv")


# In[4]:


df.head()


# In[5]:


print(f"df shape     : {df.shape}")
print(f"Columns       : {df.columns.tolist()}")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")


# - The dataset is event-heavy but highly duplicated (67,631 duplicates ≈ ~40%), meaning deduplication is critical before any analysis to avoid biased results.
# 
# - A large portion of CONVERTED_AT is missing (136,291 rows), which is expected for non-converted organizations, but must be handled carefully when creating conversion-related features.
# 
# - All key columns are present with no missing IDs, timestamps, or activity names, indicating good structural integrity, but still requires validation (e.g., date consistency within trial period).

# In[6]:


df.info()


# In[ ]:





# ### Task 1: Data Cleaning, EDA & Conversion Driver Analysis

# Normalize column names

# In[7]:


# Normalise column names

df.columns = df.columns.str.lower()

df.head(2)


# In[ ]:





# Parse datetime columns

# In[8]:


# Fix data type:

for col in ["timestamp", "converted_at", "trial_start", "trial_end"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")


# In[9]:


df.head(2)


# In[ ]:





# ### Remove exact duplicate rows (same org + activity + timestamp)
# 

# In[10]:


# Remove exact duplicate rows (same org + activity + timestamp)

df = df.drop_duplicates(subset=["organization_id", "activity_name", "timestamp"])

print(f"After dedup   : {df.shape}")


# In[ ]:





# ### Drop events outside trial window (± 1 min tolerance)

# In[11]:


# Drop events outside trial window (± 1 min tolerance)

mask_in_window = (
    (df["timestamp"] >= df["trial_start"] - pd.Timedelta("1min")) &
    (df["timestamp"] <= df["trial_end"]   + pd.Timedelta("1min"))
)
df = df[mask_in_window]
print(f"After window  : {df.shape[0]} rows")


# In[ ]:





# In[12]:


# Validate converted_at logic
# converted=True must have a non-null converted_at inside the trial window

bad_converted = (df["converted"] == True) & (df["converted_at"].isna())
print(f"Converted=True with null converted_at: {bad_converted.sum()}")


# The missing values in converted_at are expected and represent organisations that did not convert during the trial period. Instead of imputing or removing these values, they were preserved to maintain the integrity of conversion analysis. 

# ### Create Derived Features

# In[13]:


# Days since trial start
df['days_from_start'] = (df['timestamp'] - df['trial_start']).dt.days

# Trial duration
df['trial_length'] = (df['trial_end'] - df['trial_start']).dt.days

# Before/after conversion
df['before_conversion'] = df['timestamp'] <= df['converted_at']


# In[14]:


df.head(2)


# In[ ]:





# In[15]:


# Checking Unique values in the activity_name column

df['activity_name'].unique()


# In[ ]:





# The **activity_name** column contains granular event-level interactions performed by organisations during their trial period. While detailed, these raw event names are inconsistent in structure and overly fragmented, making direct analysis noisy and less interpretable.
# 
# To address this, the column will be cleaned and transformed in three stages:
# 
# 
# - Normalization: Merge similar or duplicate actions (e.g., different “view” or “decision” events) into unified labels.
# 
# - Feature Transformation: Map activities into higher-level product categories (e.g., scheduling, payroll, communication) and action types (e.g., create, approve, view) to better reflect user behavior and product usage.
# 
# This transformation reduces dimensionality, improves interpretability, and enables more meaningful analysis of engagement patterns and conversion drivers.

# In[ ]:





# ### Adding a Canonical Name Column

# In[16]:


# Adding a Canonical Name Column
# some activities listed as a single entry covering two event names(same action)
# are grouped under one canonical name for analysis.

# Map each raw activity to the clean label from the activity name description list

activity_canonical = {
    # Scheduling
    "Scheduling.Availability.Set":               "Set Availability",
    "Scheduling.Shift.Created":                  "Shift Created",
    "Scheduling.Shift.AssignmentChanged":        "Shift Assignment Changed",
    "Scheduling.Template.ApplyModal.Applied":    "Template Applied",
    "Scheduling.ShiftSwap.Created":              "Shift Swap Requested",
    "Scheduling.ShiftSwap.Accepted":             "Shift Swap Accepted",
    "Scheduling.ShiftHandover.Created":          "Shift Handover Requested",
    "Scheduling.ShiftHandover.Accepted":         "Shift Handover Accepted",
    "Scheduling.OpenShiftRequest.Created":       "Open Shift Requested",
    "Scheduling.OpenShiftRequest.Approved":      "Open Shift Approved",
    "Scheduling.Shift.Approved":                 "Shift Approved",

    # Viewing
    "Mobile.Schedule.Loaded":                    "Schedule Viewed",
    "Shift.View.Opened":                         "Shift Details Viewed",
    "ShiftDetails.View.Opened":                  "Shift Details Viewed",   # ← merged

    # Absence
    "Absence.Request.Created":                   "Absence Requested",
    "Absence.Request.Approved":                  "Absence Approved/Rejected",
    "Absence.Request.Rejected":                  "Absence Approved/Rejected", # ← merged

    # PunchClock
    "PunchClock.PunchedIn":                      "Punched In/Out",  # ← merged
    "PunchClock.PunchedOut":                     "Punched In/Out",  # ← merged
    "Break.Activate.Started":                    "Break Started/Finished",  # ← merged
    "Break.Activate.Finished":                   "Break Started/Finished",  # ← merged
    "PunchClockStartNote.Add.Completed":         "Punch-In Note Added",
    "PunchClockEndNote.Add.Completed":           "Punch-Out Note Added",
    "PunchClock.Entry.Edited":                   "Clock Entry Edited",

    # Approval & Payroll
    "Timesheets.BulkApprove.Confirmed":          "Timesheets Bulk Approved",
    "Integration.Xero.PayrollExport.Synced":     "Payroll Synced (Xero)",
    "Revenue.Budgets.Created":                   "Budget Created",

    # Communication
    "Communication.Message.Created":             "Message Sent",
}

df["activity_canonical"] = df["activity_name"].map(activity_canonical)


# In[17]:


df['activity_canonical'].unique()


# In[18]:


df.head(2)


# ### Create Activity Categories 

# We'll create a lookup dictionary that maps each raw activity prefix to a clean product category. It acts as a translation table, the keys are the raw prefixes, the values are the category labels we want to assign.

# In[19]:


categorize_activity = {
    "Scheduling":          "Scheduling",
    "Mobile":              "Scheduling",        # mobile schedule view = scheduling feature
    "Shift":               "Scheduling",        # Shift.View.Opened
    "ShiftDetails":        "Scheduling",        # ShiftDetails.View.Opened
    "Absence":             "Absence",
    "PunchClock":          "Time Tracking",
    "PunchClockStartNote": "Time Tracking",     # ← fix grouping
    "PunchClockEndNote":   "Time Tracking",     # ← fix grouping
    "Break":               "Time Tracking",
    "Timesheets":          "Payroll",
    "Integration":         "Payroll",
    "Revenue":             "Revenue & Budgeting",   # ← its own group
    "Communication":       "Communication",
}

df["activity_category"] = df["activity_name"].str.split(".").str[0].map(categorize_activity)


# In[20]:


# Confirm the activity_category

df.head()


# In[21]:


import os

# Create the output directory if it doesn't exist
os.makedirs("splendor_analytics/outputs", exist_ok=True)


# In[22]:


print(f"\nFinal clean shape: {df.shape}")
print(f"Unique orgs      : {df['organization_id'].nunique()}")
print(f"Unique activities: {df['activity_name'].nunique()}")


# In[ ]:





# In[23]:


df.to_csv("splendor_analytics/outputs/clean_events.csv", index=False)
print("Saved → splendor_analytics/outputs/clean_events.csv")


# In[ ]:





# ### Organization Level Feature Table

# In[24]:


# One row per organization with activity counts + metadata
org_meta = (
    df.drop_duplicates("organization_id")
    [["organization_id", "converted", "converted_at", "trial_start", "trial_end", "trial_length"]]
)

# Pivot: count of each activity per organization
activity_counts = (
    df.groupby(["organization_id", "activity_name"])
    .size()
    .unstack(fill_value=0)
)

# Aggregate engagement metrics
agg = df.groupby("organization_id").agg(
    total_events        = ("activity_name", "count"),
    distinct_activities = ("activity_name", "nunique"),
    distinct_category   = ("activity_category", "nunique"),
    active_days         = ("days_from_start", lambda x: x.astype(int).nunique()),
    first_event_day     = ("days_from_start", "min"),
    last_event_day      = ("days_from_start", "max"),
).reset_index()




orgs = org_meta.merge(agg, on="organization_id").merge(activity_counts, on="organization_id")

# Time-to-convert (days from trial_start)
orgs["days_to_convert"] = (orgs["converted_at"] - orgs["trial_start"]).dt.total_seconds() / 86400

print(f"Org feature table: {orgs.shape}")


# In[25]:


orgs.to_csv("splendor_analytics/outputs/org_features.csv", index=False)


# In[26]:


orgs


# Organisation-Level Feature Table
# 
# The raw event log contains one row per activity event, which is not suitable for conversion analysis since the target variable (converted) is measured at the organisation level. The data is therefore reshaped so that each organisation is represented by a single row.
# 
# Key points:
# 
# - One row per organisation (966 rows, collapsed from 102,895 events)
# 
# - Activity counts: Each activity is pivoted into its own column showing how many times each organisation performed it
# 
# - Engagement metrics: Derived fields like total_events, distinct_activities, and active_days capture trial engagement depth
# 
# - Foundation: All downstream analyses such as conversion driver analysis, trial goal definition, and product metrics will be built on top of this table
# 

# In[ ]:





# In[27]:


# Styling

PURPLE   = "#6C3FC8"
LAVENDER = "#A78BFA"
MINT     = "#34D399"
CORAL    = "#F87171"
AMBER    = "#FBBF24"
DARK     = "#1E1B2E"
LIGHT    = "#F3F0FF"
GRAY     = "#6B7280"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   DARK,
    "axes.edgecolor":   "#3D3557",
    "axes.labelcolor":  "white",
    "xtick.color":      GRAY,
    "ytick.color":      GRAY,
    "text.color":       "white",
    "grid.color":       "#2D2A45",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "figure.dpi":       130,
})


# ### Exploratory Data Analysis (EDA)

# 1. Conversion Rate
# 
# What percentage of trial organisations convert to paid customers?

# In[28]:


# What percentage of trial organisations convert to paid customers? 

# Get one row per organisation
org_conversion = df.drop_duplicates("organization_id")[["organization_id", "converted"]]

# Calculate counts and percentages
total     = len(org_conversion)
converted = org_conversion["converted"].sum()
not_conv  = total - converted
conv_pct  = converted / total * 100
not_pct   = not_conv  / total * 100

# Plot 
fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK)
ax.set_facecolor(DARK)

sizes  = [conv_pct, not_pct]
labels = ["Converted", "Not Converted"]
colors = [MINT, CORAL]

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": DARK, "linewidth": 2},
    textprops={"color": "white", "fontsize": 13}
)

for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight("bold")

# Add counts as annotation in the centre
ax.text(0, 0, f"{total}\nOrgs", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")

ax.set_title("What percentage of trial organisations\nconvert to paid customers?",
             fontsize=14, fontweight="bold", color="white", pad=20)

# Add a text summary below the chart
fig.text(0.5, 0.02,
         f"{converted} converted  |  {not_conv} did not convert  |  {total} total organisations",
         ha="center", fontsize=11, color=GRAY)

plt.tight_layout()
plt.show()


# In[ ]:





# ### 2. How long does it take organisations to convert?

# In[29]:


# How long does it take organisations to convert? 

# Get days_to_convert for converted orgs only
ttc = orgs[orgs["converted"] == True]["days_to_convert"].dropna()

median_days = ttc.median()
mean_days   = ttc.mean()

# Plot
fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
ax.set_facecolor(DARK)

ax.hist(ttc, bins=25, color=LAVENDER, edgecolor=DARK, alpha=0.85)

# Reference lines
ax.axvline(median_days, color=MINT,  linestyle="--", linewidth=1.8,
           label=f"Median: {median_days:.0f} days")
ax.axvline(mean_days,   color=AMBER, linestyle=":",  linewidth=1.8,
           label=f"Mean: {mean_days:.0f} days")

ax.set_title("How long does it take organisations to convert?",
             fontsize=14, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Days from Trial Start to Conversion", fontsize=11)
ax.set_ylabel("Number of Organisations",             fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, axis="y")

plt.tight_layout()
plt.show()


# In[ ]:





# ### 3. Which activities are most commonly performed during the trial?

# In[30]:


# Which activities are most commonly performed during the trial?

# Count total events per activity across all orgs
activity_counts = (
    df.groupby("activity_canonical")
    .size()
    .reset_index(name="event_count")
    .sort_values("event_count", ascending=True)
)

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK)
ax.set_facecolor(DARK)

bars = ax.barh(
    activity_counts["activity_canonical"],
    activity_counts["event_count"],
    color=LAVENDER, edgecolor=DARK, alpha=0.85
)

# Add value labels on each bar
for bar in bars:
    width = bar.get_width()
    ax.text(width + 200, bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}", va="center", fontsize=9, color="white")

ax.set_title("Which activities are most commonly performed during the trial?",
             fontsize=14, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Total Event Count", fontsize=11)
ax.set_ylabel("Activity",          fontsize=11)
ax.grid(True, axis="x")
ax.set_xlim(0, activity_counts["event_count"].max() * 1.15)

plt.tight_layout()
plt.show()


# In[ ]:





# ### How does trial engagement change day by day?

# In[31]:


# How does trial engagement change day by day?

# Count events per day into trial
daily_activity = (
    df.groupby("days_from_start")
    .size()
    .reset_index(name="event_count")
)

# ── Plot ──
fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
ax.set_facecolor(DARK)

ax.fill_between(daily_activity["days_from_start"],
                daily_activity["event_count"],
                alpha=0.3, color=LAVENDER)
ax.plot(daily_activity["days_from_start"],
        daily_activity["event_count"],
        color=LAVENDER, linewidth=2)

ax.set_title("How does trial engagement change day by day?",
             fontsize=14, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Day into Trial", fontsize=11)
ax.set_ylabel("Total Events",   fontsize=11)
ax.grid(True, axis="y")

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### CONVERSION DRIVER ANALYSIS — CORE TO PROJECT TASK

# To understand what separates converting organisations from non-converting ones, three complementary methods are applied:
# 
# 1. Chi-Square Test — tests whether the usage of each activity is statistically associated with conversion
# 
# 2. Random Forest — ranks activities by their importance in predicting conversion
# 
# 3. Logistic Regression — estimates the odds of conversion associated with each activity
# 
# Together, these methods provide a statistically grounded view of which in-app behaviours are most indicative of conversion.

# In[32]:


# METHOD 1: Statistical Tests (Chi-Square)

print("\n[Method 1] Chi-Square Tests — Activity Usage vs Conversion")

# Define activity columns — all pivoted activity columns in the org-level table
activity_cols = [c for c in orgs.columns if "." in c]

chi_results = []
for act in activity_cols:
    if act not in orgs.columns:
        continue

    # Create a 2x2 Frequency Table (Contingency Table)
    # Rows: Did they use the activity? (True/False)
    # Columns: Did they convert? (True/False)
    contingency = pd.crosstab(orgs[act] > 0, orgs["converted"])
    if contingency.shape == (2, 2):          # needs 2x2 for chi2
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        chi_results.append({"activity": act, "chi2": chi2, "p_value": p})

chi_df = pd.DataFrame(chi_results).sort_values("p_value")   # Sort by p_value (lowest p-values = highest statistical significance)
chi_df["significant"] = chi_df["p_value"] < 0.05            # Add a boolean column: True if the p-value is below the standard 0.05 threshold
print(chi_df.to_string(index=False))


# - No activity is statistically significant. All p-values are above **0.05**, meaning we cannot conclude that the usage of any single activity is strongly associated with conversion
#   
# - Communication.Message.Created has the highest chi-square score (1.99) but still falls short of significance, suggesting it has the weakest association with conversion among all activities
# 
# - This is an honest null finding — conversion is likely driven by the volume and breadth of engagement across multiple activities rather than any one specific feature trigger.

# In[ ]:





# In[33]:


# METHOD 2: Random Forest Feature Importance

# Method 2 uses Machine Learning to find which features have the most predictive power, 
#even if they have complex or non-linear relationships with conversion.

print("\n[Method 2] Random Forest Feature Importance")

feat_cols = [c for c in activity_cols if c in orgs.columns]
feat_cols += ["total_events", "distinct_activities", "distinct_category", "active_days"]

X = orgs[feat_cols].fillna(0)    # Prepare the Features (X): Replace any missing data with 0 (essential for ML models)
y = orgs["converted"].astype(int)  # Prepare the Target (y): Ensure 'converted' is an integer (0 or 1) for the classifier

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

rf.fit(X, y)     # Train the model on the full feature set


# Validate the model's reliability using 5-Fold Cross-Validation
# It splits the data 5 ways to ensure the model isn't just "getting lucky" on one slice
# scoring="roc_auc": Measures the model's ability to distinguish between classes (0.5 is random, 1.0 is perfect)
cv_score = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
print(f"RF CV ROC-AUC: {cv_score:.3f}")

importance_df = (
    pd.DataFrame({"feature": feat_cols, "importance": rf.feature_importances_})  # Extract "Feature Importance" — a score showing how much each variable helped the trees make decisions
    .sort_values("importance", ascending=False)
    .head(15)     # show top 15
)
print(importance_df.to_string(index=False))


# - Overall engagement volume matters most — total_events and active_days are the top predictors, confirming that how much an organisation engages matters more than which specific feature they use
# 
# - Scheduling is the dominant activity — three of the top ten features are scheduling activities (Shift.Created, AssignmentChanged, Shift.Approved), making it the clearest behavioural signal during the trial
# 
# - The model's ROC-AUC of 0.533 is barely above chance (0.5), indicating that individual activity usage alone is a weak predictor of conversion, consistent with the chi-square null findings.
# 

# In[ ]:





# In[34]:


# METHOD 3: Logistic Regression Odds Ratios
print("\n[Method 3] Logistic Regression — Odds Ratios")

# Use binary flags for activities to get interpretable ORs
lr_feats  = [c for c in activity_cols if c in orgs.columns]    # Selecting all activity columns available in the dataframe
X_bin     = (orgs[lr_feats].fillna(0) > 0).astype(int)
scaler    = StandardScaler()    # Feature Scaling (Standardization)
X_scaled  = scaler.fit_transform(X_bin)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)   # Initialize Logistic Regression
lr.fit(X_scaled, y)

lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc").mean()   # Validate model performance using Cross-Validation
print(f"LR CV ROC-AUC: {lr_cv:.3f}")

or_df = pd.DataFrame({
    "feature":  lr_feats,
    "coef":     lr.coef_[0],
    "odds_ratio": np.exp(lr.coef_[0])   # Calculate Odds Ratios (OR), easy to read
}).sort_values("odds_ratio", ascending=False)

print(or_df.to_string(index=False))    # Print the table ranked from most positive impact to most negative


# - Scheduling.ShiftHandover.Created has the highest odds ratio (1.37), meaning organisations that requested shift handovers were 37% more likely to convert, suggesting deeper scheduling workflow adoption is a positive conversion signal
# 
# - All odds ratios are close to 1.0, indicating that no single activity dramatically increases or decreases the probability of conversion, reinforcing the weak individual-feature findings from the previous two methods
# 
# - Timesheets.BulkApprove.Confirmed has the lowest odds ratio (0.59), which is counterintuitive, organisations that bulk-approved timesheets were actually less likely to convert, possibly because they completed the trial workflow without feeling the need to pay for continued access.

# In[ ]:





# ### ENGAGEMENT SEGMENTATION

# In[35]:


#  How does conversion rate vary by level of trial engagement? 

# Segment orgs by distinct activities using manual bins
bins   = [0, 1, 3, 7, 100]
labels = ["Low (1)", "Medium (2-3)", "High (4-7)", "Power (8+)"]

orgs["engagement_segment"] = pd.cut(
    orgs["distinct_activities"],
    bins=bins,
    labels=labels
)

# Summary per segment
seg_summary = (
    orgs.groupby("engagement_segment", observed=True)
    .agg(
        n_orgs          = ("organization_id", "count"),
        conversion_rate = ("converted", "mean"),
        avg_events      = ("total_events", "mean"),
        avg_active_days = ("active_days", "mean")
    )
    .reset_index()
)

print(seg_summary)


# In[ ]:





# In[36]:


#  Plot 
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle("How does conversion rate vary by level of trial engagement?",
             fontsize=14, fontweight="bold", color="white", y=1.02)

seg_colors = [CORAL, AMBER, LAVENDER, MINT]

# Panel 1: Conversion rate per segment 
axes[0].set_facecolor(DARK)
bars = axes[0].bar(
    seg_summary["engagement_segment"],
    seg_summary["conversion_rate"] * 100,
    color=seg_colors, edgecolor=DARK, alpha=0.88
)
axes[0].set_title("Conversion Rate by Segment", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Engagement Segment", fontsize=11)
axes[0].set_ylabel("Conversion Rate (%)",  fontsize=11)
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
axes[0].grid(True, axis="y")

for bar, (_, row) in zip(bars, seg_summary.iterrows()):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{row['conversion_rate']:.0%}\nn={row['n_orgs']}",
        ha="center", fontsize=10, color="white"
    )

#  Panel 2: Average events per segment 
axes[1].set_facecolor(DARK)
bars2 = axes[1].bar(
    seg_summary["engagement_segment"],
    seg_summary["avg_events"],
    color=seg_colors, edgecolor=DARK, alpha=0.88
)
axes[1].set_title("Average Events per Segment", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Engagement Segment", fontsize=11)
axes[1].set_ylabel("Avg Total Events",   fontsize=11)
axes[1].grid(True, axis="y")

for bar, (_, row) in zip(bars2, seg_summary.iterrows()):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{row['avg_events']:.0f}",
        ha="center", fontsize=10, color="white"
    )

plt.tight_layout()
plt.show()


# - Conversion generally increases with deeper engagement, with Power users (8+ activities) showing the highest conversion rate (~25%), suggesting strong product interaction drives value realization.
# 
# - Moderate engagement performs inconsistently, as High (4–7) users convert less (~19%) than Medium (2–3) users (~23%), indicating that not all activity types contribute equally to conversion.
# 
# - Low engagement still converts (~20%), implying that some organisations may convert with minimal usage, but scaling conversion likely depends on driving meaningful, not just frequent, engagement.

# In[ ]:





# ### DEFINE TRIAL GOALS  (data-driven, product-value grounded)

# Here, we are defining specific "Success Milestones" (Trial Goals) to see if reaching them predicts a customer actually paying for the product.

# """
# APPROACH:
# We combine three signals to select goals:
#   1. Statistically significant chi-square (p<0.05)
#   2. High odds ratio from logistic regression (OR > 1.5)
#   3. Product-value logic: activities that represent completing a workflow
#      end-to-end (scheduling → shift live → punch-in/out → timesheet approval)
#  
# The goals are intentionally achievable (completion rate 25-70%) so they
# discriminate between engaged and non-engaged orgs. 
# 
# We will define 5 goals covering the platform's core value pillars.
# """

# In[ ]:





# In[37]:


# =============================================================================
# DEFINE TRIAL GOALS (data-driven, product-value grounded)
# =============================================================================

# Core Goal: Creating at least 3 shifts (indicates they are actually using the scheduler)
# Visibility Goal: Did they check the schedule on mobile at least once?
# Feature Adoption: Did they try the PunchClock (In OR Out)?
# Admin Efficiency: Did they approve a shift or bulk-approve timesheets?
# Engagement: Did they send at least one team message?

TRIAL_GOALS = {
    "goal_scheduling_core":    {"activities": ["Scheduling.Shift.Created"],                                        "threshold": 3},
    "goal_schedule_viewed":    {"activities": ["Mobile.Schedule.Loaded"],                                          "threshold": 1},
    "goal_punchclock_used":    {"activities": ["PunchClock.PunchedIn", "PunchClock.PunchedOut"],                   "threshold": 1, "combine": "any"},
    "goal_timesheet_approved": {"activities": ["Scheduling.Shift.Approved", "Timesheets.BulkApprove.Confirmed"],   "threshold": 1, "combine": "any"},
    "goal_team_comms":         {"activities": ["Communication.Message.Created"],                                   "threshold": 1},
}

# Compute goal completion per org
goal_results = orgs[["organization_id", "converted"]].copy()

for goal, cfg in TRIAL_GOALS.items():
    acts = [a for a in cfg["activities"] if a in orgs.columns]   # Filter for activity columns that actually exist in the 'orgs' DataFrame
    if not acts:
        goal_results[goal] = False                         # If none of the required activities exist in the data, mark the goal as failed (False)
        continue
    if cfg.get("combine") == "any":
        # Org passes if ANY of the activities was used at least once
        goal_results[goal] = (orgs[acts] > 0).any(axis=1)
    else:
        # Sum of all listed activities must meet threshold
        goal_results[goal] = orgs[acts].sum(axis=1) >= cfg["threshold"]  

goal_cols = list(TRIAL_GOALS.keys())   # Storing the list of goal names for easy reference in future plotting or tables


# In[ ]:





# In[38]:


# Print completion rates overall and by conversion status
print("Goal completion rates:\n")
for g in goal_cols:
    overall = goal_results[g].mean()
    c_rate  = goal_results[goal_results["converted"] == True][g].mean()
    n_rate  = goal_results[goal_results["converted"] == False][g].mean()
    print(f"  {g:<35} overall={overall:.0%}  converters={c_rate:.0%}  non={n_rate:.0%}")

# Save goal results
goal_results.to_csv("splendor_analytics/outputs/trial_goals.csv", index=False)
print("\nSaved → outputs/trial_goals.csv")


# In[39]:


#  Visualize: How many organisations completed each trial goal? 

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK)
fig.suptitle("How many organisations completed each trial goal?",
             fontsize=14, fontweight="bold", color="white", y=1.02)

goal_labels = {
    "goal_scheduling_core":    "Created ≥3 Shifts",
    "goal_schedule_viewed":    "Viewed Schedule (Mobile)",
    "goal_punchclock_used":    "Used PunchClock",
    "goal_timesheet_approved": "Approved Timesheet/Shift",
    "goal_team_comms":         "Sent Team Message",
}

for i, (label, subset) in enumerate([
    ("Converted",     goal_results[goal_results["converted"] == True]),
    ("Not Converted", goal_results[goal_results["converted"] == False]),
]):
    rates  = subset[goal_cols].mean()
    labels = [goal_labels[g] for g in goal_cols]
    color  = MINT if i == 0 else CORAL

    axes[i].set_facecolor(DARK)
    bars = axes[i].barh(labels, rates.values * 100, color=color, edgecolor=DARK, alpha=0.85)
    axes[i].set_title(label, fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Completion Rate (%)", fontsize=11)
    axes[i].set_xlim(0, 115)
    axes[i].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[i].grid(True, axis="x")

    for bar, v in zip(bars, rates.values):
        axes[i].text(v * 100 + 1, bar.get_y() + bar.get_height() / 2,
                     f"{v:.0%}", va="center", fontsize=10, color="white")

plt.tight_layout()
plt.show()


# - Scheduling Core and Schedule Viewed are the most completed goals — **58%** and **47%** of all organisations respectively, confirming that shift creation and mobile schedule viewing are the most natural entry points into the platform.
#   
# - Completion rates are nearly identical between converters and non-converters across all five goals, reinforcing the earlier finding that no single activity or goal is a strong standalone predictor of conversion
# 
# - Team Communication is the least completed goal **(15%)** and is actually slightly higher among non-converters **(16%)** than converters **(12%)**, suggesting that sending messages during the trial does not drive conversion and may not belong in the activation definition without further validation

# In[ ]:





# ### TRIAL ACTIVATION — Organizations that completed ALL 5 goals

# In[40]:


# =============================================================================
# TRIAL ACTIVATION — Organizations that completed ALL 5 goals
# =============================================================================

goal_results["is_activated"]    = goal_results[goal_cols].all(axis=1)
goal_results["goals_completed"] = goal_results[goal_cols].sum(axis=1)  # Create a numeric count: How many total goals (out of 5) did this org complete?

# Activation tier, a function to bucket organizations into 'Engagement Tiers'
def assign_tier(row):
    if row["is_activated"]:         return "Fully Activated"   # Tier 1: The Gold Standard (100% completion)
    if row["goals_completed"] >= 3: return "Partially (≥3 goals)"  # Tier 2: Heavy users who haven't quite finished everything
    if row["goals_completed"] >= 1: return "Early (1-2 goals)"     # Tier 3: Casual users just testing the waters
    return "No Activation"                                         # Tier 4: Registered but did essentially nothing


goal_results["activation_tier"] = goal_results.apply(assign_tier, axis=1)

# Summary
tier_order   = ["No Activation", "Early (1-2 goals)", "Partially (≥3 goals)", "Fully Activated"]
tier_summary = (
    goal_results.groupby("activation_tier")
    .agg(
        n_orgs          = ("organization_id", "count"),  # Count how many orgs fall in this bucket
        conversion_rate = ("converted", "mean")          # Calculate the actual conversion % for this bucket
    )
    .reindex(tier_order)
    .reset_index()
)

print(tier_summary)


# In[ ]:





# In[41]:


# ── Visualize: Does reaching activation milestones improve conversion? ────────

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
ax.set_facecolor(DARK)

tier_colors = [GRAY, AMBER, LAVENDER, MINT]
bars = ax.bar(
    tier_summary["activation_tier"],
    tier_summary["conversion_rate"] * 100,
    color=tier_colors, edgecolor=DARK, alpha=0.88
)

ax.set_title("Does reaching activation milestones improve conversion?",
             fontsize=14, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Activation Tier",    fontsize=11)
ax.set_ylabel("Conversion Rate (%)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.tick_params(axis="x", labelsize=10)
ax.grid(True, axis="y")

for bar, (_, row) in zip(bars, tier_summary.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{row['conversion_rate']:.0%}\nn={row['n_orgs']}",
        ha="center", fontsize=10, color="white"
    )

plt.tight_layout()
plt.show()


# - Fully Activated organisations have the lowest conversion rate (16%). Surprisingly, completing all 5 goals does not lead to the highest conversion, suggesting the current activation definition may not be well-calibrated to predict conversion
# 
# - Partially activated organisations (≥3 goals) have the highest conversion rate (27%) — indicating that a 3-goal threshold may be a more meaningful activation benchmark than requiring all 5 goals
# 
# - Even organisations with no activation convert at 25% — this challenges the entire activation framework and suggests that conversion at Splendor Analytics may be driven by factors outside of in-app behaviour, such as sales outreach, pricing, or organisational decision-making timelines.

# In[ ]:





# ## TASK 2

# ### Building an SQL-based model providing two sources that would live in the marts layer of the data warehouse:
# 
# 

# connecting Python directly to MySQL

# In[42]:


import pandas as pd
import mysql.connector


# ### Connect Python to MySQL and insert the data

# In[43]:


import mysql.connector
import pandas as pd
import numpy as np

# ── Load your cleaned dataframe ───────────────────────────────────────────────
df2 = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Splendor Data Challenge\splendor_analytics\outputs\clean_events.csv")

# ── Replace NaN with None so MySQL stores them as NULL 
df2 = df2.replace({np.nan: None})


# In[44]:


conn = mysql.connector.connect(
    host        = "localhost",
    user        = "root",
    password    = "1234",
    database    = "splendor_analytics",
    auth_plugin = "mysql_native_password"   # ← handles auth issues
)
cursor = conn.cursor()
print("✓ Connected to MySQL")





# In[47]:


import pandas as pd
import numpy as np

# Load cleaned data
df2 = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Splendor Data Challenge\splendor_analytics\outputs\clean_events.csv")

# Replace NaN with None so MySQL stores as NULL
df2 = df2.where(pd.notnull(df2), None)

print(f"Total rows to insert: {len(df2)}")
print(df2.dtypes)
print(df2.head(3))


# In[ ]:





# In[48]:


# Fix converted_at — replace NaN with None for MySQL NULL
df2["converted_at"] = df2["converted_at"].replace({np.nan: None})

# Confirm fix
print(f"converted_at NaN count  : {df2['converted_at'].isna().sum()}")
print(f"converted_at None count : {(df2['converted_at'].isnull()).sum()}")
print(df2[["converted", "converted_at", "before_conversion"]].head(3))


# In[ ]:





# In[49]:


insert_query = """
    INSERT INTO stg_trial_events (
        organization_id, activity_name, timestamp, converted,
        converted_at, trial_start, trial_end, days_from_start,
        trial_length, before_conversion, activity_canonical, activity_category
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

rows       = list(df2.itertuples(index=False, name=None))
batch_size = 5000
total      = len(rows)

for i in range(0, total, batch_size):
    batch = rows[i : i + batch_size]
    cursor.executemany(insert_query, batch)
    conn.commit()
    print(f"  Inserted rows {i+1} to {min(i+batch_size, total)} of {total}")

print(f"\n✓ All {total} rows inserted")


# In[ ]:





# In[50]:


cursor.execute("SELECT COUNT(*) FROM stg_trial_events")
print(f"✓ Total rows in MySQL: {cursor.fetchone()[0]}")


# In[ ]:





# ###  Create intermediate view

# In[51]:


cursor.execute("DROP VIEW IF EXISTS int_org_activity_summary")

cursor.execute("""
    CREATE VIEW int_org_activity_summary AS
    SELECT
        organization_id,
        MAX(converted)                             AS converted,
        MAX(converted_at)                          AS converted_at,
        MAX(trial_start)                           AS trial_start,
        MAX(trial_end)                             AS trial_end,
        COUNT(*)                                   AS total_events,
        COUNT(DISTINCT activity_name)              AS distinct_activities,
        COUNT(DISTINCT activity_category)          AS distinct_categories,
        COUNT(DISTINCT DATE(timestamp))            AS active_days,
        SUM(activity_name = 'Scheduling.Shift.Created')                AS cnt_shift_created,
        SUM(activity_name = 'Scheduling.Shift.Approved')               AS cnt_shift_approved,
        SUM(activity_name = 'Mobile.Schedule.Loaded')                  AS cnt_schedule_loaded,
        SUM(activity_name = 'PunchClock.PunchedIn')                    AS cnt_punched_in,
        SUM(activity_name = 'PunchClock.PunchedOut')                   AS cnt_punched_out,
        SUM(activity_name = 'Timesheets.BulkApprove.Confirmed')        AS cnt_timesheet_bulk_approved,
        SUM(activity_name = 'Communication.Message.Created')           AS cnt_message_created,
        SUM(activity_name = 'Scheduling.Shift.AssignmentChanged')      AS cnt_shift_assignment_changed,
        SUM(activity_name = 'Scheduling.Template.ApplyModal.Applied')  AS cnt_template_applied,
        SUM(activity_name = 'Scheduling.Availability.Set')             AS cnt_availability_set,
        SUM(activity_name = 'Scheduling.ShiftSwap.Created')            AS cnt_shift_swap_created,
        SUM(activity_name = 'Scheduling.ShiftSwap.Accepted')           AS cnt_shift_swap_accepted,
        SUM(activity_name = 'Scheduling.ShiftHandover.Created')        AS cnt_shift_handover_created,
        SUM(activity_name = 'Scheduling.ShiftHandover.Accepted')       AS cnt_shift_handover_accepted,
        SUM(activity_name = 'Scheduling.OpenShiftRequest.Created')     AS cnt_open_shift_req_created,
        SUM(activity_name = 'Scheduling.OpenShiftRequest.Approved')    AS cnt_open_shift_req_approved,
        SUM(activity_name IN (
            'Shift.View.Opened','ShiftDetails.View.Opened'))           AS cnt_shift_viewed,
        SUM(activity_name = 'Absence.Request.Created')                 AS cnt_absence_created,
        SUM(activity_name = 'Absence.Request.Approved')                AS cnt_absence_approved,
        SUM(activity_name = 'Absence.Request.Rejected')                AS cnt_absence_rejected,
        SUM(activity_name = 'PunchClock.Entry.Edited')                 AS cnt_entry_edited,
        SUM(activity_name = 'PunchClockStartNote.Add.Completed')       AS cnt_start_note,
        SUM(activity_name = 'PunchClockEndNote.Add.Completed')         AS cnt_end_note,
        SUM(activity_name IN (
            'Break.Activate.Started','Break.Activate.Finished'))       AS cnt_break_events,
        SUM(activity_name = 'Integration.Xero.PayrollExport.Synced')   AS cnt_payroll_synced,
        SUM(activity_name = 'Revenue.Budgets.Created')                 AS cnt_budget_created
    FROM stg_trial_events
    GROUP BY organization_id
""")
conn.commit()

# Verify
cursor.execute("SELECT COUNT(*) FROM int_org_activity_summary")
print(f"✓ int_org_activity_summary rows: {cursor.fetchone()[0]}")


# ### Create mart_trial_goals

# In[52]:


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
print(f"✓ mart_trial_goals rows: {cursor.fetchone()[0]}")


# In[ ]:





# ### Create mart_trial_activation

# In[53]:


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
print(f"✓ mart_trial_activation rows: {cursor.fetchone()[0]}")


# In[ ]:





# ### Final verification

# In[54]:


# Activation tier summary
cursor.execute("""
    SELECT
        activation_tier,
        COUNT(*)                        AS n_orgs,
        ROUND(AVG(converted) * 100, 1) AS conversion_rate_pct
    FROM mart_trial_activation
    GROUP BY activation_tier
""")

results = cursor.fetchall()
print(f"\n{'Tier':<35} {'Orgs':>6} {'Conv Rate':>10}")
print("-" * 55)
for row in results:
    print(f"{row[0]:<35} {row[1]:>6} {row[2]:>9}%")

cursor.close()
conn.close()
print("\n✓ Connection closed")


# In[ ]:





# Task 2 — SQL Model Building Summary
# 
# - The SQL models were built using a **Python-to-MySQL** connection via **mysql-connector-python** rather than direct SQL execution, due to difficulty importing the large dataset size (102,895 rows) using MySQL Workbench's import wizard limits.
# 
# The following steps were taken:
# 
# - Database Setup — A dedicated splendor_analytics database was created in MySQL and a staging table **stg_trial_events** was defined to receive the cleaned event data.
# 
# - Data Import — The cleaned dataset was loaded from Python into MySQL in batches of 5,000 rows using **cursor.executemany()**, successfully inserting all 102,895 rows
# 
# - Intermediate View — **int_org_activity_summary** was created as a SQL view, aggregating all 28 activities into count columns at the organisation level (966 rows)
# 
# - Mart Table 1 — **mart_trial_goals** was created from the intermediate view, computing a **True/False** flag for each of the 5 defined trial goals per organisation
# 
# - Mart Table 2 — **mart_trial_activation** was created from mart_trial_goals, deriving the full activation flag (is_activated) and assigning each organisation an activation tier
# 
# 
# All SQL logic is fully documented in task2_sql_models.sql in the project repository.
#  

# In[ ]:





# ## TASK 3

# ### Basic Descriptive analyses and Product Metrics

# ###  Load data

# In[55]:


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# Load all needed tables 
events     = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Splendor Data Challenge\splendor_analytics\outputs\clean_events.csv")
orgs       = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Splendor Data Challenge\splendor_analytics\outputs\org_features.csv")
goals      = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Splendor Data Challenge\splendor_analytics\outputs\trial_goals.csv")

# Fix data types
events["timestamp"]   = pd.to_datetime(events["timestamp"])
events["trial_start"] = pd.to_datetime(events["trial_start"])
orgs["converted"]     = orgs["converted"].astype(bool)
goals["converted"]    = goals["converted"].astype(bool)

GOAL_COLS = [c for c in goals.columns if c.startswith("goal_")]
goals["is_activated"]    = goals[GOAL_COLS].all(axis=1)
goals["goals_completed"] = goals[GOAL_COLS].sum(axis=1)

print("✓ Data loaded successfully")
print(f"  Events : {len(events):,}")
print(f"  Orgs   : {len(orgs):,}")
print(f"  Goals  : {len(goals):,}")


# In[ ]:





# ### Computing product metrics

# In[59]:


# Metric 1: Overall Conversion Rate 
total_orgs  = len(orgs)
converted   = orgs["converted"].sum()
not_conv    = total_orgs - converted
conv_rate   = converted / total_orgs * 100

print(f"\n Metric 1 — Overall Conversion Rate")
print(f"   Total Trialists    : {total_orgs:,}")
print(f"   Converted          : {converted:,}")
print(f"   Not Converted      : {not_conv:,}")
print(f"   Conversion Rate    : {conv_rate:.1f}%")


# In[ ]:





# In[61]:


# Metric 2: Trial Activation Rate
activation_rate = goals["is_activated"].mean() * 100

print(f"\nMetric 2 — Trial Activation Rate")
print(f"   Fully Activated    : {goals['is_activated'].sum():,}")
print(f"   Activation Rate    : {activation_rate:.1f}%")


# In[ ]:





# In[62]:


# Metric 3: Time to Convert 

orgs["converted_at"] = pd.to_datetime(orgs["converted_at"], errors="coerce")
orgs["trial_start"]  = pd.to_datetime(orgs["trial_start"],  errors="coerce")
orgs["days_to_convert"] = (
    orgs["converted_at"] - orgs["trial_start"]
).dt.total_seconds() / 86400

ttc = orgs[orgs["converted"] == True]["days_to_convert"].dropna()

print(f"\n Metric 3 — Time to Convert (days)")
print(f"   Median             : {ttc.median():.1f} days")
print(f"   Mean               : {ttc.mean():.1f} days")
print(f"   Fastest (Min)      : {ttc.min():.1f} days")
print(f"   Slowest (Max)      : {ttc.max():.1f} days")


# In[ ]:





# In[63]:


# Metric4: Feature Adoption Rate

print(f"\n Metric 4 — Feature Adoption Rate (% of orgs using each activity)")
activity_adoption = (
    events.groupby("activity_category")["organization_id"]
    .nunique()
    .reset_index()
    .rename(columns={"organization_id": "n_orgs"})
)
module_adoption["adoption_rate_pct"] = (
    module_adoption["n_orgs"] / total_orgs * 100
).round(1)
module_adoption = module_adoption.sort_values("adoption_rate_pct", ascending=False)
print(module_adoption.to_string(index=False))


# In[ ]:





# In[64]:


# Metric 5: Average Events per Organization

print(f"\n Metric 5 — Engagement Depth")
print()
print(f"   Avg Events/Org     : {orgs['total_events'].mean():.0f}")
print(f"   Avg Active Days    : {orgs['active_days'].mean():.1f}")
print(f"   Avg Activities     : {orgs['distinct_activities'].mean():.1f}")
print(f"   Median Events/Org  : {orgs['total_events'].median():.0f}")


# In[ ]:





# In[65]:


#  Metric 6: Day 1 Retention
# % of organizations that had activity beyond day 0

day1_retention = (
    events[events["days_from_start"] >= 1]["organization_id"].nunique()
    / total_orgs * 100
)
print(f"\n M6 — Retention")
print(f"   Day 1 Retention    : {day1_retention:.1f}%")


# In[ ]:





# In[67]:


# Metric 7: Activation Funnel

print(f"\n Metric 7 — Activation Funnel")
for n in range(1, 6):
    pct = (goals["goals_completed"] >= n).mean() * 100
    print(f"   Reached >= {n} goal(s) : {pct:.1f}%")


# In[ ]:





# In[68]:


#  Metric 8: Conversion Without Activation 

conv_no_activation = (
    goals[(goals["converted"] == True) & (goals["is_activated"] == False)]
).shape[0]
pct_conv_no_act = conv_no_activation / converted * 100

print(f"\n Metric 8 — Converted Without Full Activation")
print(f"   Count              : {conv_no_activation:,}")
print(f"   % of All Converted : {pct_conv_no_act:.1f}%")


# In[72]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Colour palette
PURPLE   = "#6C3FC8"
LAVENDER = "#A78BFA"
MINT     = "#34D399"
CORAL    = "#F87171"
AMBER    = "#FBBF24"
DARK     = "#1E1B2E"
GRAY     = "#6B7280"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   DARK,
    "axes.edgecolor":   "#3D3557",
    "axes.labelcolor":  "white",
    "xtick.color":      GRAY,
    "ytick.color":      GRAY,
    "text.color":       "white",
    "grid.color":       "#2D2A45",
    "grid.linewidth":   0.6,
})


# In[77]:


# Visualization 1: Core metrics dashboard

# KPI Scorecard
# Clear all existing plots first
plt.close("all")

fig, ax = plt.subplots(figsize=(14, 3), facecolor=DARK)
ax.set_facecolor(DARK)
ax.axis("off")

kpis = [
    ("Conversion Rate",      f"{conv_rate:.1f}%",              MINT),
    ("Total Trialists",      f"{total_orgs:,}",                LAVENDER),
    ("Activation Rate",      f"{activation_rate:.1f}%",        AMBER),
    ("Median Days to Convert", f"{ttc.median():.0f} days",     CORAL),
    ("Avg Events per Org",   f"{orgs['total_events'].mean():.0f}", "#60A5FA"),
]

for i, (label, value, color) in enumerate(kpis):
    x = 0.1 + i * 0.2
    ax.text(x, 0.72, value, ha="center", fontsize=22,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(x, 0.2,  label, ha="center", fontsize=10,
            color=GRAY, transform=ax.transAxes)
    if i < len(kpis) - 1:
        ax.axvline(0.2 * (i + 1), color="#3D3557",
                   linewidth=1, ymin=0.1, ymax=0.9)

ax.set_title("Key Product Metrics at a Glance",
             fontsize=13, color=GRAY, pad=10, loc="left")
plt.tight_layout()
plt.show()


# In[ ]:





# In[74]:


# Visualization 2: Feature adoption rate

# What percentage of organisations adopted each module?

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
ax.set_facecolor(DARK)

activity_colors = {
    "Scheduling":          LAVENDER,
    "Time Tracking":       MINT,
    "Payroll":             AMBER,
    "Absence":             CORAL,
    "Revenue & Budgeting": PURPLE,
    "Communication":       "#60A5FA",
}
colors = module_adoption["activity_category"].map(activity_colors)

bars = ax.barh(
    module_adoption["activity_category"],
    module_adoption["adoption_rate_pct"],
    color=colors, edgecolor=DARK, alpha=0.88
)

for bar, v in zip(bars, module_adoption["adoption_rate_pct"]):
    ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=10, color="white")

ax.set_title("What percentage of organisations adopted each activity?",
             fontsize=14, fontweight="bold", color="white", pad=15)
ax.set_xlabel("% of Organisations", fontsize=11)
ax.set_xlim(0, 115)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, axis="x")
plt.tight_layout()
plt.show()


# In[ ]:





# In[75]:


# Monthly cohort performance

# ── Fig 5: How do trial cohorts compare month by month? ──────────────────────
events["trial_month"] = pd.to_datetime(
    events["trial_start"]).dt.to_period("M").astype(str)

monthly = (
    events.drop_duplicates("organization_id")
    .groupby("trial_month")
    .agg(
        n_orgs      = ("organization_id", "count"),
        conv_rate   = ("converted",       "mean")
    )
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 5), facecolor=DARK)
ax1.set_facecolor(DARK)
ax2 = ax1.twinx()
ax2.set_facecolor(DARK)

ax1.bar(monthly["trial_month"], monthly["n_orgs"],
        color=LAVENDER, alpha=0.7, label="Trialists")
ax2.plot(monthly["trial_month"], monthly["conv_rate"] * 100,
         color=MINT, marker="o", linewidth=2, label="Conv Rate")

ax1.set_title("How do monthly trial cohorts compare in volume and conversion?",
              fontsize=13, fontweight="bold", color="white", pad=15)
ax1.set_xlabel("Trial Start Month", fontsize=11)
ax1.set_ylabel("Number of Trialists", color=LAVENDER, fontsize=11)
ax2.set_ylabel("Conversion Rate (%)", color=MINT,    fontsize=11)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax1.tick_params(axis="y", labelcolor=LAVENDER)
ax2.tick_params(axis="y", labelcolor=MINT)
ax1.grid(True, axis="y", alpha=0.4)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=10)
plt.tight_layout()
plt.show()


# In[ ]:





# ### Actionable recommendations

# 1. INTERVENE EARLY
# 
#    
#    Only 39% of organizations have any activity after Day 1.
#    An automated Day-2 nudge targeting inactive organizations
#    could significantly improve trial engagement.
# 
# 3. FOCUS ON SCHEDULING DEPTH
# 
# 
#    Scheduling is the most adopted module (88% of organizations)
#    and the strongest engagement signal. Onboarding should
#    guide organizations to create their first 3 shifts quickly.
# 
# 5. PUSH PUNCHCLOCK ADOPTION
# 
# 
#    Only 22% of organizations used the PunchClock despite it being
#    a core value driver. A dedicated onboarding step for
#    time tracking could improve activation rates.
# 
# 7. REVISIT THE ACTIVATION DEFINITION
# 
# 
#    Fully activated organizations (all 5 goals) have the LOWEST
#    conversion rate at 16.2%. Consider a 3-goal threshold
#    as a more predictive activation benchmark.
# 
# 9. TARGET HIGH-INTENT NON-CONVERTERS
# 
# 
#    Organizations that completed 3+ goals but did not convert are
#    strong candidates for direct sales outreach — they
#    experienced product value but did not commit.
# 
# 11. ADD A TRIAL-ENDING PROMPT
# 
# 
#     Median conversion happens at Day 30 (trial expiry).
#    A Day-25 prompt reminding organizations of trial expiry could
#    accelerate conversion decisions.

# In[ ]:




