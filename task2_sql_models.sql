-- SPLENDOR ANALYTICS — TRIAL ACTIVATION CHALLENGE
-- Task 2: SQL Data Models — Marts Layer

-- OVERVIEW
-- This SQL file documents the data models built for the Trial Activation
-- challenge. The models were created using a Python-to-MySQL connection
-- (mysql-connector-python) rather than direct SQL execution, due to the
-- large dataset size (102,895 rows) which exceeded MySQL Workbench's
-- import wizard limits

USE splendor_analytics;



-- LAYER 1: STAGING TABLE
-- stg_trial_events — cleaned event-level data, one row per event
-- Preview the staging table
SELECT * FROM splendor_analytics.stg_trial_events LIMIT 10;

-- Confirm total rows loaded
SELECT COUNT(*) AS total_events FROM splendor_analytics.stg_trial_events;

-- Confirm unique organisations
SELECT COUNT(DISTINCT organization_id) AS total_orgs FROM splendor_analytics.stg_trial_events;


-- LAYER 2: INTERMEDIATE VIEW
-- int_org_activity_summary — one row per organisation
-- Pivots all 28 activities into individual count columns
-- Also computes overall engagement metrics per org
-- Confirm row count (should match unique orgs = 966)
SELECT COUNT(*) AS total_orgs FROM splendor_analytics.int_org_activity_summary;

-- Top 10 most active organisations by total events
SELECT
    organization_id,
    total_events,
    distinct_activities,
    active_days,
    converted
FROM splendor_analytics.int_org_activity_summary
ORDER BY total_events DESC
LIMIT 10;


-- LAYER 3: MART TABLES
-- mart_trial_goals
-- Grain   : one row per organisation
-- Purpose : tracks whether each organisation completed each of the
--           five trial goals defined from the conversion driver analysis

ALTER TABLE splendor_analytics.mart_trial_goals
ADD PRIMARY KEY (organization_id);

ALTER TABLE splendor_analytics.mart_trial_activation
ADD CONSTRAINT fk_activation_goals
FOREIGN KEY (organization_id)
REFERENCES mart_trial_goals(organization_id);

-- Check primary key on mart_trial_goals
SHOW KEYS FROM splendor_analytics.mart_trial_goals WHERE Key_name = 'PRIMARY';

-- Check foreign key on mart_trial_activation
SELECT
    CONSTRAINT_NAME,
    TABLE_NAME,
    COLUMN_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'splendor_analytics'
  AND CONSTRAINT_NAME = 'fk_activation_goals';


-- Preview trial goals table
SELECT * FROM splendor_analytics.mart_trial_goals LIMIT 10;

-- mart_trial_activation
-- Grain   : one row per organisation
-- Purpose : tracks which organisations achieved full Trial Activation
--           (all 5 goals completed) and assigns an activation tier

-- Preview trial activation table
SELECT * FROM splendor_analytics.mart_trial_activation LIMIT 10;

-- View all trial activation
SELECT * FROM splendor_analytics.mart_trial_activation;

-- BUSINESS INSIGHT QUERIES
-- Q1: What is the overall conversion rate across all trialists?
SELECT
    COUNT(*)                        AS total_orgs,
    SUM(converted)                  AS total_converted,
    ROUND(AVG(converted) * 100, 1) AS conversion_rate_pct
FROM splendor_analytics.mart_trial_activation;

-- Q2: How many organisations are in each activation tier
--     and what is their conversion rate?
SELECT 
    activation_tier,
    COUNT(*)                        AS n_orgs,
    ROUND(AVG(converted) * 100, 1) AS conversion_rate_pct
FROM splendor_analytics.mart_trial_activation
GROUP BY activation_tier;

-- Q3: What is the completion rate for each individual trial goal?
SELECT
    'Scheduling Core'    AS goal,
    ROUND(AVG(goal_scheduling_core)    * 100, 1) AS completion_rate_pct
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'Schedule Viewed',
    ROUND(AVG(goal_schedule_viewed)    * 100, 1)
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'PunchClock Used',
    ROUND(AVG(goal_punchclock_used)    * 100, 1)
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'Timesheet Approved',
    ROUND(AVG(goal_timesheet_approved) * 100, 1)
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'Team Communication',
    ROUND(AVG(goal_team_comms)         * 100, 1)
FROM splendor_analytics.mart_trial_goals
ORDER BY completion_rate_pct DESC;

-- Q5: How many organisations converted without achieving any activation?
--     (Indicates conversion driven by factors outside in-app behaviour)
SELECT
    COUNT(*)                        AS converted_without_activation,
    ROUND(COUNT(*) * 100.0 /
    (SELECT COUNT(*) FROM splendor_analytics.mart_trial_activation
     WHERE converted = 1), 1)      AS pct_of_all_conversions
FROM splendor_analytics.mart_trial_activation
WHERE converted_without_activation = 1;

-- Q6: How many organisations fully activated but never converted?
--     (Indicates activation leakage — value experienced but not monetised)
SELECT
    COUNT(*)                        AS activated_without_converting,
    ROUND(COUNT(*) * 100.0 /
    (SELECT COUNT(*) FROM splendor_analytics.mart_trial_activation
     WHERE is_activated = 1), 1)   AS pct_of_fully_activated
FROM splendor_analytics.mart_trial_activation
WHERE activated_without_converting = 1;

-- Q7: What percentage of all trialists reached each activation milestone?
SELECT
    ROUND(SUM(CASE WHEN goals_completed >= 1
              THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_reached_1_goal,
    ROUND(SUM(CASE WHEN goals_completed >= 2
              THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_reached_2_goals,
    ROUND(SUM(CASE WHEN goals_completed >= 3
              THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_reached_3_goals,
    ROUND(SUM(CASE WHEN goals_completed >= 4
              THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_reached_4_goals,
    ROUND(SUM(CASE WHEN goals_completed = 5
              THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_fully_activated
FROM splendor_analytics.mart_trial_activation;

-- Q8: Which single goal has the highest association with conversion?
--      (Compares conversion rate of orgs that completed vs skipped each goal) 
SELECT
    'Scheduling Core'   AS goal,
    ROUND(AVG(CASE WHEN goal_scheduling_core    = 1
              THEN converted END) * 100, 1)    AS conv_rate_completed,
    ROUND(AVG(CASE WHEN goal_scheduling_core    = 0
              THEN converted END) * 100, 1)    AS conv_rate_skipped
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'Schedule Viewed',
    ROUND(AVG(CASE WHEN goal_schedule_viewed    = 1
              THEN converted END) * 100, 1),
    ROUND(AVG(CASE WHEN goal_schedule_viewed    = 0
              THEN converted END) * 100, 1)
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'PunchClock Used',
    ROUND(AVG(CASE WHEN goal_punchclock_used    = 1
              THEN converted END) * 100, 1),
    ROUND(AVG(CASE WHEN goal_punchclock_used    = 0
              THEN converted END) * 100, 1)
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'Timesheet Approved',
    ROUND(AVG(CASE WHEN goal_timesheet_approved = 1
              THEN converted END) * 100, 1),
    ROUND(AVG(CASE WHEN goal_timesheet_approved = 0
              THEN converted END) * 100, 1)
FROM splendor_analytics.mart_trial_goals
UNION ALL
SELECT
    'Team Communication',
    ROUND(AVG(CASE WHEN goal_team_comms         = 1
              THEN converted END) * 100, 1),
    ROUND(AVG(CASE WHEN goal_team_comms         = 0
              THEN converted END) * 100, 1)
FROM splendor_analytics.mart_trial_goals
ORDER BY conv_rate_completed DESC;
