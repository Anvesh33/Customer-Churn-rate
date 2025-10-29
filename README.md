# Telco Customer Churn — Data Load to SQLite and Quick EDA

This project loads the Telco Customer Churn dataset into a local SQLite database and runs a few exploratory queries to validate the data and provide quick insights.

## Dataset

- Source: https://www.kaggle.com/blastchar/telco-customer-churn
- Expected path: data/Telco-Customer-Churn.csv

## Requirements

- Python 3.8+
- pandas

Install:

- pip install pandas

## Getting Started

1. Place Telco-Customer-Churn.csv in data/ (create if missing).
2. Run:

- python 1_load_data_to_sql.py

What it does:

- Creates folders: data, database, models, outputs (if missing)
- Loads CSV into SQLite: database/churn_analysis.db
- Creates table: customers
- Prints dataset info and runs sample queries

## Sample Queries Executed

- SELECT Churn, COUNT(\*) FROM customers GROUP BY Churn
- SELECT gender, AVG(tenure) FROM customers GROUP BY gender
- SELECT Contract, AVG(MonthlyCharges) FROM customers GROUP BY Contract
- SELECT InternetService, COUNT(\*) FROM customers GROUP BY InternetService

## Outputs

- Database: database/churn_analysis.db
- Table: customers
- Console output: dataset shape, columns, missing values, and query summaries

## Inspecting the Database

- Use DB Browser for SQLite or:
- sqlite3 database/churn_analysis.db
- .tables
- PRAGMA table_info(customers);
- SELECT COUNT(\*) FROM customers;

## Project Structure

- 1_load_data_to_sql.py
- data/
- database/
- models/
- outputs/
- README.md

## Troubleshooting

- CSV not found: Ensure file exists at data/Telco-Customer-Churn.csv (exact name).
- Encoding issues: Open CSV in UTF-8 or re-save via spreadsheet tools.

## Next Steps

- Train a churn model (see 2_churn_model_training.py if available).
- Add feature engineering, model tracking, and dashboards.

## Push to GitHub

From Windows Terminal/PowerShell, run:

```powershell
# 1) Go to the project folder
cd "D:\Data Analytics\Customer Churn"

# 2) Initialize git (if not already)
git init
git branch -m main

# (optional) configure your identity once
git config user.name "Your Name"
git config user.email "you@example.com"

# 3) Stage and commit
git add .
git commit -m "Initial commit: data load script and README"

# 4) Add the remote
git remote add origin https://github.com/Anvesh33/Customer-Churn-rate.git

# 5) Push to GitHub (if the remote is empty)
git push -u origin main
```

If the remote already has commits (e.g., created with a README), pull and rebase, then push:

```powershell
cd "D:\Data Analytics\Customer Churn"
git fetch origin
git pull --rebase origin main  # resolve conflicts if any
git push -u origin main
```

Notes:

- Ensure you’re authenticated to GitHub (Git will prompt in browser or use: gh auth login).
- If remote exists already in your repo, update it: git remote set-url origin https://github.com/Anvesh33/Customer-Churn-rate.git
