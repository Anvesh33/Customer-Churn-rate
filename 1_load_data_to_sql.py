import pandas as pd
import sqlite3
import os

os.makedirs('data', exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

if not os.path.exists('data/Telco-Customer-Churn.csv'):
    print("❌ Error: Telco-Customer-Churn.csv not found in data/ folder")
    print("Please download from: https://www.kaggle.com/blastchar/telco-customer-churn")
    exit()

print("📊 Loading data from CSV...")
df = pd.read_csv('data/Telco-Customer-Churn.csv')
print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")

print("\n📋 Dataset Info:")
print(f"- Shape: {df.shape}")
print(f"- Columns: {list(df.columns)}")
print(f"- Missing values: {df.isnull().sum().sum()}")

print("\n🗄️ Creating SQLite database...")
conn = sqlite3.connect('database/churn_analysis.db')

df.to_sql('customers', conn, if_exists='replace', index=False)
print("✅ Data loaded into 'customers' table")

print("\n🔍 Running sample queries...\n")

query1 = "SELECT Churn, COUNT(*) as count FROM customers GROUP BY Churn"
print("--- Churn Distribution ---")
print(pd.read_sql_query(query1, conn))

query2 = "SELECT gender, AVG(tenure) as avg_tenure FROM customers GROUP BY gender"
print("\n--- Average Tenure by Gender ---")
print(pd.read_sql_query(query2, conn))

query3 = "SELECT Contract, AVG(MonthlyCharges) as avg_charges FROM customers GROUP BY Contract"
print("\n--- Average Monthly Charges by Contract Type ---")
print(pd.read_sql_query(query3, conn))

query4 = "SELECT InternetService, COUNT(*) as count FROM customers GROUP BY InternetService"
print("\n--- Internet Service Distribution ---")
print(pd.read_sql_query(query4, conn))

conn.close()

print("\n✅ All done! Database created at: database/churn_analysis.db")
print("📌 Next step: Run 2_churn_model_training.py")
