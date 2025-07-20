import duckdb, pathlib

con = duckdb.connect("../data/processed/enron.duckdb")

# One‑liner: read every parquet in interim and create a table
con.execute("""
    CREATE OR REPLACE TABLE parsed AS
    SELECT * FROM parquet_scan('../data/interim/cleaned_emails_part*.parquet')
""")
# Verify row count
print(con.table("parsed").count("*").fetchone()[0])   # ➜ 517401 (full set)
