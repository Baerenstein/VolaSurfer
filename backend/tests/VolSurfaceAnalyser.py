import pandas as pd
from sqlalchemy import create_engine

# Database connection URI
DB_URI = "postgresql://mikeb:postgres@localhost:5432/optionsdb"

# Connect to the database
engine = create_engine(DB_URI)

# Query all volatility surface data across timestamps
query = """
    SELECT sp.moneyness, sp.days_to_expiry, sp.implied_vol, s.timestamp
    FROM surface_points sp
    JOIN surfaces s ON sp.surface_id = s.id;
"""

# Load data into a Pandas DataFrame
df = pd.read_sql_query(query, engine)

# Check if data is available
if df.empty:
    print("No volatility surfaces found in the database.")
else:
    # Count the number of unique timestamps
    unique_timestamps = df["timestamp"].nunique()
    print(f"\nNumber of unique volatility surfaces stored: {unique_timestamps}")

    # Group by (moneyness, days_to_expiry) and compute min, max, and average implied volatility
    summary = df.groupby(["moneyness", "days_to_expiry"])["implied_vol"].agg(
        min_vol="min", max_vol="max", avg_vol="mean"
    ).reset_index()

    # Sort the results for easier reading
    summary = summary.sort_values(by=["moneyness", "days_to_expiry"])

    # Display results in the terminal
    print("\n--- Min Volatility Surface ---")
    print(summary[["moneyness", "days_to_expiry", "min_vol"]].head(10))

    print("\n--- Max Volatility Surface ---")
    print(summary[["moneyness", "days_to_expiry", "max_vol"]].head(10))

    print("\n--- Average Volatility Surface ---")
    print(summary[["moneyness", "days_to_expiry", "avg_vol"]].head(10))

    # Save results to a CSV file for further analysis
    summary.to_csv("volatility_surface_summary.csv", index=False)
    print("\nSaved summary to volatility_surface_summary.csv")
