import pandas as pd
import os

# Adjust path based on where you're running this script
csv_path = os.path.join("data", "faqs.csv")


# Load the CSV
df = pd.read_csv(csv_path)

# Preview the data
print("✅ Data Loaded:")
print(df.head())

print(f"\nTotal FAQs loaded: {len(df)}")
