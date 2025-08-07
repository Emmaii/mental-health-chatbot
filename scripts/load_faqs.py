import pandas as pd
import os

csv_path = os.path.join("data", "faqs.csv")
df = pd.read_csv(csv_path)

print("✅ Data Loaded:")
print(df.head(), "\n")
print(f"Total FAQs loaded: {len(df)}")
