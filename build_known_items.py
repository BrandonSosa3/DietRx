import pandas as pd
import re

# Path to your CSV
csv_path = "data/drug_food_interactions.csv"

# Load CSV
df = pd.read_csv(csv_path)

# 1. Build known drugs list from 'drug' column
known_drugs = sorted(set(df["drug"].dropna().str.lower()))
with open("data/known_drugs.txt", "w") as f:
    f.write("\n".join(known_drugs))
print(f"✅ Saved {len(known_drugs)} known drugs to data/known_drugs.txt")

# 2. Build known foods list from 'food' column (new column expected)
known_foods = sorted(set(df["food"].dropna().str.lower()))
print(f"Extracted {len(known_foods)} known foods from 'food' column.")

# Additionally, supplement known foods by extracting food keywords from 'interaction' text (optional)
food_keywords = [
    "grapefruit", "bananas", "milk", "cheese", "spinach", "alcohol",
    "coffee", "tea", "orange", "chocolate", "broccoli", "cabbage",
    "garlic", "fish", "yogurt", "avocado", "soy", "wine", "beer"
]

# Compile regex for faster searching
food_pattern = re.compile(r"\b(" + "|".join(map(re.escape, food_keywords)) + r")\b", re.IGNORECASE)

# Extract foods mentioned in interaction text to supplement list
for interaction in df["interaction"].dropna():
    matches = food_pattern.findall(interaction)
    for m in matches:
        known_foods.append(m.lower())

# Remove duplicates and sort again
known_foods = sorted(set(known_foods))

with open("data/known_foods.txt", "w") as f:
    f.write("\n".join(known_foods))
print(f"✅ Saved {len(known_foods)} known foods to data/known_foods.txt")


