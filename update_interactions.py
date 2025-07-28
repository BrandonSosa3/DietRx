import requests
import pandas as pd
import os
import re

os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/drug_food_interactions.csv"

# List of food-related keywords to identify food interactions
FOOD_TERMS = [
    "grapefruit", "alcohol", "caffeine", "milk", "cheese", "banana", "spinach",
    "broccoli", "green tea", "charcoal grilled meat", "vitamin K", "cranberry",
    "garlic", "ginseng", "St. John's wort", "licorice", "fiber supplements",
    "iron supplements", "calcium supplements", "chocolate", "salt substitutes",
    "fish oil", "soy products", "citrus", "high-fat meals"
]

def fetch_openfda_interactions(queries=None, limit=50):
    if queries is None:
        queries = [f"drug_interactions:{term}" for term in FOOD_TERMS]

    base_url = "https://api.fda.gov/drug/label.json"
    all_results = []

    for query in queries:
        params = {"search": query, "limit": limit}
        try:
            print(f"Fetching: {query}")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            all_results.extend(data.get("results", []))
        except Exception as e:
            print(f"Error fetching {query}: {e}")

    return all_results

def contains_food_term(text):
    """Return True if text mentions any known food term (case-insensitive)."""
    text_lower = text.lower()
    for term in FOOD_TERMS:
        if re.search(r'\b' + re.escape(term.lower()) + r'\b', text_lower):
            return True
    return False

def process_results(results):
    interactions = []
    for item in results:
        drug_name = item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
        interaction_texts = item.get("drug_interactions", [])
        for text in interaction_texts:
            # Only keep if interaction text mentions any food term
            if contains_food_term(text):
                interactions.append({"drug": drug_name, "interaction": text})
    return pd.DataFrame(interactions)

def update_interactions():
    print("Fetching drug-food interactions from OpenFDA...")
    results = fetch_openfda_interactions()
    if not results:
        print("No data fetched.")
        return

    df = process_results(results)
    if df.empty:
        print("No drug-food interactions found.")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Updated drug-food interaction file saved at {OUTPUT_FILE}")

if __name__ == "__main__":
    update_interactions()



