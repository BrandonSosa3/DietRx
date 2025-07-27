import requests
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/drug_food_interactions.csv"

def fetch_openfda_interactions(queries=None, limit=50):
    if queries is None:
        queries = [
            "drug_interactions:grapefruit",
            "drug_interactions:alcohol",
            "drug_interactions:caffeine",
            "drug_interactions:milk",
            "drug_interactions:cheese",
            "drug_interactions:bananas",
            "drug_interactions:spinach",
            "drug_interactions:broccoli",
            "drug_interactions:green tea",
            "drug_interactions:charcoal grilled meat",
            "drug_interactions:vitamin K",
            "drug_interactions:cranberry",
            "drug_interactions:garlic",
            "drug_interactions:ginseng",
            "drug_interactions:St. John's wort",
            "drug_interactions:licorice",
            "drug_interactions:fiber supplements",
            "drug_interactions:iron supplements",
            "drug_interactions:calcium supplements",
            "drug_interactions:chocolate",
            "drug_interactions:salt substitutes",
            "drug_interactions:fish oil",
            "drug_interactions:soy products",
            "drug_interactions:citrus",
            "drug_interactions:high-fat meals"
        ]

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

def process_results(results):
    interactions = []
    for item in results:
        drug_name = item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
        interaction_texts = item.get("drug_interactions", [])
        for text in interaction_texts:
            interactions.append({"drug": drug_name, "interaction": text})
    return pd.DataFrame(interactions)

def update_interactions():
    print("Fetching drug-food interactions...")
    results = fetch_openfda_interactions()
    if not results:
        print("No data fetched.")
        return

    df = process_results(results)
    if df.empty:
        print("No interactions found.")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Updated interaction file saved at {OUTPUT_FILE}")

if __name__ == "__main__":
    update_interactions()

