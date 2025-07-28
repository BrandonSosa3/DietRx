#!/usr/bin/env python3
"""
Script to extract all unique drugs and foods from drug_food_interactions.csv
and update known_drugs.txt and known_foods.txt files
"""

import pandas as pd
import os

def update_known_items():
    """Extract unique drugs and foods from CSV and update known files"""
    
    # Read the CSV file
    try:
        df = pd.read_csv('data/drug_food_interactions.csv')
        print(f"✅ Loaded CSV with {len(df)} interactions")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Extract unique drugs and foods
    unique_drugs = sorted(df['drug'].unique())
    unique_foods = sorted(df['food'].unique())
    
    print(f"📊 Found {len(unique_drugs)} unique drugs and {len(unique_foods)} unique foods")
    
    # Write unique drugs to known_drugs.txt
    try:
        with open('data/known_drugs.txt', 'w') as f:
            for drug in unique_drugs:
                f.write(f"{drug}\n")
        print(f"✅ Updated data/known_drugs.txt with {len(unique_drugs)} drugs")
    except Exception as e:
        print(f"❌ Error writing known_drugs.txt: {e}")
    
    # Write unique foods to known_foods.txt
    try:
        with open('data/known_foods.txt', 'w') as f:
            for food in unique_foods:
                f.write(f"{food}\n")
        print(f"✅ Updated data/known_foods.txt with {len(unique_foods)} foods")
    except Exception as e:
        print(f"❌ Error writing known_foods.txt: {e}")
    
    # Print some examples
    print(f"\n📋 Sample drugs: {unique_drugs[:10]}")
    print(f"📋 Sample foods: {unique_foods[:10]}")
    
    return unique_drugs, unique_foods

if __name__ == "__main__":
    update_known_items() 