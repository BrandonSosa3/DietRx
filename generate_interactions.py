import pandas as pd

def generate_additional_interactions():
    """Generate additional realistic drug-food interactions"""
    
    # Additional drug-food interactions data
    additional_interactions = [
        # Statins
        ("atorvastatin", "pomegranate", "Pomegranate may increase atorvastatin levels, potentially raising the risk of muscle damage."),
        ("simvastatin", "pomegranate", "Pomegranate can inhibit enzymes that break down simvastatin, increasing drug levels."),
        ("rosuvastatin", "grapefruit", "Grapefruit may increase rosuvastatin absorption, leading to higher drug levels."),
        ("pravastatin", "alcohol", "Alcohol can increase the risk of liver problems when combined with pravastatin."),
        
        # ACE Inhibitors
        ("enalapril", "potassium supplements", "Potassium supplements can cause dangerously high potassium levels when taken with enalapril."),
        ("ramipril", "salt substitutes", "Salt substitutes containing potassium can increase potassium levels dangerously with ramipril."),
        ("quinapril", "banana", "Bananas are high in potassium and can cause hyperkalemia when taken with quinapril."),
        ("benazepril", "avocado", "Avocados are potassium-rich and can increase potassium levels when taking benazepril."),
        
        # ARBs
        ("losartan", "potassium supplements", "Potassium supplements can cause dangerously high potassium levels with losartan."),
        ("valsartan", "salt substitutes", "Salt substitutes with potassium can increase potassium levels with valsartan."),
        ("irbesartan", "banana", "Bananas can increase potassium levels when taking irbesartan."),
        ("candesartan", "avocado", "Avocados may increase potassium levels when combined with candesartan."),
        
        # Beta Blockers
        ("metoprolol", "grapefruit", "Grapefruit may increase metoprolol levels, potentially causing more side effects."),
        ("atenolol", "alcohol", "Alcohol can increase the sedative effects of atenolol."),
        ("propranolol", "caffeine", "Caffeine may reduce the effectiveness of propranolol."),
        ("carvedilol", "grapefruit", "Grapefruit can increase carvedilol levels and side effects."),
        
        # Calcium Channel Blockers
        ("amlodipine", "grapefruit", "Grapefruit can increase amlodipine levels, potentially causing more side effects."),
        ("diltiazem", "grapefruit", "Grapefruit may increase diltiazem absorption and effects."),
        ("verapamil", "grapefruit", "Grapefruit can significantly increase verapamil levels."),
        ("nifedipine", "grapefruit", "Grapefruit may increase nifedipine levels and side effects."),
        
        # Diuretics
        ("furosemide", "licorice", "Licorice can reduce the effectiveness of furosemide."),
        ("hydrochlorothiazide", "licorice", "Licorice may counteract the effects of hydrochlorothiazide."),
        ("spironolactone", "potassium supplements", "Potassium supplements can cause dangerously high potassium with spironolactone."),
        ("chlorthalidone", "licorice", "Licorice can reduce the effectiveness of chlorthalidone."),
        
        # Anticoagulants
        ("warfarin", "cranberry", "Cranberry may increase the effects of warfarin, raising bleeding risk."),
        ("warfarin", "garlic", "Garlic may increase the anticoagulant effects of warfarin."),
        ("warfarin", "ginger", "Ginger may increase the effects of warfarin and bleeding risk."),
        ("warfarin", "ginkgo", "Ginkgo may increase the effects of warfarin and bleeding risk."),
        ("warfarin", "green tea", "Green tea may decrease the effectiveness of warfarin."),
        ("warfarin", "alcohol", "Alcohol can increase the effects of warfarin and bleeding risk."),
        
        # Antiplatelets
        ("aspirin", "alcohol", "Alcohol can increase the risk of stomach bleeding when taking aspirin."),
        ("clopidogrel", "grapefruit", "Grapefruit may reduce the effectiveness of clopidogrel."),
        ("ticagrelor", "grapefruit", "Grapefruit may affect ticagrelor metabolism."),
        
        # Antibiotics
        ("ciprofloxacin", "dairy products", "Dairy products can reduce the absorption of ciprofloxacin."),
        ("doxycycline", "dairy products", "Dairy products can reduce the absorption of doxycycline."),
        ("tetracycline", "dairy products", "Dairy products can significantly reduce tetracycline absorption."),
        ("azithromycin", "aluminum antacids", "Aluminum antacids can reduce azithromycin absorption."),
        ("amoxicillin", "dairy products", "Dairy products may reduce amoxicillin absorption."),
        ("levofloxacin", "dairy products", "Dairy products can reduce levofloxacin absorption."),
        
        # Antifungals
        ("fluconazole", "grapefruit", "Grapefruit may increase fluconazole levels and side effects."),
        ("itraconazole", "grapefruit", "Grapefruit can significantly increase itraconazole levels."),
        ("ketoconazole", "grapefruit", "Grapefruit may increase ketoconazole levels."),
        
        # Antidepressants
        ("fluoxetine", "alcohol", "Alcohol can increase the sedative effects of fluoxetine."),
        ("sertraline", "alcohol", "Alcohol may increase sertraline side effects."),
        ("paroxetine", "alcohol", "Alcohol can increase the sedative effects of paroxetine."),
        ("escitalopram", "alcohol", "Alcohol may increase escitalopram side effects."),
        ("venlafaxine", "alcohol", "Alcohol can increase the sedative effects of venlafaxine."),
        ("bupropion", "alcohol", "Alcohol can increase the risk of seizures with bupropion."),
        
        # MAOIs
        ("phenelzine", "aged cheese", "Aged cheese contains tyramine, which can cause hypertensive crisis with phenelzine."),
        ("tranylcypromine", "red wine", "Red wine contains tyramine, which can cause hypertensive crisis with tranylcypromine."),
        ("isocarboxazid", "sauerkraut", "Sauerkraut contains tyramine, which can cause hypertensive crisis with isocarboxazid."),
        ("selegiline", "aged meats", "Aged meats contain tyramine, which can cause hypertensive crisis with selegiline."),
        
        # Antipsychotics
        ("risperidone", "alcohol", "Alcohol can increase the sedative effects of risperidone."),
        ("quetiapine", "alcohol", "Alcohol can increase the sedative effects of quetiapine."),
        ("olanzapine", "alcohol", "Alcohol can increase the sedative effects of olanzapine."),
        ("aripiprazole", "alcohol", "Alcohol may increase aripiprazole side effects."),
        
        # Benzodiazepines
        ("alprazolam", "alcohol", "Alcohol can increase the sedative effects of alprazolam."),
        ("diazepam", "alcohol", "Alcohol can increase the sedative effects of diazepam."),
        ("lorazepam", "alcohol", "Alcohol can increase the sedative effects of lorazepam."),
        ("clonazepam", "alcohol", "Alcohol can increase the sedative effects of clonazepam."),
        ("temazepam", "alcohol", "Alcohol can increase the sedative effects of temazepam."),
        
        # Stimulants
        ("methylphenidate", "caffeine", "Caffeine may increase the stimulant effects of methylphenidate."),
        ("amphetamine", "caffeine", "Caffeine may increase the stimulant effects of amphetamine."),
        ("dextroamphetamine", "caffeine", "Caffeine may increase the stimulant effects of dextroamphetamine."),
        
        # Opioids
        ("morphine", "alcohol", "Alcohol can increase the sedative effects of morphine."),
        ("oxycodone", "alcohol", "Alcohol can increase the sedative effects of oxycodone."),
        ("hydrocodone", "alcohol", "Alcohol can increase the sedative effects of hydrocodone."),
        ("fentanyl", "alcohol", "Alcohol can increase the sedative effects of fentanyl."),
        ("codeine", "alcohol", "Alcohol can increase the sedative effects of codeine."),
        
        # NSAIDs
        ("ibuprofen", "alcohol", "Alcohol can increase the risk of stomach bleeding with ibuprofen."),
        ("naproxen", "alcohol", "Alcohol can increase the risk of stomach bleeding with naproxen."),
        ("diclofenac", "alcohol", "Alcohol can increase the risk of stomach bleeding with diclofenac."),
        ("celecoxib", "alcohol", "Alcohol can increase the risk of stomach bleeding with celecoxib."),
        
        # Thyroid Medications
        ("levothyroxine", "soy", "Soy products can interfere with the absorption of levothyroxine."),
        ("levothyroxine", "iron supplements", "Iron supplements can interfere with levothyroxine absorption."),
        ("levothyroxine", "fiber", "High-fiber foods can reduce levothyroxine absorption."),
        ("levothyroxine", "coffee", "Coffee can reduce the absorption of levothyroxine."),
        ("levothyroxine", "walnuts", "Walnuts can interfere with levothyroxine absorption."),
        
        # Diabetes Medications
        ("metformin", "alcohol", "Alcohol can increase the risk of lactic acidosis with metformin."),
        ("glipizide", "alcohol", "Alcohol can cause low blood sugar when taking glipizide."),
        ("glyburide", "alcohol", "Alcohol can cause low blood sugar when taking glyburide."),
        ("pioglitazone", "alcohol", "Alcohol can increase the risk of liver problems with pioglitazone."),
        ("sitagliptin", "alcohol", "Alcohol may affect blood sugar control with sitagliptin."),
        
        # Heart Medications
        ("digoxin", "fiber", "High-fiber foods can reduce digoxin absorption."),
        ("digoxin", "licorice", "Licorice can increase potassium loss and affect digoxin."),
        ("amiodarone", "grapefruit", "Grapefruit can increase amiodarone levels and side effects."),
        ("dronedarone", "grapefruit", "Grapefruit can increase dronedarone levels."),
        
        # Blood Pressure Medications
        ("clonidine", "alcohol", "Alcohol can increase the sedative effects of clonidine."),
        ("hydralazine", "alcohol", "Alcohol can increase the effects of hydralazine."),
        ("minoxidil", "alcohol", "Alcohol can increase the effects of minoxidil."),
        
        # Cholesterol Medications
        ("ezetimibe", "grapefruit", "Grapefruit may increase ezetimibe levels."),
        ("fenofibrate", "alcohol", "Alcohol can increase the risk of liver problems with fenofibrate."),
        ("gemfibrozil", "alcohol", "Alcohol can increase the risk of liver problems with gemfibrozil."),
        
        # Anti-seizure Medications
        ("phenytoin", "alcohol", "Alcohol can increase phenytoin side effects."),
        ("carbamazepine", "grapefruit", "Grapefruit can increase carbamazepine levels."),
        ("valproic acid", "alcohol", "Alcohol can increase valproic acid side effects."),
        ("lamotrigine", "alcohol", "Alcohol can increase lamotrigine side effects."),
        ("levetiracetam", "alcohol", "Alcohol can increase levetiracetam side effects."),
        
        # Pain Medications
        ("tramadol", "alcohol", "Alcohol can increase the sedative effects of tramadol."),
        ("acetaminophen", "alcohol", "Alcohol can increase the risk of liver damage with acetaminophen."),
        ("aspirin", "alcohol", "Alcohol can increase the risk of stomach bleeding with aspirin."),
        
        # Anti-anxiety Medications
        ("buspirone", "grapefruit", "Grapefruit may increase buspirone levels."),
        ("hydroxyzine", "alcohol", "Alcohol can increase the sedative effects of hydroxyzine."),
        
        # Sleep Medications
        ("zolpidem", "alcohol", "Alcohol can increase the sedative effects of zolpidem."),
        ("eszopiclone", "alcohol", "Alcohol can increase the sedative effects of eszopiclone."),
        ("zaleplon", "alcohol", "Alcohol can increase the sedative effects of zaleplon."),
        
        # Migraine Medications
        ("sumatriptan", "alcohol", "Alcohol can increase sumatriptan side effects."),
        ("rizatriptan", "alcohol", "Alcohol can increase rizatriptan side effects."),
        ("eletriptan", "grapefruit", "Grapefruit can increase eletriptan levels."),
        
        # Allergy Medications
        ("fexofenadine", "fruit juices", "Fruit juices can reduce fexofenadine absorption."),
        ("cetirizine", "alcohol", "Alcohol can increase the sedative effects of cetirizine."),
        ("loratadine", "alcohol", "Alcohol can increase loratadine side effects."),
        
        # Acid Reflux Medications
        ("omeprazole", "alcohol", "Alcohol can increase omeprazole side effects."),
        ("esomeprazole", "alcohol", "Alcohol can increase esomeprazole side effects."),
        ("lansoprazole", "alcohol", "Alcohol can increase lansoprazole side effects."),
        
        # Additional Common Interactions
        ("metformin", "vitamin B12", "Long-term metformin use can reduce vitamin B12 absorption."),
        ("warfarin", "cranberry juice", "Cranberry juice may increase warfarin effects and bleeding risk."),
        ("levothyroxine", "vitamin D", "Vitamin D can affect thyroid hormone absorption."),
        ("digoxin", "magnesium", "Magnesium supplements can affect digoxin levels."),
        ("lithium", "salt", "Changes in salt intake can affect lithium levels."),
        ("lithium", "caffeine", "Caffeine can affect lithium levels."),
        ("theophylline", "caffeine", "Caffeine can increase theophylline side effects."),
        ("cyclosporine", "grapefruit", "Grapefruit can increase cyclosporine levels."),
        ("tacrolimus", "grapefruit", "Grapefruit can increase tacrolimus levels."),
        ("sirolimus", "grapefruit", "Grapefruit can increase sirolimus levels."),
    ]
    
    return additional_interactions

def update_interactions_file():
    """Update the drug-food interactions CSV file with new data"""
    
    # Generate new interactions
    new_interactions = generate_additional_interactions()
    
    # Create DataFrame for new data - fix the columns parameter
    new_data = pd.DataFrame(new_interactions)
    new_data.columns = ['drug', 'food', 'interaction']
    
    # Read existing data
    try:
        existing_data = pd.read_csv('data/drug_food_interactions.csv')
        print(f"Found existing data with {len(existing_data)} interactions")
    except FileNotFoundError:
        print("No existing data found, creating new file")
        existing_data = pd.DataFrame()
        existing_data['drug'] = []
        existing_data['food'] = []
        existing_data['interaction'] = []
    
    # Combine existing and new data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Remove any duplicates based on drug and food combination
    combined_data = combined_data.drop_duplicates(subset=['drug', 'food'], keep='first')
    
    # Save the updated data
    combined_data.to_csv('data/drug_food_interactions.csv', index=False)
    
    print(f"✅ Added {len(new_data)} new interactions")
    print(f"✅ Total interactions in file: {len(combined_data)}")
    print("✅ File updated successfully!")

if __name__ == "__main__":
    update_interactions_file()
