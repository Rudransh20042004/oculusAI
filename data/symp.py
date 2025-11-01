import csv
from pathlib import Path

# Define symptoms for each disease (1 = present, 0 = absent)
DISEASE_SYMPTOMS = {
    'Uveitis': {
        'eye_redness': 1,
        'eye_pain': 1,
        'blurred_vision': 1,
        'sensitivity_to_light': 1,
        'floating_spots': 1,
        'itching': 0,
        'tearing': 0,
        'discharge': 0,
        'crusting_eyelids': 0,
        'cloudy_vision': 0,
        'difficulty_seeing_at_night': 0,
        'sensitivity_to_glare': 0,
        'drooping_eyelids': 0,
        'swelling': 0,
        'irritation': 0,
        'lumps_on_eyelids': 0
    },
    'Conjunctivitis': {
        'eye_redness': 1,
        'eye_pain': 0,
        'blurred_vision': 0,
        'sensitivity_to_light': 0,
        'floating_spots': 0,
        'itching': 1,
        'tearing': 1,
        'discharge': 1,
        'crusting_eyelids': 1,
        'cloudy_vision': 0,
        'difficulty_seeing_at_night': 0,
        'sensitivity_to_glare': 0,
        'drooping_eyelids': 0,
        'swelling': 0,
        'irritation': 0,
        'lumps_on_eyelids': 0
    },
    'Cataract': {
        'eye_redness': 0,
        'eye_pain': 0,
        'blurred_vision': 1,
        'sensitivity_to_light': 0,
        'floating_spots': 0,
        'itching': 0,
        'tearing': 0,
        'discharge': 0,
        'crusting_eyelids': 0,
        'cloudy_vision': 1,
        'difficulty_seeing_at_night': 1,
        'sensitivity_to_glare': 1,
        'drooping_eyelids': 0,
        'swelling': 0,
        'irritation': 0,
        'lumps_on_eyelids': 0
    },
    'Eyelid': {  # Eyelid Drooping
        'eye_redness': 0,
        'eye_pain': 0,
        'blurred_vision': 0,
        'sensitivity_to_light': 0,
        'floating_spots': 0,
        'itching': 0,
        'tearing': 0,
        'discharge': 0,
        'crusting_eyelids': 0,
        'cloudy_vision': 0,
        'difficulty_seeing_at_night': 0,
        'sensitivity_to_glare': 0,
        'drooping_eyelids': 1,
        'swelling': 1,
        'irritation': 1,
        'lumps_on_eyelids': 1
    },
    'Normal': {
        'eye_redness': 0,
        'eye_pain': 0,
        'blurred_vision': 0,
        'sensitivity_to_light': 0,
        'floating_spots': 0,
        'itching': 0,
        'tearing': 0,
        'discharge': 0,
        'crusting_eyelids': 0,
        'cloudy_vision': 0,
        'difficulty_seeing_at_night': 0,
        'sensitivity_to_glare': 0,
        'drooping_eyelids': 0,
        'swelling': 0,
        'irritation': 0,
        'lumps_on_eyelids': 0
    }
}

# Symptom columns in order
SYMPTOM_COLUMNS = [
    'eye_redness',
    'eye_pain',
    'blurred_vision',
    'sensitivity_to_light',
    'floating_spots',
    'itching',
    'tearing',
    'discharge',
    'crusting_eyelids',
    'cloudy_vision',
    'difficulty_seeing_at_night',
    'sensitivity_to_glare',
    'drooping_eyelids',
    'swelling',
    'irritation',
    'lumps_on_eyelids'
]

def add_symptoms_to_csv(input_csv, output_csv=None):
    """
    Adds symptom columns to existing image labels CSV.
    
    Args:
        input_csv: Path to the existing CSV file
        output_csv: Path for the output CSV (if None, creates 'input_with_symptoms.csv')
    """
    
    input_path = Path(input_csv)
    
    if not input_path.exists():
        print(f"Error: File '{input_csv}' does not exist!")
        return
    
    # Determine output path
    if output_csv is None:
        output_path = input_path.parent / f"{input_path.stem}_with_symptoms.csv"
    else:
        output_path = Path(output_csv)
    
    print("=" * 70)
    print("ADDING SYMPTOMS TO CSV")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Read input CSV and add symptoms
    updated_rows = []
    disease_counts = {}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        
        # New fieldnames with symptoms
        new_fieldnames = list(original_fieldnames) + SYMPTOM_COLUMNS
        
        for row in reader:
            disease = row['disease_label']
            
            # Count diseases
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            # Get symptoms for this disease
            if disease in DISEASE_SYMPTOMS:
                symptoms = DISEASE_SYMPTOMS[disease]
            else:
                print(f"Warning: Unknown disease '{disease}' - using all 0s")
                symptoms = {col: 0 for col in SYMPTOM_COLUMNS}
            
            # Add symptom values to row
            for symptom_col in SYMPTOM_COLUMNS:
                row[symptom_col] = symptoms[symptom_col]
            
            updated_rows.append(row)
    
    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    print("=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print(f"\nTotal images processed: {len(updated_rows)}")
    print("\nBreakdown by disease:")
    for disease, count in sorted(disease_counts.items()):
        print(f"  {disease}: {count} images")
    
    print("\nSymptom columns added:")
    for i, symptom in enumerate(SYMPTOM_COLUMNS, 1):
        print(f"  {i}. {symptom}")
    
    print(f"\n✓ New CSV saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("EYE CONDITION CSV - ADD SYMPTOMS")
    print("=" * 70)
    
    # Get input CSV path
    input_csv = input("\nEnter the path to your existing CSV file: ").strip()
    
    if not Path(input_csv).exists():
        print(f"Error: File '{input_csv}' does not exist!")
        exit()
    
    # Ask about output filename
    print("\nOutput options:")
    print("  1. Auto-generate filename (adds '_with_symptoms' to original name)")
    print("  2. Specify custom output filename")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    output_csv = None
    if choice == "2":
        output_csv = input("Enter output CSV filename: ").strip()
    
    print()
    
    # Add symptoms to CSV
    add_symptoms_to_csv(input_csv, output_csv)