import pandas as pd
import os
from collections import defaultdict

def analyze_csv_structure(file_path):
    """Analyze the structure of a single CSV file"""
    print(f"\nAnalyzing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Initialize counters and storage
    line_types = defaultdict(int)
    field_counts = defaultdict(int)
    sample_lines = {}
    headers = set()
    
    # Store all unique values for each field position
    field_values = defaultdict(set)
    
    current_race = None
    race_count = 0
    horse_count = 0
    
    print("\nAnalyzing line by line...")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        fields = [f.strip() for f in line.split(';')]
        field_count = len(fields)
        field_counts[field_count] += 1
        
        # Store unique values for each field
        for j, field in enumerate(fields):
            if field:  # Only store non-empty values
                field_values[j].add(field)
        
        # Try to identify line type
        if 'Kosu' in line:
            line_types['race_header'] += 1
            race_count += 1
            if 'race_header' not in sample_lines:
                sample_lines['race_header'] = (fields, i+1)
            current_race = race_count
            
            # Extract headers
            for field in fields:
                if field and not field.isdigit():
                    headers.add(field)
                    
        elif current_race and fields[0].isdigit():
            line_types['horse_result'] += 1
            horse_count += 1
            if 'horse_result' not in sample_lines:
                sample_lines['horse_result'] = (fields, i+1)
        else:
            line_types['other'] += 1
            if 'other' not in sample_lines:
                sample_lines['other'] = (fields, i+1)
    
    # Print analysis
    print("\n=== File Structure Analysis ===")
    print(f"\nTotal lines processed: {sum(line_types.values())}")
    
    print("\nLine type distribution:")
    for type_name, count in line_types.items():
        print(f"- {type_name}: {count}")
    
    print("\nField count distribution:")
    for count, freq in sorted(field_counts.items()):
        print(f"- {count} fields: {freq} lines")
    
    print("\nRace statistics:")
    print(f"- Total races: {race_count}")
    print(f"- Total horses: {horse_count}")
    print(f"- Average horses per race: {horse_count/race_count:.2f}")
    
    print("\nIdentified headers:")
    for header in sorted(headers):
        print(f"- {header}")
    
    print("\nSample lines with field analysis:")
    for line_type, (fields, line_num) in sample_lines.items():
        print(f"\n{line_type} (line {line_num}):")
        for i, field in enumerate(fields):
            if field:  # Only show non-empty fields
                unique_values = field_values[i]
                value_count = len(unique_values)
                sample_values = list(unique_values)[:5]  # Show up to 5 example values
                print(f"  Field {i}: {field}")
                print(f"    - Unique values: {value_count}")
                print(f"    - Sample values: {', '.join(str(v) for v in sample_values)}")
                if value_count <= 10:  # If few unique values, show all of them
                    print(f"    - All values: {', '.join(str(v) for v in sorted(unique_values))}")

def main():
    # Find a sample CSV file
    for root, dirs, files in os.walk('data/raw'):
        for file in files:
            if file.endswith('.csv'):
                analyze_csv_structure(os.path.join(root, file))
                print(f"\nAnalyzed file: {file}")
                user_input = input("\nAnalyze another file? (y/n): ")
                if user_input.lower() != 'y':
                    return

if __name__ == "__main__":
    main() 