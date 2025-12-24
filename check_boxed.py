#!/usr/bin/env python3
import pyarrow.parquet as pq
import re

# Check multiple files for \boxed patterns
data_dir = '/mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data'

# Check first 5 files
files_to_check = [
    f'{data_dir}/train-00000-of-00135.parquet',
    f'{data_dir}/train-00001-of-00135.parquet',
    f'{data_dir}/train-00002-of-00135.parquet',
    f'{data_dir}/train-00003-of-00135.parquet',
    f'{data_dir}/train-00010-of-00135.parquet',
]

boxed_pattern = r'\\boxed'
total_checked = 0
total_with_solution = 0
total_with_boxed = 0

for file_path in files_to_check:
    try:
        print(f"\nChecking: {file_path}")
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        found_in_file = False
        file_count = 0
        file_boxed_count = 0
        
        # Check up to 100 records per file
        for i in range(min(100, len(df))):
            total_checked += 1
            rec = df.iloc[i].to_dict()
            reasoning = rec.get('reasoning', [])
            
            if isinstance(reasoning, (list, tuple)) and len(reasoning) > 0:
                item = reasoning[0]
                if isinstance(item, dict):
                    if 'solution' in item and re.search(boxed_pattern, str(item['solution'])):
                        total_with_solution += 1
                        found_in_file = True
                        file_count += 1
                    elif 'full_response' in item:
                        full_response = str(item.get('full_response', ''))
                        if re.search(boxed_pattern, full_response):
                            total_with_boxed += 1
                            found_in_file = True
                            file_boxed_count += 1
        
        print(f"  Records with \\boxed in 'solution': {file_count}")
        print(f"  Records with \\boxed in 'full_response': {file_boxed_count}")
        
        # If we found any, show an example
        if found_in_file:
            print(f"  ✓ Found \\boxed patterns in this file!")
        else:
            print(f"  ✗ No \\boxed patterns found in sampled records")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total records checked: {total_checked}")
print(f"Records with \\boxed in 'solution': {total_with_solution}")
print(f"Records with \\boxed in 'full_response': {total_with_boxed}")
print(f"Percentage with any \\boxed: {(total_with_solution + total_with_boxed) / total_checked * 100:.2f}%")

if total_with_solution + total_with_boxed == 0:
    print("\n⚠️  WARNING: No \\boxed patterns found in any sampled data!")
    print("This means the filter script will always return 0 results.")
    print("\nThe data likely needs different filtering criteria.")