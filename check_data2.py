#!/usr/bin/env python3
import pyarrow.parquet as pq
import json

# Read a few records from the parquet file
table = pq.read_table('/mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data/train-00000-of-00135.parquet')
df = table.to_pandas()

print("=" * 60)
print("CHECKING DATA FORMAT")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

boxed_pattern = r'\boxed'

# Check first 5 records
for i in range(min(5, len(df))):
    rec = df.iloc[i].to_dict()
    reasoning = rec.get('reasoning', None)
    
    print(f"\n--- Record {i} ---")
    print(f"Question preview: {str(rec.get('question', ''))[:100]}...")
    print(f"Reasoning type: {type(reasoning)}")
    
    if isinstance(reasoning, (list, tuple)):
        print(f"Reasoning length: {len(reasoning)}")
        if len(reasoning) > 0:
            first_item = reasoning[0]
            print(f"First item type: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"First item keys: {list(first_item.keys())}")
                # Check for solution field
                if 'solution' in first_item:
                    print(f"Has 'solution' field: True")
                    solution = first_item.get('solution', '')
                    print(f"Solution type: {type(solution)}")
                    print(f"Solution preview: {str(solution)[:200]}...")
                    has_boxed = boxed_pattern in str(solution)
                    print(f"Contains \\boxed: {has_boxed}")
                else:
                    print(f"Has 'solution' field: False")
                
                # Check for full_response field (commonly in the data)
                if 'full_response' in first_item:
                    full_response = first_item.get('full_response', '')
                    print(f"Has 'full_response' field: True")
                    print(f"Full response preview: {str(full_response)[:200]}...")
                    has_boxed = boxed_pattern in str(full_response)
                    print(f"Full response contains \\boxed: {has_boxed}")
                    print(f"Full response has '</think>' tag: {'</think>' in str(full_response)}")
                    # Check if there's a final answer pattern
                    if has_boxed:
                        print("Found \\boxed{- pattern!")
    elif reasoning is None:
        print("Reasoning is None")
    else:
        print(f"Reasoning raw: {str(reasoning)[:200]}...")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Check all records for solution and boxed patterns
total_with_solution = 0
total_with_boxed = 0
sample_count = min(100, len(df))

print(f"Sampling {sample_count} records...")
for i in range(sample_count):
    rec = df.iloc[i].to_dict()
    reasoning = rec.get('reasoning', [])
    
    if isinstance(reasoning, (list, tuple)) and len(reasoning) > 0:
        item = reasoning[0]
        if isinstance(item, dict):
            if 'solution' in item and boxed_pattern in str(item['solution']):
                total_with_solution += 1
            elif 'full_response' in item and boxed_pattern in str(item['full_response']):
                total_with_boxed += 1

print(f"Records with \\boxed in 'solution': {total_with_solution}")
print(f"Records with \\boxed in 'full_response': {total_with_boxed}")
print(f"Total records checked: {sample_count}")