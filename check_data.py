#!/usr/bin/env python3
import pyarrow.parquet as pq
import json
import sys

try:
    table = pq.read_table('/mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data/train-00000-of-00135.parquet')
    df = table.to_pandas()
    
    print('Total rows:', len(df))
    print('Columns:', list(df.columns))
    
    print('\nFirst record structure:')
    rec = df.iloc[0].to_dict()
    
    # Show field types
    field_types = {k: type(v).__name__ for k, v in rec.items()}
    print(json.dumps(field_types, indent=2))
    
    print('\nSample reasoning content:')
    reasoning = rec.get('reasoning', None)
    print('Type:', type(reasoning))
    
    if isinstance(reasoning, (list, tuple)):
        print('Length:', len(reasoning))
        if len(reasoning) > 0:
            item = reasoning[0]
            print('First item type:', type(item))
            if isinstance(item, dict):
                print('First item keys:', list(item.keys()))
                if 'solution' in item:
                    solution = str(item.get('solution', ''))
                    print('Has solution:', True)
                    print('Solution contains \\boxed:', '\\boxed' in solution)
                    print('\nSnippet of solution:')
                    print(solution[:500])
    elif reasoning is not None:
        print('Reasoning is not a list/tuple:', reasoning[:200] if hasattr(reasoning, '__getitem__') else reasoning)
    else:
        print('Reasoning is None')
        
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
