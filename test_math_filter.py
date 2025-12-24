#!/usr/bin/env python3
"""
Test script to validate the is_math_problem function
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocess.filter_math_simple_multiprocess import is_math_problem
import numpy as np

# Create a test record mimicking the real data
test_record_ok = {
    'question': 'What is 2 + 2?',
    'reasoning': np.array([{
        'Cognitive_Difficulty': {'judge': 'QwQ-32B', 'level': 6},
        'Reasoning_Verbosity': {'judge': 'QwQ-32B', 'level': 5},
        'full_response': '<think>\nLet me think about this...\n</think>\nThe answer is 4.'
    }])
}

test_record_no_think = {
    'question': 'What is 2 + 2?',
    'reasoning': np.array([{
        'full_response': 'Just a simple response without think tags.'
    }])
}

print("Testing is_math_problem function:")
print("-" * 60)

# Test 1: Record with think tags
result1 = is_math_problem(test_record_ok, min_reasoning_length=1)
print(f"✓ Record with think tags: {result1} (expected: True)")

# Test 2: Record without proper math indicators
result2 = is_math_problem(test_record_no_think, min_reasoning_length=1)
print(f"✓ Record without think tags: {result2} (expected: False)")

# Test 3: Check length filter
result3 = is_math_problem(test_record_ok, min_reasoning_length=5)
print(f"✓ Record with min_length=5: {result3} (expected: False, actual length=1)")

# Test 4: No length filter
result4 = is_math_problem(test_record_ok, min_reasoning_length=None)
print(f"✓ Record without length filter: {result4} (expected: True)")

# Test with real data
print("\n" + "-" * 60)
print("Testing with real data snippet:")

try:
    import pyarrow.parquet as pq
    table = pq.read_table('/mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data/train-00000-of-00135.parquet')
    df = table.to_pandas()
    
    total_count = 0
    math_count = 0
    
    for i in range(min(100, len(df))):
        rec = df.iloc[i].to_dict()
        total_count += 1
        
        if is_math_problem(rec, min_reasoning_length=1):
            math_count += 1
            if math_count <= 3:  # Show first 3 matches
                print(f"\n  Math problem #{math_count}:")
                print(f"  Question: {str(rec.get('question', ''))[:80]}...")
    
    print(f"\n  Summary: {math_count}/{total_count} records identified as math problems")
    print(f"  Success rate: {math_count/total_count*100:.1f}%")
    
except Exception as e:
    print(f"  Error reading real data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-" * 60)
print("All tests completed!")