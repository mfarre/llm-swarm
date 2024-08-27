import json
import sys

# Step 1: Load the JSON files
with open('taxonomy_leaves.json', 'r') as file:
    taxonomy_leaves = json.load(file)

with open(sys.argv[1], 'r') as file:
    results = json.load(file)

# Step 2: Initialize a count for completions containing categories
matching_count = 0

# Function to normalize text for better matching
def normalize_text(text):
    # Convert text to lowercase
    normalized_text = text.lower()
    # Replace dashes and underscores with spaces
    normalized_text = normalized_text.replace('-', ' ').replace('_', ' ')
    return normalized_text

# Normalize all taxonomy leaves once to avoid repeated normalization
normalized_taxonomy_leaves = [normalize_text(leaf) for leaf in taxonomy_leaves]

# Step 3: Process each result and check for matches
for result in results:
    # Normalize the completion text
    completion_text = normalize_text(result.get('completion', ""))

    # Check if any normalized taxonomy leaf is in the normalized completion text
    if any(leaf in completion_text for leaf in normalized_taxonomy_leaves):
        matching_count += 1
    else:
        print(completion_text)

# Step 4: Output the count of matching completions
print(f"Number of completions containing at least one taxonomy leaf: {matching_count / len(results)}")
