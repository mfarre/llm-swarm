import json
import pandas as pd
import os
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
import re
# Ensure the output directory exists
os.makedirs("processed", exist_ok=True)

# Function to load prompts from a single JSON file
def load_prompts_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tasks = json.load(file)
    return tasks

# Function to process a single file's tasks and save results
def process_file(file_path, tokenizer):
    # Load tasks from the current file
    tasks = load_prompts_from_file(file_path)
    results = []
    
    # URL of your model endpoint
    endpoint_url = 'http://ip-26-0-161-153:1456/generate'
    
    # Headers for the HTTP request
    headers = {
        "Content-Type": "application/json",
    }

    # Process each task
    for i, task in enumerate(tqdm(tasks, desc="Processing tasks")):
        video_id = task['video_id']
        input_text = task['prompt']
        input_text = input_text.replace("Given those categories:", "Given this taxonomy:")
        pattern = r"Categories: \[.*?\]\n?"
        input_text = re.sub(pattern, '', input_text)
        pattern = r"Tags: \[.*?\]\n?"
        input_text = re.sub(pattern, '', input_text)
        pattern = r"Description: \[.*?\]\n?"
        input_text = re.sub(pattern, '', input_text)
        input_text = input_text + "RETURN  A CATEGORY FROM THE TAXONOMY PROVIDED: "


        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": input_text},
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare the data for the request
        data = {
            "inputs": prompt_tokens,
            "parameters": {
                "max_new_tokens": 20,  # Adjust as needed
            },
        }


        # Make a synchronous request to the model endpoint
        response = requests.post(endpoint_url, headers=headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            completion = response_data.get('generated_text', '')
        else:
            completion = "Error: Unable to get response"

        # Append the result
        results.append({"video_id": video_id, "completion": completion})

        # Save results to file every 10 results
        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
            output_filename = os.path.splitext(os.path.basename(file_path))[0]
            with open(f"processed/{output_filename}_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

# Main function to process all files
def main():
    # Process each file in the prompts directory
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    for filename in os.listdir("prompts"):
        if filename.endswith(".json"):
            file_path = os.path.join("prompts", filename)
            process_file(file_path, tokenizer)

# Run the main function
if __name__ == "__main__":
    main()