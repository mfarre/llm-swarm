import asyncio
import json
import pandas as pd
import os
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, AddedToken

# Ensure the output directory exists
os.makedirs("processed", exist_ok=True)

# Function to load prompts from a single JSON file
def load_prompts_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tasks = json.load(file)
    return tasks

# Function to process a single file's tasks and save results
async def process_file(file_path, client, tokenizer):
    # Load tasks from the current file
    tasks = load_prompts_from_file(file_path)

    async def process_task(task):
        video_id = task['video_id']
        prompt = task['prompt']

        # Tokenize and prepare the prompt
        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        # Send the prompt to the model and get the response
        response = await client.text_generation(
            prompt=prompt_tokens,
            max_new_tokens=50
        )

        return {"video_id": video_id, "completion": response}

    # Process all tasks in the current file with progress tracking
    results = await tqdm_asyncio.gather(*(process_task(task) for task in tasks))

    # Convert results to DataFrame and save to a pickle file
    df = pd.DataFrame(results)
    output_filename = os.path.splitext(os.path.basename(file_path))[0]

    with open(f"processed/{output_filename}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Configure and start the LLM-Swarm with Llama 3.1 70B
with LLMSwarm(
    LLMSwarmConfig(
        instances=1,
        inference_engine="tgi",
        slurm_template_path="../templates/tgi_h100.template.slurm",  # Ensure the template matches your setup
        load_balancer_template_path="../templates/nginx.template.conf",
#        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
) as llm_swarm:
    # Initialize the client with the model endpoint
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    
    # Load the tokenizer for Llama 3.1 70B
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # Given this issue: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/discussions/10
    # we apply the configuration settings to the tokenizer
    # config_path = "tokenizer_config.json"
    # with open(config_path, "r", encoding="utf-8") as f:
    #     tokenizer_config = json.load(f)

    # tokenizer.bos_token = tokenizer_config["bos_token"]
    # tokenizer.eos_token = tokenizer_config["eos_token"]
    # chat_template = tokenizer_config["chat_template"]
    # tokenizer.model_input_names = tokenizer_config["model_input_names"]
    # tokenizer.model_max_length = tokenizer_config["model_max_length"]

    # if "clean_up_tokenization_spaces" in tokenizer_config:
    #     tokenizer.clean_up_tokenization_spaces = tokenizer_config["clean_up_tokenization_spaces"]

    # for token_id, token_attributes in tokenizer_config["added_tokens_decoder"].items():
    #     # Create an AddedToken object with the token's attributes
    #     added_token = AddedToken(
    #         token_attributes["content"],
    #         lstrip=token_attributes.get("lstrip", False),
    #         rstrip=token_attributes.get("rstrip", False),
    #         single_word=token_attributes.get("single_word", False),
    #         normalized=token_attributes.get("normalized", False)
    #     )
    #     # Add the AddedToken to the tokenizer
    #     tokenizer.add_tokens([added_token], special_tokens=token_attributes.get("special", False))


    # Main async function to process all files one by one
    async def main():
        # Process each file in the prompts directory
        for filename in os.listdir("prompts"):
            if filename.endswith(".json"):
                file_path = os.path.join("prompts", filename)
                await process_file(file_path, client, tokenizer)

    # Run the async main function
    asyncio.run(main())
