import asyncio
import json
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

# Load the prompts from the JSON file
with open("prompts.json", "r") as f:
    tasks = json.load(f)

# Configure and start the LLM-Swarm with Llama 3.1 70B
with LLMSwarm(
    LLMSwarmConfig(
        instances=1,
        inference_engine="tgi",
        slurm_template_path="../templates/tgi_h100.template.slurm",  # Ensure the template matches your setup
        load_balancer_template_path="../templates/nginx.template.conf",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
) as llm_swarm:
    # Initialize the client with the model endpoint
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    
    # Load the tokenizer for Llama 3.1 70B
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    # Adjust special tokens if necessary (Llama models often have specific tokenization)
    tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})

    async def process_task(task):
        video_id = task['video_id']
        prompt = task['prompt']
        
        # Tokenize and prepare the prompt
        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
        )
        
        # Send the prompt to the model and get the response
        response = await client.text_generation(
            prompt=prompt_tokens,
            max_new_tokens=200,
        )
        
        return {"video_id": video_id, "completion": response}

    async def main():
        # Process all tasks with progress tracking
        results = await tqdm_asyncio.gather(*(process_task(task) for task in tasks))
        
        # Convert results to DataFrame and save to a pickle file
        df = pd.DataFrame(results)
        print(df)
        df.to_pickle("results.pkl")

        # Optionally, save to a JSON file instead of or in addition to pickle
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    # Run the async main function
    asyncio.run(main())
