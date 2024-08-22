import asyncio
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

tasks = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the formula for water?"]

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
    # Update the model to Llama 3.1 70B
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    
    # Update the tokenizer to use with Llama 3.1 70B
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    # Adjust special tokens if necessary (Llama models often have specific tokenization)
    tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})
    async def process_text(task):
        # Adjust prompt formatting if needed for Llama
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task},
            ],
            tokenize=False,
        )
        return await client.text_generation(
            prompt=prompt,
            max_new_tokens=200,
        )

    async def main():
        # Process tasks with progress tracking
        results = await tqdm_asyncio.gather(*(process_text(task) for task in tasks))
        df = pd.DataFrame({"Task": tasks, "Completion": results})
        print(df)

    # Run the async main function
    asyncio.run(main())
