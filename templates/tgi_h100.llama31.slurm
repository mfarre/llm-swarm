#!/bin/bash
#SBATCH --job-name=llm-swarm
#SBATCH --partition hopper-prod
#SBATCH --gpus={{gpus}}
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs/%x_%j.out

# For HF cluster internal users: Check if /fsx directory exists
if [ -d "/fsx/miquel/llm-swarm/cache" ]; then
    export volume="/fsx/miquel/llm-swarm/cache"
else
    export volume=".cache"
fi
export model="meta-llama/Meta-Llama-3.1-70B-Instruct"
export revision="33101ce6ccc08fa6249c10a543ebfcac65173393"

function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export PORT=$(unused_port)
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    # try reading from file
    export HUGGING_FACE_HUB_TOKEN=$(cat /fsx/miquel/cache/token)
fi
echo "Starting TGI container port $PORT"
echo "http://$(hostname -I | awk '{print $1}'):$PORT" >> {{slurm_hosts_path}}
# unset cache dirs to avoid pyxis having host env var somehow get into the container
unset HF_HUB_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE
srun --container-image='ghcr.io#huggingface/text-generation-inference' \
    --container-env=HUGGING_FACE_HUB_TOKEN,PORT \
    --container-mounts="$volume:/data" \
    --no-container-mount-home \
    --qos normal \
    /usr/local/bin/text-generation-launcher \
    --model-id $model \
    --revision $revision \
    --max-concurrent-requests 2000 \
    --max-total-tokens 8000 \
    --max-input-length 7600 \
    --max-batch-prefill-tokens 8000 \

echo "End of job"
