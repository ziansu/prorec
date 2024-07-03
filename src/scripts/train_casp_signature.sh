CONFIG=$1
CURRENT_DIR=$(pwd)

WANDB_org=YOUR_PROJECT_ORG
WANDB_project=casp_sig

# Check if the CONFIG file is provided
if [ -z "$CONFIG" ]; then
    echo "Please provide a config file"
    exit 1
fi

# Check if the CONFIG file exists
if [ ! -f "$CURRENT_DIR/$CONFIG" ]; then
    echo "Config file $CURRENT_DIR/$CONFIG does not exist"
    exit 1
fi

echo "Config: $CONFIG"


cd ../

# HUGGINGFACE_TOKEN=
# huggingface-cli login --token $HUGGINGFACE_TOKEN

# export CUDA_VISIBLE_DEVICES=6,7

WANDB_ENTITY=$WANDB_org \
WANDB_PROJECT=$WANDB_project \
torchrun --nproc_per_node=2 run_casp_signature.py $CURRENT_DIR/$CONFIG