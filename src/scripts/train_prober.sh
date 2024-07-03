CONFIG=$1
CURRENT_DIR=$(pwd)

WANDB_org=YOUR_PROJECT_ORG
WANDB_project=src_prober

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

rand=$(( RANDOM % 10000 ))
MASTER_PORT=$(( 40000 + rand ))

export CUDA_VISIBLE_DEVICES=1,2,3,4

WANDB_ENTITY=$WANDB_org \
WANDB_PROJECT=$WANDB_project \
torchrun --nproc_per_node=4  --master_port=$MASTER_PORT run_prober.py $CURRENT_DIR/$CONFIG

# CUDA_LAUCH_BLOCKING=1 \
# WANDB_ENTITY=$WANDB_org \
# WANDB_PROJECT=$WANDB_project \
# python run_prober.py $CURRENT_DIR/$CONFIG