CONFIG=$1
CURRENT_DIR=$(pwd)

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

export CUDA_VISIBLE_DEVICES=4,5

python update_index.py $CURRENT_DIR/$CONFIG