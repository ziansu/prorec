import pickle
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os

index_dir = '../save/index'


with open(f"{index_dir}/keys.pkl", 'rb') as f:
    dup_keys = pickle.load(f)
dup_key_embeddings = np.load(f"{index_dir}/key_embeddings.npy")

dedup_keys = OrderedDict()
dedup_key_embeddings = []
for i, key in tqdm(enumerate(dup_keys)):
    if key not in dedup_keys:
        dedup_keys[key] = None
        dedup_key_embeddings.append(dup_key_embeddings[i])

dedup_key_embeddings = np.array(dedup_key_embeddings)

print(len(dedup_keys.keys()))

# if not exist index_dir_dedup, create it
if not os.path.exists(f"{index_dir}_dedup"):
    os.makedirs(f"{index_dir}_dedup")

with open(f"{index_dir}_dedup/keys.pkl", 'wb') as f:
    pickle.dump(list(dedup_keys.keys()), f)
np.save(f"{index_dir}_dedup/key_embeddings.npy", dedup_key_embeddings)