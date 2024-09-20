import datasets
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
dataset = datasets.load_dataset("cerebras/SlimPajama-627B", streaming=True)

# This should really be done in parallel, but the streaming dataset doesn't support that and it's not worth implementing myself.
def tokenize_split(dataset, split, total_tokens):
    source_map = {}
    data_file = np.memmap(f'./data_{split}.bin', dtype=np.uint32, mode='w+', shape=total_tokens)
    with tqdm(total=total_tokens) as bar:
        idx = 0
        for data in iter(dataset[split]):
            tokens = tokenizer.encode(data['text'])
            l = len(tokens)
            if idx + l > total_tokens:
                l = total_tokens - idx
            data_file[idx:idx+l] = tokens[:l]
            idx += l
            bar.update(l)
            source = data['meta']['redpajama_set_name']
            if source in source_map:
                source_map[source] += 1
            else:
                source_map[source] = 1
            if idx >= total_tokens:
                break
    data_file.flush()
    print(source_map)

train_tokens = 500_000_000
tokenize_split(dataset, 'train', train_tokens)
tokenize_split(dataset, 'validation', train_tokens // 500)