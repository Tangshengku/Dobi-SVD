import torch
from datasets import Dataset, concatenate_datasets, load_dataset
import random
import re
from tqdm import tqdm


MIXTURE_DATASETS = {
    "wikitext2_evol_codealpaca_tulu_math": {
        "wikitext2": ("wikitext", "wikitext-2-raw-v1", "train", "test"),
        "evol_codealpaca": ("theblackcat102/evol-codealpaca-v1", None, "train", None),
        "tulu_math": ("allenai/tulu-3-sft-personas-math", None, "train", None),
    },
}


def _text_from_example(example):
    if "text" in example and example["text"]:
        return example["text"]
    if "messages" in example and example["messages"]:
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in example["messages"]
            if message.get("content")
        )
    if "conversations" in example and example["conversations"]:
        return "\n".join(
            turn.get("value") or turn.get("content", "")
            for turn in example["conversations"]
            if turn.get("value") or turn.get("content")
        )
    instruction = example.get("instruction") or example.get("prompt") or example.get("question") or ""
    input_text = example.get("input") or ""
    output = example.get("output") or example.get("response") or example.get("answer") or ""
    return "\n".join(part for part in (instruction, input_text, output) if part)


def _load_hf_dataset(path, config, split, cache_dir):
    if config:
        return load_dataset(path, config, split=split, cache_dir=cache_dir)
    return load_dataset(path, split=split, cache_dir=cache_dir)


def _load_mixture_dataset(dataset_cache_dir, seed, n_train_samples):
    spec = MIXTURE_DATASETS["wikitext2_evol_codealpaca_tulu_math"]
    wiki_path, wiki_config, wiki_train_split, wiki_val_split = spec["wikitext2"]
    wiki_train = _load_hf_dataset(wiki_path, wiki_config, wiki_train_split, dataset_cache_dir)
    wiki_val = _load_hf_dataset(wiki_path, wiki_config, wiki_val_split, dataset_cache_dir)

    evol_path, evol_config, evol_split, _ = spec["evol_codealpaca"]
    evol_train = _load_hf_dataset(evol_path, evol_config, evol_split, dataset_cache_dir)

    tulu_path, tulu_config, tulu_split, _ = spec["tulu_math"]
    tulu_train = _load_hf_dataset(tulu_path, tulu_config, tulu_split, dataset_cache_dir)

    def to_text_dataset(dataset, prefix, max_rows=None):
        has_math_source = "source" in dataset.column_names or "dataset" in dataset.column_names
        texts = []
        if max_rows is not None:
            pool_size = max_rows * 20 if has_math_source else max_rows
            dataset = dataset.shuffle(seed=seed).select(range(min(pool_size, len(dataset))))
        for row in tqdm(dataset, desc=f"Formatting {prefix}", unit=" row"):
            if has_math_source and prefix == "tulu-math":
                source = str(row.get("source", row.get("dataset", ""))).lower()
                if "math" not in source:
                    continue
            text = _text_from_example(row)
            if text:
                texts.append(text)
            if max_rows is not None and len(texts) >= max_rows:
                break
        return Dataset.from_dict({"text": texts})

    train_rows_per_source = max(n_train_samples * 8, 1024)
    train_parts = [
        to_text_dataset(wiki_train, "wikitext2", train_rows_per_source),
        to_text_dataset(evol_train, "evol-codealpaca", train_rows_per_source),
        to_text_dataset(tulu_train, "tulu-math", train_rows_per_source),
    ]
    valdata = to_text_dataset(wiki_val, "wikitext2 validation")
    traindata = concatenate_datasets(train_parts).shuffle(seed=seed)
    return traindata, valdata


def _safe_cache_key(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _tokenizer_cache_key(tokenizer, args):
    model_key = getattr(args, "model_id", None) or getattr(args, "updated_model_path", None)
    tokenizer_key = getattr(tokenizer, "name_or_path", None) or model_key or tokenizer.__class__.__name__
    vocab_size = len(tokenizer)
    return _safe_cache_key(f"{tokenizer_key}_{tokenizer.__class__.__name__}_v{vocab_size}")


def _tokenized_data_is_valid(tokenized_data, tokenizer):
    if not tokenized_data:
        return False
    vocab_size = len(tokenizer)
    for sample in tokenized_data[: min(8, len(tokenized_data))]:
        input_ids = sample["input_ids"]
        if input_ids.numel() == 0:
            return False
        if input_ids.min().item() < 0 or input_ids.max().item() >= vocab_size:
            return False
    return True


def prepare_train_loaders(tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args):
    traindata_cache_file = dataset_cache_dir / f"traindata.pt"
    valdata_cache_file = dataset_cache_dir / f"valdata.pt"
    LOAD = traindata_cache_file.exists() and valdata_cache_file.exists()
    SEQ_LEN = args.seq_len
    NSAMPLES_train = args.n_train_samples
    NSAMPLES_val = args.n_eval_samples
    SAVE = args.SAVE
    SEED = args.seed
    
    if LOAD and not args.RECREATE:
        traindata = torch.load(traindata_cache_file)
        valdata = torch.load(valdata_cache_file)
    else:
        if DATASET_NAME == 'c4':
            traindata = load_dataset("json", 
                                      data_files={"train": str(dataset_cache_dir / "en/c4-train.00000-of-01024.json.gz")},
                                      cache_dir=dataset_cache_dir,
                                      split="train")
            valdata = load_dataset("json", 
                                    data_files={"validation": str(dataset_cache_dir /"en/c4-validation.00000-of-00008.json.gz")},
                                    cache_dir=dataset_cache_dir,
                                    split="validation")
        elif DATASET_NAME == 'wikitext2':
            traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
            valdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)
        elif DATASET_NAME == 'ptb':
            traindata = load_dataset("ptb_text_only", "penn_treebank", split="train", cache_dir=dataset_cache_dir)
            valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation", cache_dir=dataset_cache_dir)
        elif DATASET_NAME == "wikitext2_evol_codealpaca_tulu_math":
            traindata, valdata = _load_mixture_dataset(dataset_cache_dir, SEED, NSAMPLES_train)
        else:
            raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
        if SAVE:
            torch.save(traindata, traindata_cache_file)
            torch.save(valdata, valdata_cache_file)
            print("Training and validation data has been processed and saved!")
    
    print("Training and validation has been loaded!")
    
    
    tokenizer_key = _tokenizer_cache_key(tokenizer, args)
    tokenized_traindata_cache_file = data_cache_dir / f"traindata_{DATASET_NAME}_{tokenizer_key}_{NSAMPLES_train}_{SEQ_LEN}.pt"
    tokenized_valdata_cache_file = data_cache_dir / f"valdata_{DATASET_NAME}_{tokenizer_key}_{NSAMPLES_val}_{SEQ_LEN}.pt"
    LOAD = tokenized_traindata_cache_file.exists() and tokenized_valdata_cache_file.exists()
    
    if LOAD and not args.RECREATE:
        tokenized_traindata = torch.load(tokenized_traindata_cache_file)
        tokenized_valdata = torch.load(tokenized_valdata_cache_file)
        if not _tokenized_data_is_valid(tokenized_traindata, tokenizer) or not _tokenized_data_is_valid(tokenized_valdata, tokenizer):
            print("Cached tokenized data is invalid for this tokenizer; regenerating.")
            LOAD = False
    if LOAD and not args.RECREATE:
        pass
    else:
        # traindata
        tokenized_traindata = []
        tokenized_valdata = []
        if DATASET_NAME == 'c4':
            random.seed(SEED)
            for _ in tqdm(range(NSAMPLES_train), desc="Processing training data", unit=" sample"):
                while True:
                    i = random.randint(0, len(traindata) - 1)
                    trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                    if trainenc.input_ids.shape[1] >= SEQ_LEN:
                        break
                if trainenc.input_ids.shape[1] - SEQ_LEN - 1 < 0:
                    i = 0
                else:
                    i = random.randint(0, trainenc.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = trainenc.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_traindata.append({"input_ids": inp, "attention_mask": attention_mask})
                
            # val
            random.seed(0)
            for _ in tqdm(range(NSAMPLES_val), desc="Processing validate data", unit=" sample"):
                while True:
                    i = random.randint(0, len(valdata) - 1)
                    tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                    if tmp.input_ids.shape[1] >= SEQ_LEN:
                        break
                if tmp.input_ids.shape[1] - SEQ_LEN - 1 <= 0:
                    i = 0
                else:
                    i = random.randint(0, tmp.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = tmp.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_valdata.append({"input_ids": inp, "attention_mask": attention_mask})

        
        elif DATASET_NAME in {'wikitext2', 'ptb', "wikitext2_evol_codealpaca_tulu_math"}:
            if DATASET_NAME in {'wikitext2', "wikitext2_evol_codealpaca_tulu_math"}:
                train_tot_text = "\n\n".join(traindata["text"])
                val_tot_text = "\n\n".join(valdata["text"])
            elif DATASET_NAME == 'ptb':
                train_tot_text = "\n\n".join(traindata["sentence"])
                val_tot_text = "\n\n".join(valdata["sentence"])
            # train
            random.seed(SEED)
            for s in tqdm(range(NSAMPLES_train), desc="Processing training data", unit=" sample"):
                i = random.randint(0, len(train_tot_text) - SEQ_LEN - 1)
                j = i + SEQ_LEN * 10
                trainenc = tokenizer(train_tot_text[i:j], return_tensors="pt")
                if trainenc.input_ids.shape[1] < SEQ_LEN:
                    s = s - 1
                    continue
                if trainenc.input_ids.shape[1] - SEQ_LEN - 1 < 0:
                    i = 0
                else:
                    i = random.randint(0, trainenc.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = trainenc.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_traindata.append({"input_ids": inp, "attention_mask": attention_mask})
            
            # val
            random.seed(0)
            for s in tqdm(range(NSAMPLES_val), desc="Processing validation data", unit=" sample"):
                i = random.randint(0, len(val_tot_text) - SEQ_LEN - 1)
                j = i + SEQ_LEN * 10
                trainenc = tokenizer(val_tot_text[i:j], return_tensors="pt")
                if trainenc.input_ids.shape[1] < SEQ_LEN:
                    s = s - 1
                    continue
                if trainenc.input_ids.shape[1] - SEQ_LEN - 1 < 0:
                    i = 0
                else:
                    i = random.randint(0, trainenc.input_ids.shape[1] - SEQ_LEN - 1)
                j = i + SEQ_LEN
                inp = trainenc.input_ids[:, i:j]
                assert inp.dim() == 2
                inp = inp.squeeze(0)
                assert inp.dim() == 1
                attention_mask = torch.ones_like(inp)
                tokenized_valdata.append({"input_ids": inp, "attention_mask": attention_mask})
            
        if SAVE:
            torch.save(tokenized_traindata, tokenized_traindata_cache_file)
            torch.save(tokenized_valdata, tokenized_valdata_cache_file)
            print("Tokenized data has been processed and saved!")
    
    print("Tokenized data has been loaded!")
    
    return  tokenized_traindata, tokenized_valdata
