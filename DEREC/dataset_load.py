import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
CLASSIFIER_MODEL_NAME = 'deberta'

# Dataset configurations
DATASET_CONFIGS = {
    'LIAR-RAW': {
        'num_labels': 6,
        'label_map': {
            "pants-fire": 0,
            "false": 1,
            "barely-true": 2,
            "half-true": 3,
            "mostly-true": 4,
            "true": 5
        },
        'file_pattern': '{split}.json'
    },
    'RAWFC': {
        'num_labels': 3,
        'label_map': {
            "false": 0,
            "half": 1,
            "true": 2
        },
        'file_pattern': '{split}'
    },
    'DRAGON': {
        'num_labels': 2,
        'label_map': {False: 0, True: 1},  # ungrounded=0, grounded=1
        'file_pattern': '{split}.json'
    }
}

class DatasetReader:
    @staticmethod
    def read_dataset(dataset_name: str, dataset_path: str, split: str):
        if dataset_name == 'LIAR-RAW':
            return DatasetReader._read_liar_raw(dataset_path, split)
        elif dataset_name == 'RAWFC':
            return DatasetReader._read_rawfc(dataset_path, split)
        elif dataset_name == 'DRAGON':
            return DatasetReader._read_dragon(dataset_path, split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    @staticmethod
    def _read_liar_raw(base_path: str, split: str):
        file_path = os.path.join(base_path, f"{split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in {file_path}: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {str(e)}")
            raise
        
    @staticmethod
    def _read_rawfc(base_path: str, split: str):
        split_path = os.path.join(base_path, split)
        all_data = []

        try:
            for filename in os.listdir(split_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(split_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
        except Exception as e:
            print(f"Error reading RAWFC directory {split_path}: {e}")

        return all_data

    @staticmethod
    def _read_dragon(base_path: str, split: str):
        file_path = os.path.join(base_path, f"{split}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        dataset = []
        label_map = DATASET_CONFIGS['DRAGON']['label_map']
        for record in raw_data:
            claim = f"question: {record['question']}\nanswer: {record['model_answer']}"
            evidence = "\n".join(record['evidence_texts'])
            label = label_map[record['is_grounded']]
            dataset.append({
                'claim': claim,
                'evidence': evidence,
                'label': label,
            })

        return dataset

class UnifiedDataset(Dataset):
    def __init__(self, claims, evidences, labels, dataset_name: str, tokenizer, max_length: int = 512):
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.label_map = self.config['label_map']

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        if CLASSIFIER_MODEL_NAME == 'qwen':
            text = f"<|im_start|>claim: {self.claims[idx]} [SEP] evidence: {self.evidences[idx]}<|im_end|>"
        else:
            text = f"claim: {self.claims[idx]} [SEP] evidence: {self.evidences[idx]}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
