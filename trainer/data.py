import torch
import transformers
import json

from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm



@dataclass
class DataCollatorForMambaDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        labels=batch["input_ids"].clone()
        labels_new = torch.where(labels == self.tokenizer.pad_token_id, torch.tensor(-100), labels).contiguous()

        return dict(
            input_ids=batch["input_ids"],
            labels=labels_new,
            attention_mask=batch["input_ids"].ne(self.tokenizer.pad_token_id),
        )

