import torch
import transformers
import json
import numpy as np

from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)

import re
def proc_text(text):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)  # remove any mimic special token like [**2120-2-28**] or [**Hospital1 3278**]
    # text = re.sub(r'*{2,}', '*', text) # Collapse multiple * into a single *
    text = re.sub('Dr\.','doctor',text)
    text = re.sub('dr\.','doctor',text)
    text = re.sub('M\.D\.','doctor',text)
    text = re.sub('--|__|==','',text)
    step1 = re.sub("\r", "\n", text)
    step2 = re.sub(r'(?<!\n)\n(?!\n)', ' ', step1) # Replace single \n with a space
    text = re.sub(r'\n{2,}', '\n', step2) # Collapse multiple \n into a single \n
    return re.sub(r'  +', ' ', text)

def proc_note(notes):
    results = []
    for docs in notes:
        docs_new = proc_text(docs)
        results.append(docs_new)
    return results

def proc_prompt(index2prompt, return_mask=True):
    num_prompts = len(index2prompt)
    prompts = []
    for i in range(num_prompts):
        prompta = index2prompt[i]
        prompts.append(prompta)
    if return_mask:
        prompts = "<mask>\n".join(prompts)
        prompts += "<mask>"
    else:
        prompts = "\n".join(prompts)
        prompts += "\n"
    return prompts


def proc_labels(label_no, label_yes, labelsdict_list, label2index, return_raw=False):
    labelindexpair = sorted(label2index.items(), key=lambda p: p[1])
    labels = []
    for label,index in labelindexpair:
        labels.append(labelsdict_list[label])
    labels = np.array(labels).T
    if return_raw:
        return labels
    results = np.zeros_like(labels)
    results[labels==1] = label_yes
    results[labels==0] = label_no
    return results


@dataclass
class DataCollatorForMambaDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
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

    

        return batch

