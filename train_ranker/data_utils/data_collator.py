import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

@dataclass
class DataCollatorForReranking_train:

    tokenizer: PreTrainedTokenizerBase
    model_type: str = "roberta"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        '''
        feature list of {
            "input_ids": [C, L]
        }
        '''
        max_step1_len = max([max([len(c) for c in x['step1_data']]) for x in features])
        max_step2_len = max([max([len(c) for c in x['step2_data']]) for x in features])

        def bert_pad(X, max_len=-1):
            if max_len < 0:
                max_len = max(len(x) for x in X)
            result = []
            for x in X:
                if len(x) < max_len:
                    x.extend([self.tokenizer.pad_token_id] * (max_len - len(x)))
                result.append(x)
            return torch.LongTensor(result)

        step1_ids = [bert_pad(x['step1_data'], max_step1_len) for x in features]
        step2_ids = [bert_pad(x['step2_data'], max_step2_len) for x in features]
        
        step1_ids = torch.stack(step1_ids) # (B, C, L)
        step2_ids = torch.stack(step2_ids)

        step1_attention_mask = step1_ids != self.tokenizer.pad_token_id
        step2_attention_mask = step2_ids != self.tokenizer.pad_token_id
        
        
        batch = [{'input_ids': step1_ids,'attention_mask': step1_attention_mask,'step_type':'step1'},
                 {'input_ids': step2_ids,'attention_mask': step2_attention_mask,'step_type':'step2'}]

        if "results" in features[0].keys():
            batch['results'] = [x['results'] for x in features]  # {'source': untokenized sentence, "target": untokenized sentence, "candidates": list of untokenized sentence}

        return batch
    

@dataclass
class DataCollatorForReranking_eval:

    tokenizer: PreTrainedTokenizerBase
    model_type: str = "roberta"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        '''
        feature list of {
            "input_ids": [C, L],"ids":str,"result":bool
        }
        '''
        max_len = max([max([len(c) for c in x['input_ids']]) for x in features])

        def bert_pad(X, max_len=-1):
            if max_len < 0:
                max_len = max(len(x) for x in X)
            result = []
            for x in X:
                if len(x) < max_len:
                    x.extend([self.tokenizer.pad_token_id] * (max_len - len(x)))
                result.append(x)
            return torch.LongTensor(result)

        ids = [bert_pad(x['input_ids'], max_len) for x in features]
        
        ids = torch.stack(ids) # (B, C, L)

        attention_mask = ids != self.tokenizer.pad_token_id
        
        batch = {'input_ids': ids,'attention_mask': attention_mask}

        if "result" in features[0].keys():
            batch['results'] = [x['result'] for x in features]  # {'source': untokenized sentence, "target": untokenized sentence, "candidates": list of untokenized sentence}
            batch['ids'] = [x['ids'] for x in features]
        return batch