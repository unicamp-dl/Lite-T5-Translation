from typing import List
from typing import Tuple
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import gzip


EXTRA_TOKENS = ['à', 'è', 'ì', 'ò', 'ù', 'Á', 'É', 'Í', 'Ó', 'Ú', 'á', 'é', 'í', 'ó', 'ú', 'Â', 'Ê',
                'â', 'ê', 'ô', 'Ã', 'Õ', 'ã', 'õ', 'ü']

WORDS = ['não']#, 'Não', 'nós', 'só', 'Há']


class T5Dataset(Dataset):
    def __init__(self, text_pairs: List[Tuple[str]], tokenizer,
                 source_max_length: int = 32, target_max_length: int = 32):
        self.tokenizer = tokenizer
        self.text_pairs = text_pairs
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        source, target = self.text_pairs[idx]
        source_modified = 'translate English to Portuguese: '+ source + self.tokenizer.eos_token
        target_modified = target + self.tokenizer.eos_token

        source_tok = self.tokenizer.batch_encode_plus([source_modified], add_special_tokens=True,
                                                      max_length=self.source_max_length, pad_to_max_length=True,
                                                      return_tensors='pt')
        target_tok = self.tokenizer.batch_encode_plus([target_modified], add_special_tokens=True,
                                                      max_length=self.target_max_length, pad_to_max_length=True,
                                                      return_tensors='pt')

        return (source_tok['input_ids'][0], source_tok['attention_mask'][0], target_tok['input_ids'][0],
                target_tok['attention_mask'][0], source, target)


def create_adapted_tokenizer(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    for word in WORDS:
        tokenizer.add_tokens(word)

    added_tokens = []
    for tok in EXTRA_TOKENS:
        enc = tokenizer.encode(tok)
        if 2 in enc:
            added_tokens.append(tok)
            tokenizer.add_tokens(tok)

    return tokenizer, added_tokens


def create_ptt5_tokenizer():
    model_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    return tokenizer