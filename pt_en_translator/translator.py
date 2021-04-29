import config
from models import T5FineTuner
import torch
from data import create_adapted_tokenizer


class PredictModel:
    def __init__(self, resume_from_checkpoint, max_length = 256, use_ptt5=True):
        model_name = config.get_model_name()
        self.tokenizer, self.added_tokens = create_adapted_tokenizer(model_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = T5FineTuner.load_from_checkpoint(resume_from_checkpoint, tokenizer=self.tokenizer,
                                                      train_dataloader=None, val_dataloader=None,
                                                      test_dataloader=None, learning_rate=1e-5,
                                                      added_tokens=self.added_tokens, target_max_length=max_length).to(self.device)

    def predict_pt_en(self, text):
        max_length = config.get_source_max_length()
        is_ptt5 = config.get_ptt5_checker()

        sent = "translate Portuguese to English: " + text + self.tokenizer.eos_token
        tok = self.tokenizer.encode_plus(sent, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=max_length, pad_to_max_length = True)
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))
        
        if is_ptt5:
            sys = [self.tokenizer.decode(tokens) for tokens in pred]
        else:
            sys = [fix_accent_breaks(self.tokenizer.decode(tokens), self.added_tokens) for tokens in pred]

        return sys


    def predict_batch_pt_en(self, text_list):
        max_length = config.get_source_max_length()
        is_ptt5 = config.get_ptt5_checker()
        sent_list=[]

        for text in text_list:
            sent_list.append("translate Portuguese to English: " + text + self.tokenizer.eos_token)

        tok = self.tokenizer.batch_encode_plus(sent_list, return_tensors='pt', truncation=True, add_special_tokens=True, max_length=max_length, pad_to_max_length = True)
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))
        
        if is_ptt5:
            sys = [self.tokenizer.decode(tokens) for tokens in pred]
        else:
            sys = [fix_accent_breaks(self.tokenizer.decode(tokens), self.added_tokens) for tokens in pred]

        return sys


    def predict_en_pt(self, text):
        max_length = config.get_source_max_length()
        is_ptt5 = config.get_ptt5_checker()

        sent = "translate English to Portuguese: " + text + self.tokenizer.eos_token
        tok = self.tokenizer.encode_plus(sent, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=max_length, pad_to_max_length = True)
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))

        if is_ptt5:
            sys = [self.tokenizer.decode(tokens) for tokens in pred]
        else:
            sys = [fix_accent_breaks(self.tokenizer.decode(tokens), self.added_tokens) for tokens in pred]

        return sys


    def predict_batch_en_pt(self, text_list):
        max_length = config.get_source_max_length()
        is_ptt5 = config.get_ptt5_checker()
        sent_list=[]

        for text in text_list:
            sent_list.append("translate English to Portuguese: " + text + self.tokenizer.eos_token)

        tok = self.tokenizer.batch_encode_plus(sent_list, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=max_length, pad_to_max_length = True)
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))

        if is_ptt5:
            sys = [self.tokenizer.decode(tokens) for tokens in pred]
        else:
            sys = [fix_accent_breaks(self.tokenizer.decode(tokens), self.added_tokens) for tokens in pred]

        return sys


def fix_accent_breaks(text, added_tokens):
    """
    A ideia é fazer a junção de letras com acento de volta para frases na validação.
    Isso serve para melhorar o BLEU.

    Args
        text: texto que terá acentuação corrigida

    Returns:
        Texto completo com acentuação corrigida

    """
    words = text.split(" ")
    out_words = []
    merge_pos = [idx for idx, dat in enumerate(words) if dat in added_tokens]
    for pos in sorted(merge_pos, reverse=True):
        if pos == 0:
            new_word = words[pos] + words[pos + 1]
            for i in range(2): words.pop
            words.pop(pos + 1)
            words.pop(pos)
            words.insert(pos, new_word)
        elif pos == len(words) - 1:
            new_word = words[pos - 1] + words[pos]
            words.pop(pos)
            words.pop(pos - 1)
            words.insert(pos - 1, new_word)
        else:
            new_word = words[pos - 1] + words[pos] + words[pos + 1]
            words.pop(pos + 1)
            words.pop(pos)
            words.pop(pos - 1)
            words.insert(pos - 1, new_word)

    return " ".join(words)
