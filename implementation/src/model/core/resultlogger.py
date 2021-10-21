import pandas as pd
import numpy as np
from pathlib import Path

class ResultLogger(object):
    def __init__(self):
        self.dialogue_list = []
        self.meta_dialogue_list = []
        self.y_list = []
        self.preds_list = []
        self.preds_prob_list = []

    def decode_text(self, batch_text, batch_meta_text, TEXT, META_TEXT):
        texts = batch_text.transpose(0, 1)
        meta_texts = batch_meta_text.transpose(0, 1)

        for text in texts:
            dialogue = ""
            for index in text:
                token = TEXT.vocab.itos[index]
                if token == '<pad>':
                    continue
                dialogue = dialogue + ' ' + token
            self.dialogue_list.append(dialogue)
        
        for meta_text in meta_texts:
            dialogue = ""
            for index in meta_text:
                token = META_TEXT.vocab.itos[index]
                if token == '<pad>':
                    continue
                dialogue = dialogue + ' ' + token
            self.meta_dialogue_list.append(dialogue)
    
    def set_group(self, y, preds, probs):
        self.y_list = y
        self.preds_list = preds
        self.preds_prob_list = probs

    def export(self, run_start_time: str, base_path: str, fold_index: int):
        ret = []
        for y, pred, prob, text, meta_text in zip(self.y_list, self.preds_list,
                                                  self.preds_prob_list, 
                                                  self.dialogue_list, self.meta_dialogue_list):
            ret.append([y, pred, prob, text, meta_text])
            df = pd.DataFrame(ret, columns=['y', 'pred', 'prob', 'text', 'meta text'])
        df.to_csv(f'{base_path}/log/csv/{run_start_time}/{fold_index}.csv', index=False)