import torch
from torchtext import data
import random
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from sklearn.model_selection import StratifiedKFold, train_test_split


def batch_input_make(batch, device, input_dim):
    # 10 tags -> 9 tags + 1 for padding
    batch_onehot = torch.zeros((batch.size()[1], batch.size()[0], input_dim))
    for batch_index, seq_data in enumerate(batch.permute(1, 0)):
        batch_onehot[batch_index, torch.arange(seq_data.size()[0]), seq_data] = 1

    return batch_onehot.permute(1, 0, 2).to(device)


def batch_mask_make(batch, device):
    """Create an attention mask
    Args:
        batch: (seq_len, batch_size)
        device: GPU or CPU device tensor
    
    Returns:
        attn_mask: (batch_size, seq_len)
    """
    attn_mask = torch.where(batch.permute(1, 0) != 0,
                            torch.ones((batch.size()[1], batch.size()[0]), device=device),
                            torch.zeros((batch.size()[1], batch.size()[0]), device=device))
    return attn_mask.to(device)


def batch_replace_token(batch, token_index, unk_index, device):
    """Replace an token index with an unk index
    Args:
        batch: (seq_len, batch_size)
        token_index (int): token index that will be replaced
        unk_index (int): unk token index
        device: GPU or CPU device tensor
    
    Returns:
        replaced_batch: ()
    """ 
    replaced_batch = torch.where(batch == token_index,
                                 torch.tensor([unk_index], device=device),
                                 batch)
    return replaced_batch


def convert_to_tag_element(data_arr):
    """Convert string to tag elements"""
    ret = []
    for str_feature in data_arr:
        str_feature = str_feature.replace('[', '')
        str_feature = str_feature.replace(']', '')
        str_feature = str_feature.replace(' ', '')
        ret.append(np.array([element.replace("'", '') for element in str_feature.split(',')]))

    return np.array(ret, dtype=object)


def combine_examples(fields, data_arr, meta_arr):
    """Combine arrays into one array"""
    # Ref: https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb
    ret = []
    for main_data, meta_text in zip(data_arr, meta_arr):
        ret.append(data.Example.fromlist([main_data.text, meta_text.tolist(), main_data.flag], fields))
    return np.array(ret, dtype=object)


class DataProcessor(object):
    def __init__(self, path: str, SEED=1234):
        # init
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        self.SEED = SEED

        # load data
        TEXT = data.Field(sequential=True)
        LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.long)
        fields = [('text', TEXT), ('meta_text', LABEL), ('flag', LABEL)]
        self._data = data.TabularDataset(
                        path=path,
                        format='csv',
                        skip_header=True,
                        fields=fields
                    )


    def get_fold_data(self, num_folds: int):
        # settings
        TEXT = data.Field(sequential=True)
        TEXT2 = data.Field(unk_token=None)
        LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        temp_fields = [('text', TEXT), ('meta_text', LABEL), ('flag', LABEL)]
        self.fields = [('text', TEXT), ('meta_text', TEXT2), ('flag', LABEL)]

        # load data arrays
        _data_arr = data.Dataset(self._data.examples, fields=temp_fields).examples
        self._data_arr_label = [int(example.flag) for example in _data_arr]
        _meta_arr = [example.meta_text for example in _data_arr]
        
        # decode meta texts
        _meta_arr = convert_to_tag_element(_meta_arr)

        # combine data arrays
        self._data_arr = combine_examples(self.fields, _data_arr, _meta_arr)

        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self.SEED)
        for train_index, test_index in kf.split(self._data_arr, self._data_arr_label):
            yield(
                TEXT,
                TEXT2,
                self.fields,
                train_index,
                data.Dataset(self._data_arr[test_index], fields=self.fields),
                data.Dataset(self._data_arr, fields=self.fields)
            )

    
    def get_train_val_data(self, train_index):
        """Returns train and validation data"""
        data_arr = self._data_arr[train_index]
        label_arr = np.array(self._data_arr_label)[train_index]
        X_train, X_val, _, _ = train_test_split(data_arr, label_arr, test_size=0.2, 
                                                random_state=self.SEED, stratify=label_arr, shuffle=True)
        return data.Dataset(X_train, fields=self.fields), data.Dataset(X_val, fields=self.fields)


class DataHolder(object):
    def __init__(self, X_train, X_val, text_field, meta_field):
        self.X_train = X_train
        self.X_val = X_val
        self.text_field = text_field
        self.meta_field = meta_field
    
    def get_train_data(self):
        return self.X_train

    def get_val_data(self):
        return self.X_val
    
    def get_fields(self):
        return self.text_field, self.meta_field