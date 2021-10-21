"""
MIT License

Copyright (c) 2018 Bjarte Mehus Sunde & 2019-2020 Atsuki Yamaguchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, patience=7, verbose=False, metric_type='loss'):
        """
        Args:
            `patience` (int): How long to wait after last time validation loss improved.
                            Default: 7
            `verbose` (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logger

        if metric_type == 'loss':
            self.val_pred_min = np.Inf
        else:
            self.val_pred_max = -np.Inf
        self.metric_type = metric_type


    def __call__(self, val_value, model, save_path):
        if self.metric_type == 'loss':
            score = -val_value
        else:
            score = val_value
        
        if self.best_score is None: # init
            self.best_score = score
            self.show_checkpoint(val_value, model, save_path)
        elif score < self.best_score: # not improved
            self.counter += 1
            if self.verbose:
                self.logger.info(f'\tEarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # improved
            self.best_score = score
            self.show_checkpoint(val_value, model, save_path)
            self.counter = 0


    def show_checkpoint(self, val_value, model, save_path):
        if self.metric_type == 'loss':
            if self.verbose:
                self.logger.info(f'\tValidation loss decreased: {self.val_pred_min:.6f} --> {val_value:.6f}')
            self.val_pred_min = val_value
        else:
            if self.verbose:
                self.logger.info(f'\tValidation {self.metric_type} improved: {self.val_pred_max:.6f} --> {val_value:.6f}')
            self.val_pred_max = val_value
        self.save_model(model, save_path)
    

    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)