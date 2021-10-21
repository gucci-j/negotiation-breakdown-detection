from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

import optuna

from sklearn.dummy import DummyClassifier

from core import Model
from core import DataProcessor, DataHolder, EarlyStopping
from core import ConfusionMatrix, ROC, PrecRecall, ResultLogger
from core import compute_f1_prec_rec, compute_accuracy, batch_input_make, batch_mask_make

#
# Loading Arguments
#
if len(sys.argv) <= 1:
    raise Exception("""
    =======
    JSON config helper
    =======
    `mode` (str): Choose `train`, `train_with_tuning` or `random`.
    `base_path` (str): Specify a base path for saving log data.
    `data_path` (str): Specify a dataset (csv) path.
    `save_path` (str): [Train & Tuning] Where to save checkpoint weights? 
    `cache_path` (str): Specify a GloVe cache path.
    `seed` (int): Specify a random seed.
    `num_folds` (int): How many folds do you want to test?
    `batch_size` (int): Specify the size of a batch.
    `epochs` (int): Maximum number of epochs to train.
    `patience` (int): Number of epochs to terminate the training process
                      if validation loss does not improve.
    `gpu` (bool): Whether to use a GPU?
    `gpu_number` (int): When gpu is `true`, which gpu do you want to use?
    `rnn_type` (str): Choose `LSTM` or `GRU`.
    `attention` (bool): Whether to employ a self-attention mechanism.
    `lr` (float): Learning rate.
    `input_dim` (int): How many nego tags do you consider?
    `hidden_dim` (int): Specify the dimension of each LSTM/GRU hidden state.
    `num_layers` (int): How many layers for LSTM/GRU?
    `bidirectional` (bool): Use bidirectional LSTM or GRUs?
    `recurrent_dropout` (float): Specify a dropout rate for the recurrent component.
    `dense_dropout` (float): Specify a dropout rate for the classification layer component.
    `strategy` (str): [Random] Choose a DummyClassifier's strategy from
                      `prior`, `most_frequent` or `stratified`.""")
args_p = Path(sys.argv[1])
if args_p.exists() is False:
    raise Exception('Path not found. Please check an argument again!')

with args_p.open(mode='r') as f:
    true = True
    false = False
    null = None
    args = json.load(f)

#
# Logger
#
import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

import logging
logfile = str('{}/log/log-{}.txt'.format(args['base_path'], run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

#
# Settings
#

# Make a directory for saving cm & roc-auc figures & csv log files
fig_path = Path(f"{args['base_path']}/fig/{run_start_time}")
csv_path = Path(f"{args['base_path']}/log/csv/{run_start_time}")
if fig_path.exists() is True or csv_path.exists() is True:
    raise Exception('Your saving path already exists. Please give another.')
else:
    fig_path.mkdir()
    csv_path.mkdir()

# Seed Settings
torch.manual_seed(args['seed'])
if args["gpu"] is True:
    torch.cuda.manual_seed_all(args['seed'])


def train():
    """
    Train a model with k-fold cross validation
    """
    logger.info("***** Setup *****")
    logger.info(f"{run_start_time} | Configs: {args}")

    # make a save directory
    save_path = Path(f'{args["save_path"]}/{run_start_time}')
    if save_path.exists() is True:
        raise Exception('Your saving path already exists. Please give another.')
    save_path.mkdir()

    # settings
    data_processor = DataProcessor(args["data_path"], SEED=args["seed"])
    cm_processor = ConfusionMatrix()
    roc_processor = ROC()
    pr_processor = PrecRecall()
    _history = []
    fold_index = 0

    for TEXT, META_TEXT, fields, train_index, test_data, whole_data in data_processor.get_fold_data(num_folds=args['num_folds']):
        logger.info("***** Training *****")
        logger.info(f"Now fold: {fold_index + 1} / {args['num_folds']}")

        # 1. build a model
        model = Model(args['rnn_type'], args['input_dim'], args['hidden_dim'], args['num_layers'], 
                      args['recurrent_dropout'], args['dense_dropout'], args["bidirectional"],
                      is_attention=args["attention"])
        
        # 2. define an optimizer & criterion
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])

        # 3. additional settings (e.g., early stopping & tensorboard)
        early_stopping = EarlyStopping(logger, patience=args['patience'], verbose=True, metric_type="f1")
        save_path = Path(f'{args["save_path"]}/{run_start_time}/{fold_index}.pth')

        # 4. selecting an environment
        if args['gpu'] is True and args['gpu_number'] is not None:
            torch.cuda.set_device(args['gpu_number'])
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = model.to(device)
        
        # 5. make iterators
        TEXT.build_vocab(whole_data, vectors="glove.6B.300d", min_freq=2, 
                         vectors_cache=str(args['base_path'] + '/vector_cache'))
        META_TEXT.build_vocab(whole_data)
        logger.info(f'Embedding size: {TEXT.vocab.vectors.size()}.')
        logger.info(f'Meta info: {META_TEXT.vocab.itos}.')
        train_data, val_data = data_processor.get_train_val_data(train_index)
        train_iterator = data.Iterator(train_data, batch_size=args['batch_size'], 
                                       sort_key=lambda x: len(x.text), device=device)
        val_iterator = data.Iterator(val_data, batch_size=args['batch_size'], 
                                     sort_key=lambda x: len(x.text), device=device)
        test_iterator = data.Iterator(test_data, batch_size=args['batch_size'], 
                                      sort_key=lambda x: len(x.text), device=device)

        # training
        for epoch in range(args['epochs']):
            train_loss, train_acc, train_fbeta, train_f1, train_precision, train_recall = train_run(model, train_iterator, optimizer, criterion, device)
            logger.info(f'\t[Epoch: {epoch+1:02}; Train] loss: {train_loss:.3f} '
                        f'| acc: {train_acc:.3f} | fbeta: {train_fbeta:.3f} | f1: {train_f1:.3f} '
                        f'| prec: {train_precision:.3f} | recall: {train_recall:.3f}')
            val_loss, val_cls_ret = eval_run(model, val_iterator, criterion, device)
            logger.info(f'\t[Validation] loss: {val_loss:.3f} | acc: {val_cls_ret["acc"]:.3f} '
                        f'| fbeta: {val_cls_ret["fbeta"]:.3f} | f1: {val_cls_ret["f1"]:.3f} '
                        f'| prec: {val_cls_ret["prec"]:.3f} | rec: {val_cls_ret["rec"]:.3f}')

            early_stopping(val_cls_ret["f1"], model, save_path)
            if early_stopping.early_stop:
                logger.info(f'\tEarly stopping at {epoch+1:02}')
                break

        # logging using the test fold
        pretrained_dict = torch.load(save_path)
        model.load_state_dict(pretrained_dict)
        test_loss, test_cls_ret = eval_run(model, test_iterator, criterion, device)
        logger.info(f'\t[Test] loss: {test_loss:.3f} | acc: {test_cls_ret["acc"]:.3f} '
                    f'| fbeta: {test_cls_ret["fbeta"]:.3f} | f1: {test_cls_ret["f1"]:.3f} '
                    f'| prec: {test_cls_ret["prec"]:.3f} | rec: {test_cls_ret["rec"]:.3f}')
        _history.append([test_cls_ret["acc"], test_cls_ret["fbeta"], test_cls_ret["f1"], 
                         test_cls_ret["prec"], test_cls_ret["rec"]])
        test_run(model, test_iterator, criterion, device, TEXT, META_TEXT,
                 cm_processor, roc_processor, pr_processor, fold_index)
        fold_index += 1
        torch.cuda.empty_cache()
    
    # summarise results
    logger.info('***** Cross Validation Test Result *****')
    _history = np.asarray(_history)
    roc_auc_history = np.array(roc_processor.get_auc_list()).reshape(-1, 1)
    pr_auc_history = np.array(pr_processor.get_auc_list()).reshape(-1, 1)
    _history = np.concatenate((_history, roc_auc_history, pr_auc_history), axis=1)
    df = pd.DataFrame(_history, columns=['Accuracy', 'Fbeta', 'F1', 'Precision', 'Recall',
                                         'ROC-AUC', 'PR-AUC'])
    df.to_csv(f"{args['base_path']}/log/csv/{run_start_time}/result.csv", index=False)
    acc = np.mean(_history[:, 0])
    fbeta = np.mean(_history[:, 1])
    f1 = np.mean(_history[:, 2])
    precision = np.mean(_history[:, 3])
    recall = np.mean(_history[:, 4])
    roc_auc = np.mean(_history[:, 5])
    pr_auc = np.mean(_history[:, 6])
    logger.info(f'Accuracy: {acc:.3f} ({np.std(_history[:, 0]):.3f}), '
                f'Fbeta: {fbeta:.3f} ({np.std(_history[:, 1]):.3f}), ' 
                f'F1: {f1:.3f} ({np.std(_history[:, 2]):.3f}), '
                f'Precision: {precision:.3f} ({np.std(_history[:, 3]):.3f}), '
                f'Recall: {recall:.3f} ({np.std(_history[:, 4]):.3f}), '
                f'ROC-AUC: {roc_auc:.3f} ({np.std(_history[:, 5]):.3f}), '
                f'PR-AUC: {pr_auc:.3f} ({np.std(_history[:, 6]):.3f})')


def train_with_tuning():
    logger.info("***** Setup *****")
    logger.info(f"{run_start_time} | Configs: {args}")

    # make a save directory
    save_path = Path(f'{args["save_path"]}/{run_start_time}')
    if save_path.exists() is True:
        raise Exception('Your saving path already exists. Please give another.')
    save_path.mkdir()
    best_save_path = Path(f'{args["save_path"]}/{run_start_time}/best')
    if best_save_path.exists() is True:
        raise Exception('Your saving path already exists. Please give another.')
    best_save_path.mkdir()

    # data settings
    data_processor = DataProcessor(args["data_path"], SEED=args["seed"])

    # logger settings
    cm_processor = ConfusionMatrix()
    roc_processor = ROC()
    pr_processor = PrecRecall()
    _history = []

    # device settings
    if args['gpu'] is True:
        torch.cuda.set_device(args['gpu_number'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # set a criterion
    criterion = nn.BCEWithLogitsLoss()

    # tuning -> test over five folds
    fold_index = 0
    for TEXT, META_TEXT, fields, train_index, test_data, whole_data in data_processor.get_fold_data(num_folds=args['num_folds']):
        logger.info(f"Now fold: {fold_index + 1} / {args['num_folds']}")
        
        # make iterators
        TEXT.build_vocab(whole_data, vectors="glove.6B.300d", min_freq=2,
                         vectors_cache=str(args['base_path'] + '/vector_cache'))
        META_TEXT.build_vocab(whole_data)
        logger.info(f'Embedding size: {TEXT.vocab.vectors.size()}.')
        logger.info(f'Meta info: {META_TEXT.vocab.itos}.')
        train_data, val_data = data_processor.get_train_val_data(train_index)
        dataholder = DataHolder(train_data, val_data, TEXT, META_TEXT)

        # tuning
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, dataholder), n_trials=100)

        # tuning results
        trial = study.best_trial
        logger.info('Tuning results:')
        for key, value in trial.params.items():
            logger.info('\t{}: {}'.format(key, value))
        
        # logging using the test fold
        ## 1. build a model
        if trial.params["bidirectional"] == 0:
            bidirectional = False
        else:
            bidirectional = True
        model = Model(args['rnn_type'], args['input_dim'], trial.params['hidden_dim'], trial.params['num_layers'], 
                      trial.params['recurrent_dropout_rate'], trial.params['dense_dropout_rate'],
                      bidirectional, is_attention=args["attention"])
        fold_save_path = Path(f'{args["save_path"]}/{run_start_time}/{trial.number}.pth')
        pretrained_dict = torch.load(fold_save_path)
        model.load_state_dict(pretrained_dict)
        model = model.to(device)

        ## 2. save weights for future use
        best_model_path = Path(f'{args["save_path"]}/{run_start_time}/best/{fold_index}.pth')
        torch.save(model.state_dict(), best_model_path)
        
        ## 3. test
        test_iterator = data.Iterator(test_data, batch_size=args['batch_size'], 
                                      sort_key=lambda x: len(x.text), device=device)
        test_loss, test_cls_ret = eval_run(model, test_iterator, criterion, device)
        logger.info(f'\t[Test] loss: {test_loss:.3f} | acc: {test_cls_ret["acc"]:.3f} '
                    f'| fbeta: {test_cls_ret["fbeta"]:.3f} | f1: {test_cls_ret["f1"]:.3f} '
                    f'| prec: {test_cls_ret["prec"]:.3f} | rec: {test_cls_ret["rec"]:.3f}')
        _history.append([test_cls_ret["acc"], test_cls_ret["fbeta"], test_cls_ret["f1"], 
                         test_cls_ret["prec"], test_cls_ret["rec"]])
        test_run(model, test_iterator, criterion, device, TEXT, META_TEXT, 
                 cm_processor, roc_processor, pr_processor, fold_index)
        fold_index += 1
        torch.cuda.empty_cache()
    
    logger.info('***** Cross Validation Test Result *****')
    _history = np.asarray(_history)
    roc_auc_history = np.array(roc_processor.get_auc_list()).reshape(-1, 1)
    pr_auc_history = np.array(pr_processor.get_auc_list()).reshape(-1, 1)
    _history = np.concatenate((_history, roc_auc_history, pr_auc_history), axis=1)
    df = pd.DataFrame(_history, columns=['Accuracy', 'Fbeta', 'F1', 'Precision', 'Recall',
                                         'ROC-AUC', 'PR-AUC'])
    df.to_csv(f"{args['base_path']}/log/csv/{run_start_time}/result.csv", index=False)
    acc = np.mean(_history[:, 0])
    fbeta = np.mean(_history[:, 1])
    f1 = np.mean(_history[:, 2])
    precision = np.mean(_history[:, 3])
    recall = np.mean(_history[:, 4])
    roc_auc = np.mean(_history[:, 5])
    pr_auc = np.mean(_history[:, 6])
    logger.info(f'Accuracy: {acc:.3f} ({np.std(_history[:, 0]):.3f}), '
                f'Fbeta: {fbeta:.3f} ({np.std(_history[:, 1]):.3f}), ' 
                f'F1: {f1:.3f} ({np.std(_history[:, 2]):.3f}), '
                f'Precision: {precision:.3f} ({np.std(_history[:, 3]):.3f}), '
                f'Recall: {recall:.3f} ({np.std(_history[:, 4]):.3f}), '
                f'ROC-AUC: {roc_auc:.3f} ({np.std(_history[:, 5]):.3f}), '
                f'PR-AUC: {pr_auc:.3f} ({np.std(_history[:, 6]):.3f})')


def objective(trial, dataholder):
    logger.info(f"Now trial: {trial.number}")

    # ====
    # Generate hyperparam candidates
    # ====
    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    # Int parameter
    bidirectional = trial.suggest_int('bidirectional', 0, 1)
    if bidirectional == 0:
        bidirectional = False
    else:
        bidirectional = True
    # Int parameter
    num_layers = trial.suggest_int('num_layers', 1, 4)
    # Int parameter
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    # Uniform parameter
    recurrent_dropout_rate = trial.suggest_uniform('recurrent_dropout_rate', 0.0, 1.0)
    dense_dropout_rate = trial.suggest_uniform('dense_dropout_rate', 0.0, 1.0)
    # show selected params
    for key, value in trial.params.items():
        logger.info('\t{}: {}'.format(key, value))

    # ====
    # Data & tuning configs
    # ====
    ## run device configs
    if args['gpu'] is True:
        torch.cuda.set_device(args['gpu_number'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # data settings
    train_data = dataholder.get_train_data()
    val_data = dataholder.get_val_data()
    train_iterator = data.Iterator(train_data, batch_size=args['batch_size'], 
                                   sort_key=lambda x: len(x.text), device=device)
    val_iterator = data.Iterator(val_data, batch_size=args['batch_size'], 
                                 sort_key=lambda x: len(x.text), device=device)
    save_path = Path(f'{args["save_path"]}/{run_start_time}/{trial.number}.pth')

    ## set a criterion
    criterion = nn.BCEWithLogitsLoss()

    # ====
    # tuning
    # ====
    # 1. build a model
    model = Model(args['rnn_type'], args['input_dim'], hidden_dim, num_layers, 
                  recurrent_dropout_rate, dense_dropout_rate, bidirectional, 
                  is_attention=args["attention"])
    model = model.to(device)

    # 2. define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. additional settings (e.g., early stopping)
    early_stopping = EarlyStopping(logger, patience=args['patience'], verbose=False, metric_type="f1")

    # training
    for epoch in range(args['epochs']):
        _, _, _, _, _, _ = train_run(model, train_iterator, optimizer, 
                                     criterion, device, tuning_mode=True)
        val_loss, val_cls_ret = eval_run(model, val_iterator, criterion, device)
        
        # check convergence
        early_stopping(val_cls_ret["f1"], model, save_path)
        if early_stopping.early_stop:
            logger.info(f'\tEarly stopping at {epoch+1:02}')
            break
    
    # logging
    pretrained_dict = torch.load(save_path)
    model.load_state_dict(pretrained_dict)
    val_loss, val_cls_ret = eval_run(model, val_iterator, criterion, device)
    logger.info(f'\t[Trial] loss: {val_loss:.3f} | acc: {val_cls_ret["acc"]:.3f} '
                f'| fbeta: {val_cls_ret["fbeta"]:.3f} | f1: {val_cls_ret["f1"]:.3f} '
                f'| prec: {val_cls_ret["prec"]:.3f} | rec: {val_cls_ret["rec"]:.3f}')
    torch.cuda.empty_cache()

    return val_cls_ret["f1"]


def train_run(model, iterator, optimizer, criterion, device, tuning_mode=False):
    epoch_loss = 0
    epoch_fbeta = 0
    epoch_f1 = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_acc = 0

    y_pred_list = np.array([])
    y_true_list = np.array([])

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        # generate a one-hot target & attention mask
        batch_metatext = batch_input_make(batch.meta_text, device, args['input_dim'])
        if args["attention"]:
            batch_mask = batch_mask_make(batch.meta_text, device)
        else:
            batch_mask = None
        # pass to the model
        output = model(batch_metatext, batch_mask)
        # compute loss & update weights
        loss = criterion(output, batch.flag)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # save results
        if not tuning_mode:
            epoch_loss += loss.item()
            preds = output.view(-1).sigmoid()
            y_pred_list = np.append(y_pred_list, preds.detach().cpu().numpy())
            y_true_list = np.append(y_true_list, batch.flag.float().detach().cpu().numpy())
    
    # compute metrics
    if not tuning_mode:
        epoch_acc = compute_accuracy(y_pred_list, y_true_list, sigmoid=True)
        epoch_f1, epoch_fbeta, epoch_prec, epoch_rec = compute_f1_prec_rec(y_pred_list, y_true_list, sigmoid=True)
    
    return epoch_loss / len(iterator), epoch_acc, epoch_fbeta, epoch_f1, epoch_prec, epoch_rec


def eval_run(model, iterator, criterion, device, tuning_mode=False):
    epoch_loss = 0
    cls_results = {"acc": 0.0, "fbeta": 0.0, "f1": 0.0, "prec": 0.0, "rec": 0.0}
    y_pred_list = np.array([])
    y_true_list = np.array([])

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_metatext = batch_input_make(batch.meta_text, device, args['input_dim'])
            if args["attention"]:
                batch_mask = batch_mask_make(batch.meta_text, device)
            else:
                batch_mask = None
            output = model(batch_metatext, batch_mask)
            loss = criterion(output, batch.flag)
            epoch_loss += loss.item()
            
            # save results
            if not tuning_mode:
                preds = output.view(-1).sigmoid()
                y_pred_list = np.append(y_pred_list, preds.cpu().numpy())
                y_true_list = np.append(y_true_list, batch.flag.float().cpu().numpy())

    # compute metrics
    if not tuning_mode:
        cls_results["acc"] = compute_accuracy(y_pred_list, y_true_list, sigmoid=True)
        cls_results["f1"], cls_results["fbeta"], cls_results["prec"], cls_results["rec"] = compute_f1_prec_rec(y_pred_list, y_true_list, sigmoid=True)

    return epoch_loss / len(iterator), cls_results


def test_run(model, iterator, criterion, device, TEXT, META_TEXT,
             cm_processor, roc_processor, pr_processor, fold_index):
    y_pred_list = np.array([])
    y_pred_binary_list = np.array([])
    y_true_list = np.array([])

    result_logger = ResultLogger()

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_metatext = batch_input_make(batch.meta_text, device, args['input_dim'])
            if args["attention"]:
                batch_mask = batch_mask_make(batch.meta_text, device)
            else:
                batch_mask = None
            preds = model(batch_metatext, batch_mask)
            preds = preds.sigmoid()
            preds = preds.cpu().numpy()
            rounded_preds = (preds >= 0.5).astype(int)
            y_pred_list = np.append(y_pred_list, preds)
            y_pred_binary_list = np.append(y_pred_binary_list, rounded_preds)
            y_true_list = np.append(y_true_list, batch.flag.float().cpu().numpy())
            result_logger.decode_text(batch.text, batch.meta_text, TEXT, META_TEXT)
    
    cm_processor.draw(y_pred_list, y_true_list, fold_index, run_start_time, args['base_path'])
    pr_processor.draw(y_pred_list, y_true_list, fold_index)
    roc_processor.draw(y_pred_list, y_true_list, fold_index)
    result_logger.set_group(y_true_list, y_pred_binary_list, y_pred_list)
    result_logger.export(run_start_time, args['base_path'], fold_index)

    if fold_index == (args['num_folds'] - 1):
        cm_processor.finishing(logger)
        roc_processor.finishing(run_start_time, args['base_path'], logger)
        pr_processor.finishing(run_start_time, args['base_path'])


def random():
    """
    Random with k-fold cross validation
    """
    logger.info("***** Setup *****")
    logger.info(f"{run_start_time} | Configs: {args}")

    # settings
    data_processor = DataProcessor(args["data_path"], SEED=args["seed"])
    cm_processor = ConfusionMatrix()
    roc_processor = ROC()
    pr_processor = PrecRecall()
    _history = []
    device = torch.device('cpu')
    fold_index = 0

    for TEXT, META_TEXT, fields, train_index, test_data, whole_data in data_processor.get_fold_data(num_folds=args['num_folds']):
        logger.info("***** Training *****")
        logger.info(f"Now fold: {fold_index + 1} / {args['num_folds']}")

        # 1. build a random model
        model = DummyClassifier(strategy=args["strategy"])

        # 2. create iterators
        train_data, val_data = data_processor.get_train_val_data(train_index)
        train_iterator = data.Iterator(train_data, batch_size=args['batch_size'], 
                                       sort_key=lambda x: len(x.text), device=device)
        test_iterator = data.Iterator(test_data, batch_size=args['batch_size'], 
                                      sort_key=lambda x: len(x.text), device=device)
        TEXT.build_vocab(train_data, vectors="glove.6B.300d", min_freq=2,
                         vectors_cache=str(args['base_path'] + '/vector_cache'))
        META_TEXT.build_vocab(train_data)
        logger.info(f'Embedding size: {TEXT.vocab.vectors.size()}.')
        logger.info(f'Meta info: {META_TEXT.vocab.itos}.')

        # 3. run a random model
        test_cls_ret =  random_run(model, train_iterator, test_iterator, 
                                   cm_processor, roc_processor, pr_processor, fold_index)
        logger.info(f'\t[Test] acc: {test_cls_ret["acc"]:.3f} '
                    f'| fbeta: {test_cls_ret["fbeta"]:.3f} | f1: {test_cls_ret["f1"]:.3f} '
                    f'| prec: {test_cls_ret["prec"]:.3f} | rec: {test_cls_ret["rec"]:.3f}')
        _history.append([test_cls_ret["acc"], test_cls_ret["fbeta"], test_cls_ret["f1"], 
                         test_cls_ret["prec"], test_cls_ret["rec"]])
        fold_index += 1
        torch.cuda.empty_cache()
    
    # summarise results
    logger.info('***** Cross Validation Test Result *****')
    _history = np.asarray(_history)
    roc_auc_history = np.array(roc_processor.get_auc_list()).reshape(-1, 1)
    pr_auc_history = np.array(pr_processor.get_auc_list()).reshape(-1, 1)
    _history = np.concatenate((_history, roc_auc_history, pr_auc_history), axis=1)
    df = pd.DataFrame(_history, columns=['Accuracy', 'Fbeta', 'F1', 'Precision', 'Recall',
                                         'ROC-AUC', 'PR-AUC'])
    df.to_csv(f"{args['base_path']}/log/csv/{run_start_time}/result.csv", index=False)
    acc = np.mean(_history[:, 0])
    fbeta = np.mean(_history[:, 1])
    f1 = np.mean(_history[:, 2])
    precision = np.mean(_history[:, 3])
    recall = np.mean(_history[:, 4])
    roc_auc = np.mean(_history[:, 5])
    pr_auc = np.mean(_history[:, 6])
    logger.info(f'Accuracy: {acc:.3f} ({np.std(_history[:, 0]):.3f}), '
                f'Fbeta: {fbeta:.3f} ({np.std(_history[:, 1]):.3f}), ' 
                f'F1: {f1:.3f} ({np.std(_history[:, 2]):.3f}), '
                f'Precision: {precision:.3f} ({np.std(_history[:, 3]):.3f}), '
                f'Recall: {recall:.3f} ({np.std(_history[:, 4]):.3f}), '
                f'ROC-AUC: {roc_auc:.3f} ({np.std(_history[:, 5]):.3f}), '
                f'PR-AUC: {pr_auc:.3f} ({np.std(_history[:, 6]):.3f})')


def random_run(model, train_iterator, test_iterator,
               cm_processor, roc_processor, pr_processor, fold_index):
    
    cls_results = {"acc": 0.0, "fbeta": 0.0, "f1": 0.0, "prec": 0.0, "rec": 0.0}
    
    y_train_true_list = np.array([])
    y_test_true_list = np.array([])

    # prepare dummy training data
    with torch.no_grad():
        for batch in train_iterator:
            y_train_true_list = np.append(y_train_true_list, batch.flag.float().numpy())
    X = np.zeros_like(y_train_true_list).reshape(-1, 1)

    # train a dummy model
    model.fit(X, y_train_true_list)

    # prepare test data
    with torch.no_grad():
        for batch in test_iterator:
            y_test_true_list = np.append(y_test_true_list, batch.flag.float().numpy())
    X_test = np.zeros_like(y_test_true_list).reshape(-1, 1)

    # predict
    preds_class = model.predict(X_test) # -> binary label
    preds_proba = model.predict_proba(X_test)[:, 1]

    # compute metrics
    cls_results["acc"] = compute_accuracy(preds_class, y_test_true_list, sigmoid=False)
    cls_results["f1"], cls_results["fbeta"], cls_results["prec"], cls_results["rec"] = compute_f1_prec_rec(preds_class, y_test_true_list, sigmoid=False)
    cm_processor.draw(preds_proba, y_test_true_list, fold_index, run_start_time, args['base_path'])
    pr_processor.draw(preds_proba, y_test_true_list, fold_index)
    roc_processor.draw(preds_proba, y_test_true_list, fold_index)

    if fold_index == (args['num_folds'] - 1):
        cm_processor.finishing(logger)
        roc_processor.finishing(run_start_time, args['base_path'], logger)
        pr_processor.finishing(run_start_time, args['base_path'])
    
    return cls_results


def main():
    if args["mode"] == "train":
        train()
    elif args["mode"] == "train_with_tuning":
        train_with_tuning()
    elif args["mode"] == "random":
        random()
    else:
        raise NotImplementedError("Check your mode argument!")


if __name__ == '__main__':
    main()