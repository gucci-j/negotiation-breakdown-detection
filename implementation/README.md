This is the official implementation for Dialogue Act-based Breakdown Detection in Negotiation Dialogues (EACL 2021).

## Requirements
* PyTorch 
* torchtext
* numpy
* matplotlib
* scipy
* pandas
* seaborn
* scikit-learn
* tensorboard
* tensorflow
* Optuna


## Preprocess a dataset
### 1. Download existing datasets
For DealOrNoDeal, please refer to [https://www.aclweb.org/anthology/D17-1259/](https://www.aclweb.org/anthology/D17-1259/).  
For CraigslistBargain, please refer to [https://www.aclweb.org/anthology/D18-1256/](https://www.aclweb.org/anthology/D18-1256/).

Please place the datasets wherever you want to.

### 2. Preprocesing datasets
Change directory to `./src/preprocess`. Then, run a corresponding code depending on the type of datasets.

* **DealOrNoDeal**  
`python load_negotiation_dn.py [DN dataset path] [output file name]`

* **CraigslistBargain**  
`python load_negotiation_cb.py [CB dataset path] [output file name]`  
    > Note that you have to specify a folder path for the CB dataset as it has "train," "val," and "test" data.

* **JobInterview**  
`python load_negotiation_ji.py [JI dataset path] [output file name]`


### 3. Extract dialogue acts
* **DealOrNoDeal**  
`python create_meta_dn.py [DN preprocessed data path] [output file name]`

* **CraigslistBargain**  
`python create_meta_cb.py [CB preprocessed data path] [output file name]`

* **JobInterview** 
`python create_meta_ji.py [JI preprocessed data path] [output file name]`


## Training  
The recommended directory structure to save results is as follows:
```
    .
    ├── data                 # Tokenised CSV files
    │ 
    ├── log                  # Log files 
    │   └── csv              # -- CSV files
    │ 
    ├── fig                  # PNG files (confusion matrix & ROC/PR curves)
    │
    └── weights              # Checkpoint weight files            
```
> You can change the location of weight files as they are large by specifying it in input arguments.

### 1. Configure a model
You should create an original `.json` file. We include the existing ones for reference, which are stored under `params`.  

To check the roles of each argument, please run `python src/run.py` without giving any arguments. This will display a helper message.

### 2. Run
If you want to train/tune/test a model, please type as follows:
```
python src/run.py [path to a json file]
```
> E.g., `python src/run.py params/dn_tuning.json`
