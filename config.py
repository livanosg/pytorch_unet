from os.path import dirname, abspath

ROOT_DIR = dirname(abspath(__file__))
DATASET_DIR = ROOT_DIR + '/Datasets'
INIT_DATA_DIR = DATASET_DIR + '/Init'

paths = {'train': DATASET_DIR + '/Train_Sets',  # Training set
         'eval': DATASET_DIR + '/Eval_Sets',  # Evaluation set
         'infer': DATASET_DIR + '/Test_Sets',  # Test set
         'chaos': INIT_DATA_DIR + '/Chaos',  # Chaos dataset
         'base': INIT_DATA_DIR + '/Base',  # Folder to get Train and eval sets/
         'save_model': ROOT_DIR + '/saves',  # Save model
         'save_pred': ROOT_DIR + '/predictions'}  # Save predictions
