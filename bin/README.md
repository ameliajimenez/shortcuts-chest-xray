# Training and testing models for chest X-ray
## Usage
### 1. Configure the training settings in the config file
Specify the path to the development subsets and the dataset (CheXpert-v1.0-small) in `config/config.json`.

### 2. Train a model with a development subset
```bash
$ train.py --device 0 --num_workers 2 --cfg_path ../config/config.json --save_path /logdirs/logdir-30k-2
```
In the HPC:
```bash
$ sbatch train.sh
```
Please note that to run the experiments in the HPC, you should first create the Anaconda virtual environment and modify the working directory in every shell script (.sh).

### 3. Test model: get predictions for the test set
```bash
$ test_model.py --device 0 --num_worker 2 --model_path ../logdirs/logdir-30k-2/ --in_csv_path ../config/my_test.csv
```
In the HPC:
```bash
$ sbatch test.sh
```

### 4. Compute metrics for the test set: accuracy, AUC, etc., and analysis of population subgroups. 
```bash
$ evaluate_dataframe.py
```

### 5. Retrieve embeddings from the model
Modify the forward function of `model/classifier` to return `(logits, logit_maps, feat_after_pool)`, where `feat_after_pool` corresponds to the output of the `global_pool` function. Use the file `get_embedding_model.py` to retrieve and save the embeddings for CheXpert and NIH-CXR14 datasets.
```bash
$ get_embedding_model.py --device 0 --num_worker 2 --model_path ../logdirs/logdir-30k-2/ --in_csv_path ../config/my_test.csv
```

### 6. Visualization of t-SNE projection
```bash
$ visualize_tsne_embedding.py
```