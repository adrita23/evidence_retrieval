# Evidence Retrieval in Clinical Trials
This script allows train a model for evidence retrieval in clinical trials using HuggingFace transformers for the paper titled 'Investigation of the Role of Context in Evidence Retrieval for Natural
Language Inference from Clinical Trials'.

## Requirements
- Python 3.7+
- PyTorch
- HuggingFace Transformers

Install these using `pip install -r requirements.txt`

## details on creating the dataset
check data/readme.md

## Usage
Run the script using:

1. For Baseline:
`python baseline.py --epochs <number_of_epochs> --batch_size <batch_size> --model_name <model_name> --hidden_size <hidden_size> --seed <seed_value> --lr <learning_rate> --best_model_path <path_to_save_model>`

2. For BOW:
`python BOW.py --epoch <number_of_epochs> --batch_size <batch_size> --model_name <model_name>  --hidden_size <hidden_size> --seed <seed_value>  --lr <learning_rate> --path <path_to_save_model>`

3. For BM25
`python BM25.py --epoch <number_of_epochs> --batch_size <batch_size> --model_name <model_name> --hidden_size <hidden_size> --seed <seed_value> --lr <learning_rate> --path <path_to_save_model>`

4. For summary
    -> create summary with `python create_summary.py`
    -> concatanate data with summary with `python concatanate_with_summary.py`
    -> train and evaluate with `python with_summary.py --epoch <number_of_epochs> --batch_size <batch_size>  --model_name <model_name>  --hidden_size <hidden_size> --seed <seed_value> --lr <learning_rate>--path <path_to_save_model>`

5. For BiLSTM
`python BiLSTM.py --epoch <number_of_epochs> --batch_size <batch_size>  --model_name <model_name>  --hidden_size <hidden_size> --seed <seed_value> --lr <learning_rate>--path <path_to_save_model>`

### Arguments

- `--epochs`: Number of training epochs. Default is 8.
- `--batch_size`: Batch size for training and evaluation. Default is 16.
- `--model_name`: Name of the model to be used. Default is "michiyasunaga/BioLinkBERT-base".
- `--hidden_size`: Hidden size for the model. Default is 768.
- `--seed`: Random seed for reproducibility. Default is 1.
- `--lr`: Learning rate for optimizer. Default is 5e-6.
- `--train_path`: Path to the training data. Default is "./data/train.json".
- `--dev_path`: Path to the development/validation data. Default is "./data/dev.json".
- `--best_model_path`: Path to save the best model. 

## Output
The script will train the model and save the best model based on F1-score to the provided `best_model_path`.


