## Usage


## Environment Configuration

```bash
conda create -n bioact python=3.9
conda activate bioact
pip install -r requirements.txt
```

## Model Training

### Model training consists of two parts: data preprocessing and model training.

### 1. Data Preprocessing
The ADR prediction model (HetSia-SafeNet) is trained using 9202 drugs with 18 ADR labels. The raw ADR dataset is stored in `./data_process/ADR_tree_label`. You can download the ADR_tree_label folder from [this link](https://drive.google.com/drive/folders/1BrL1Gw12G4eamExrf2lmc_fnAXidmjo_?usp=drive_link).
Run `./data_process/drug_ac50_cmax_get.ipynb` to combine off-target data predicted by PreMOTA and Cmax,free values predicted by the MotifAttnNet model. This will generate the required drug features to construct the training dataset.You can download ADR_multitask_dataset, ADR_multitask_dataset_random_split, and ADR_multitask_dataset_scaffold_split from [this link](https://drive.google.com/drive/folders/1jUWwrYmRuj47Ko7ldkTzn213Y5bgaB_Y?usp=drive_link) and place them in the `./Data` folder.

### 2. Data Splitting
Run `./Data/data_split.ipynb` to split the dataset either randomly or using a scaffold-based approach.

### 3. Model Training
Run `./train_adr_property_adremb.py` to train the model and save the trained model in the `./result_split` directory.

property_feature_adr_emb_random: Model trained with randomly split data.
property_feature_adr_emb_scaffold: Model trained with scaffold-based split data.


## Model Prediction and Visualization

### 1. Data Preparation for Prediction (Merging Off-Target and Cmax Data)
Run `./data_process/data_pro_dose_study_test_indrugs.ipynb` to combine off-target data from PreMOTA and Cmax/free values from MotifAttnNet, creating a format that is compatible with the model for prediction.

### 2. Model Prediction and Visualization
Run `./dose_study_drugs_visual.ipynb` to perform predictions and visualize the results.