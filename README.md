# PermeaCyclix: Predicting Macrocycle Permeability

## Description
This model is designed for predicting the cellular permeability of macrocyclic molecules of interest. It leverages a machine learning approach trained on 2D molecular descriptors selected using the Boruta algorithm. The model assigns target labels based on apparent permeability measurements obtained from an in vitro PAMPA assay. By utilizing this model, users can gain insights into the potential permeability properties of various molecules, aiding in drug discovery and development processes of macrocycles.

## Content
- `main.py` The main script to execute the permeability predictions
- `config.py` Centralized location for key configurations (paths, constants, feature definitions)
- `xgb_model.pkl` Serialized XGB model object employed to make predictions
- `requirements.txt` List of packages with their specified versions required for running the main script

## Requirements
Install all the packages of the `requirements.txt` into your environment by running the following command:
```
pip install -r requirements.txt
```

Ensure that Python 3 is installed on your system and that all required dependencies are satisfied before running the script.

## User guide

### 1. Prepare the data
- Store the SMILES notations of your molecules in a file without a header. Formats such as `smiles.smi` or `smiles.csv` are suitable. 

### 2. Download repository
- Clone or download this repository to your local machine. Ensure that all necessary files are within the same directory.

### 3. Run the script
- Open a terminal or command prompt.
- Navigate to the directory containing the repository files.
- Execute the following command:
```
python3 main.py 'smiles.smi'
```
(Replace 'smiles.smi' with the name of your SMILES file.)

### 4. Interpreting Results
- Upon completion, the script will generate two files:
1) `descriptors.csv` contains molecular descriptor values calculated by Mordred software and 
2) `predicted_labels.csv` contains the results of premeability predictions.

- Each SMILES notation will be assigned a predicted permeability label.
    - Label 1 indicates permeable.
    - Label 0 indicates impermeable.

- The output file will be located in the same directory where the script is executed.
