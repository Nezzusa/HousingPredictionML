# House pricing prediction machine learning model

### An end-to-end supervised machine learning pipeline built with Python and scikit-learn

This project demonstrates a functional supervised regression model predicting median house values using the California Housing Prices dataset. 
It covers the full machine learning lifecycle: exploratory data analysis (EDA) and feature engineering in a Jupyter Notebook, modular preprocessing and data cleaning script, 
and a baseline Linear Regression model with evaluation metrics (MAE, RMSE, $R^{2}$).

## Project Structure

```text
HousingPredictionML/
├── data/
│   ├── raw/                  # original dataset
│   └── processed/            # initially cleaned data from cleaner.py
│   
├── src/
|   ├── data_Analysis.ipynb   # EDA, distributions, and correlation analysis
│   ├── cleaner.py            # Initial data processing script
│   └── train.py              # Model training and evaluation script
├── requirements.txt          # Project dependencies
└── README.md
```

# Installation Guide

### Clone the Repository
```
git clone [https://github.com/Nezzusa/HousingPredictionML.git](https://github.com/Nezzusa/HousingPredictionML.git)
cd HousingPredictionML
```
### Set up Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# Usage

To see data analysis and conclusions, run:
```bash 
jupyter lab src/data_analysis.ipynb
```

To clean the data, run:
```bash
jupyter lab src/cleaner.py
```

To train and evaluate the model, run:
```bash
python src/train.py
```

