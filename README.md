# fpl-prediction-lstm
This project uses historical Fantasy Premier League (FPL) data to predict the fantasy points a player might score in the next gameweek. The system downloads data from a GitHub repository, processes and enriches it with additional features (like fixture difficulty), trains an LSTM (Long Short-Term Memory) model, and finally produces predictions and interactive plots. The data is scraped from https://github.com/vaastav/Fantasy-Premier-League/tree/master. 

**Overview**

The pipeline consists of four main stages:

1. Data Ingestion:
Downloads key CSV files (teams, fixtures, player lists, player gameweek data, etc.) from apublic GitHub repository and saves them locally.

2. Data Processing:
Loads the local data, computes additional features such as dynamic fixture difficulty (usingteam defensive strengths) and rolling averages (form). It also normalizes the features andconverts the data into time-series sequences for the LSTM model.

3. Model Training and Prediction:
Builds and trains an LSTM model using hyperparameter tuning. The model learns from thehistorical sequences and predicts fantasy points for the next gameweek.

4. Visualization and Analysis:
Produces interactive plots (using Plotly) showing the top 100 players by predicted fantasypoints versus their price, and also provides error analysis.

**Directory Structure**
```
.
├── data_ingestion.py     # Downloads data from GitHub and saves it locally.
├── data_processing.py    # Loads local data, computes new features, normalizes data, and creates LSTM input sequences.
├── model.py              # Defines, trains, and tunes the LSTM model; includes prediction functions.
├── main.py               # Runs the complete pipeline: ingestion, processing, model training, prediction, and plotting.
├── README.md             # This file.
└── data/                 # Directory where the ingested CSV files are saved.
```

**Requirements**
```
pip install pandas numpy requests scikit-learn tensorflow keras_tuner joblib plotly
```

