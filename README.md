# Kaggle Store Sales Time Series Forecast

## How to run

### Set up the docker environment
```buildoutcfg
make build
make run
```

### Retrieve the dataset from Kaggle
Note: this step requires a kaggle.json to be placed in ~.kaggle/
You will need to rename kaggle.json to kaggle.priv.json or input the username and key into the existing `kaggle.priv.json` file.
```buildoutcfg
cd src/data
kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting
```

### Run the trainer
```buildoutcfg
cd ../..
python trainer.py
```