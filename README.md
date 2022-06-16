# MIN: Multi-dimensional Interest Network for Click-Through Rate Prediction
Our code is implemented based on DeepCTR (https://github.com/shenweichen/deepctr).


## Running
We use Python 3.7 and Tensorflow 2.0.

## Prepare Data
    You can get datasets from amazon website: reviews_Electronics_5.json, reviews_Movies_and_TV_5.json and reviews_Grocery_and_Gourmet_Food_5.json.The basic data processing process reference the DIEN (https://github.com/mouna99/dien) and we have unique processing steps.You can run
    ''
    prepare_data.ipynb
    ''
    Note:'samples.pkl' is a partial sample of dataset Grocery after processing.
  

## MIN Model
    The model operation entry is at
    ''
    main.ipynb
    ''
