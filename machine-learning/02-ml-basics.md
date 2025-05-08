# Machine Learning Basics

## Supervised vs Unsupervised

Machine learning methods are categorized in two types depending of the existence of the label data in the training dataset. On key deifference bteween these types is the level of supervision during the training fase.

### Supervised

- Required training data with independent variables and a dependent variables (labelled data)
- Need labelled data that can `supervise` the algortihm when learning from the data
- Includes regressing and classification (depending of the type of dependent variable)

### Regresion

- Can be used the response variable to predicted is a continous variable (scaler), such as price, probability, etc
- Evaluation and performance metrics (measure teh difference between the predicted values and the true values with the lower value indicating a better fit for the model)
    - Residual sum suare (RSS)
    - Mean square error (MSE)
    - Root mean square error (RMSE)
    - Mean abosulute error (MAE)
- Used for: prediction
- Examples: linear regression, fixed effects regression, XGBoost Regression

#### Classification

- Can be used the response variable to predicted is a categorized (categorical) variable
- Evaluation and performance metrics (measure the ability of the machine learning model to correctly classify instances into the correct categories)
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Used for: decision-making
- Examples: Logistic regression, XGBoost classification, random forest classification

### Unsupervised

- Requires training data with independent variables only
- No need labelled data that can "supervise" the algorithm when learning from the data
- Includes clustering and outlier detection models
- Evaluation and performance metrics (measure the similarity of the data points within a cluster and the disimilarity of the data points between different clusters)
  - Homogeniety
  - Silouette score
  - Completeness

## Training & Evaluating ML models

1. ***Data preparation***: Split data into train, validation and test.
2. ***Model training***: Train the model on the training data and save the fitted model.
3. ***Hyper-parameter tuning***: Use the fitted model and validation set to find the optimal set of parameter where the model performs the best.
4. ***Prediction***: Use the optimal set of parameters from Hyper-Parameter tuning stage and training data, to train the model again with these hyper parameters, use this best fitted model to predictions on test data.
5. ***Test error rate***: Compute the performance metrics for your model using the predictions and real values of the target variable from your test data.

1:05:22
