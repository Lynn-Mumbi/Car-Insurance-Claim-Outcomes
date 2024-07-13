# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# Start coding!
car_ins=pd.read_csv("car_insurance.csv")
print(car_ins.head(3))

#dealing with missing values
print(car_ins.isna().sum())
car_ins["credit_score"].fillna(car_ins["credit_score"].mean(),inplace=True)
car_ins["annual_mileage"].fillna(car_ins["credit_score"].mean(),inplace = True)
print(car_ins.isna().sum().sort_values())

#creating empty list to store model results
models = []

#get all the feature columns
features = car_ins.drop(columns=["id", "outcome"]).columns

#create a loop to loop through the features
for col in features:
    # The model
    model = logit(f"outcome ~ {col}", data=car_ins).fit()
    # Add each model to the empty list
    models.append(model)
print(len(models))
# creating empty list to store accuracies
accuracies = []

#now we loop through models
for feature in range(0, len(models)):
    # Compute the confusion matrix
    conf_matrix = models[feature].pred_table()
    # True negatives
    tn = conf_matrix[0,0]
    # True positives
    tp = conf_matrix[1,1]
    # False negatives
    fn = conf_matrix[1,0]
    # False positives
    fp = conf_matrix[0,1]
    # Compute accuracy
    acc = (tn + tp) / (tn + fn + fp + tp)
    accuracies.append(acc)

# Find the feature with the largest accuracy
best_feature = features[accuracies.index(max(accuracies))]

# Create best_feature_df
best_feature_df = pd.DataFrame({"best_feature": best_feature,
                                "best_accuracy": max(accuracies)},
                                index=[0])
print(best_feature_df)