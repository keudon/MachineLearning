import pandas as pd
from sklearn.preprocessing import StandardScaler

def from_classifier(test_df,features,model):


    X_topredict = test_df[features].to_numpy()
    X_topredict = StandardScaler().fit_transform(X_topredict)
    y_predicted = model.predict(X_topredict)

    prediction_df = test_df[["PassengerId"]].copy()
    prediction_df = prediction_df.assign(Survived=y_predicted)
    prediction_df.to_csv('./deeplearning_submission.csv',index=False,header=True)

