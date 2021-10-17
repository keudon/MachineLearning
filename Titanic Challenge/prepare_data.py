import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def titanic_data_shaping(path_to_df):

    original_df = pd.read_csv(path_to_df)
    df = None
    df = original_df.copy()

    # Sex
    df.Sex = df.Sex.map({'female':0,'male':1})
    df.Sex = df.Sex.fillna(0)

    # Title
    # Regexp tips :
    # () : matches the group within
    # + : matches the expression to its left one or more times
    # \. search for '.' otherwise . alone is a special character
    # '([A-Za-z]+)\.' search for matches to groups of letters ending with a .

    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
    least_occuring_titles = ['Dr','Rev','Major','Col','Capt','Jonkheer','Don','Sir']
    df['Feature_Title']=df['Title']
    df['Feature_Title']=df['Feature_Title'].replace(['Ms','Mme','Countess'],'Mrs')
    df['Feature_Title']=df['Feature_Title'].replace(['Lady','Mlle'],'Miss')
    df['Feature_Title']=df['Feature_Title'].replace(least_occuring_titles,'Rare')
    df['Feature_Title'] = df['Feature_Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4},na_action='ignore')
    df['Feature_Title'] = df['Feature_Title'].fillna(4)

    df["Feature_Ticket_Number"] = [int(df.iloc[i]["Ticket"].split()[-1]) if df.iloc[i]["Ticket"].split()[-1] != 'LINE' else 0 for i in range(len(df)) ]
    df["Feature_Ticket_Number"] = (df["Feature_Ticket_Number"] - df["Feature_Ticket_Number"].mean()) / (df["Feature_Ticket_Number"].max() - df["Feature_Ticket_Number"].min())

    # Training Data shuffle
    df = df.reindex(np.random.permutation(df.index))

    # Age
    df['Feature_Age'] = df['Age'].fillna(df.groupby(['Pclass','SibSp'])['Age'].transform('mean'),inplace=False)
    auxage = pd.cut(df['Feature_Age'], 10)
    df['Feature_Age'] = LabelEncoder().fit_transform(auxage)

    # Embarked 
    df.Embarked = df.Embarked.map({'S':0,'C':1,'Q':2})
    df.Embarked = df.Embarked.fillna(0)

    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean)

    return df
