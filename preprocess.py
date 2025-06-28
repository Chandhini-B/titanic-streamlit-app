
import pandas as pd
import joblib
import re

def preprocess(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Cabin"] = df["Cabin"].fillna("U0")
    df["Name"] = df["Name"].fillna("Unknown")
    df["Ticket"] = df["Ticket"].fillna("X")

    df["deck"] = df["Cabin"].str[0].fillna('U')
    df["has_cabin"] = df["Cabin"].notnull().astype(int)
    df.drop(columns="Cabin", inplace=True)

    Q1 = df["Fare"].quantile(0.25)
    Q3 = df["Fare"].quantile(0.75)
    IQR = Q3 - Q1
    df["Fare"] = df["Fare"].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    df["Family_Size"] = df["Parch"] + df["SibSp"] + 1
    df.drop(columns=["Parch", "SibSp"], inplace=True)

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\s*\.", expand=False)
    df["Title"] = df["Title"].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    })
    df.drop(columns="Name", inplace=True)

    df["Ticket_frequency"] = df["Ticket"].map(df["Ticket"].value_counts())
    df.drop(columns="Ticket", inplace=True)

    ohe = joblib.load('Cat_encoder.pkl')
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    ohe_transformed = ohe.transform(df[cat_cols])
    ohe_df = pd.DataFrame(ohe_transformed, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df = df.drop(columns=cat_cols).join(ohe_df)

    scaler = joblib.load('scaler.pkl')
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = scaler.transform(df[num_cols])

    df.fillna(0, inplace=True)
    return df
