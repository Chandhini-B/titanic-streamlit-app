
import pandas as pd
import joblib
import re

def preprocess(df):
    df["deck"] = df["Cabin"].str[0].fillna('U')
    df["has_cabin"] = df["Cabin"].notnull().astype(int)
    df.drop(columns="Cabin", inplace=True)

    df.fillna({
        "Age": df["Age"].median(),
        "Embarked": df["Embarked"].mode()[0]
    }, inplace=True)

    Q1 = df["Fare"].quantile(0.25)
    Q3 = df["Fare"].quantile(0.75)
    IQR = Q3 - Q1
    df["Fare"] = df["Fare"].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    df["Family_Size"] = df["Parch"] + df["SibSp"] + 1
    df.drop(columns=["Parch", "SibSp"], inplace=True)

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\s*\.")
    df["Title"] = df["Title"].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    })
    df.drop("Name", axis=1, inplace=True)

    df["Ticket_frequency"] = df["Ticket"].map(df["Ticket"].value_counts())
    df.drop("Ticket", axis=1, inplace=True)

    ohe = joblib.load('Cat_encoder.pkl')
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    ohe_transformed = ohe.transform(df[cat_cols])
    ohe_df = pd.DataFrame(ohe_transformed, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df = df.drop(columns=cat_cols).join(ohe_df)

    scaler = joblib.load('scaler.pkl')
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = scaler.transform(df[num_cols])

    return df
