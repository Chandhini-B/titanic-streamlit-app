
import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess

st.title("Titanic Survival Predictor")

uploaded_file = st.file_uploader("Upload your test CSV (no target column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Input Preview")
    st.dataframe(df.head())

    try:
        poly = joblib.load('poly.pkl')
        model = joblib.load('model.pkl')

        df_processed = preprocess(df)
        df_poly = poly.transform(df_processed)

        predictions = model.predict(df_poly)
        df['Predicted_Survived'] = predictions

        st.success("Prediction complete!")
        st.dataframe(df[['PassengerId', 'Predicted_Survived']] if 'PassengerId' in df.columns else df)

        csv = df.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Oops: {e}")
