from io import StringIO
import pandas as pd
import streamlit as st
from tqdm import tqdm

from lib.machine_learning import predict


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(columns=["customerID", "Prediction", "Predict Proba"])

    for _, row in tqdm(df.iterrows()):
        pred_proba, pred = predict(
                    row["InternetService"],
                    row["Contract"],
                    row["tenure"],
                    row["TotalCharges"],
                    row["MultipleLines"],
                    row["OnlineSecurity"],
                    row["PaymentMethod"],
                )
        result = result.append(
            {
                "customerID":row["customerID"],
                "Prediction":pred,
                "Predict Proba":pred_proba,
            }, ignore_index=True
        )

    return result


uploaded_file = st.file_uploader("Upload new data", type="csv")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # Convert string based IO to Pandas DataFrame:
    dataframe = pd.read_csv(stringio)
    result = predict_dataframe(dataframe)
    st.write(result.astype(str))
    
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(result)

    st.download_button(
        "Press to Download",
        csv,
        "pred.csv",
        "text/csv",
        key='download-csv'
    )
