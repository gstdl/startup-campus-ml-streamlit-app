import streamlit as st
from io import StringIO
import pandas as pd
from lib.machine_learning import predict
from lib.dataloader import pre_selected_features

def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(columns=["customerID", "Prediction Probability", "Prediction"])
    
    for _, row in df.iterrows():
        temp = predict(*(row[pre_selected_features].values))
        result = result.append({
            "customerID": row["customerID"],
            "Prediction Probability": temp[0],
            "Prediction": temp[1],
        }, ignore_index=True)
    return result

uploaded_file = st.file_uploader("Upload new data", type="csv")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    dataframe = pd.read_csv(stringio)
    result = predict_dataframe(dataframe)
    st.write(result)
    
    csv = result.to_csv(index=False)
    st.download_button(
        "Download predictions",
        csv,
        "pred.csv"
    )
    
    with st.expander("Explain Model"):
        st.write("PLACEHOLDER")