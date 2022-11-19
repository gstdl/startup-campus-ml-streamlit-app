from typing import List
import streamlit as st
import numpy as np
from lib.dataloader import X, y
from lib.machine_learning import predict
import concurrent.futures

st.header("Business Insights")
st.write("PLACEHOLDER")
st.header("Model Insights")

@st.cache
def get_predictions() -> List[np.array, List[str]]:
    pred_proba = np.zeros(len(X), dtype=np.float64)
    pred = []
    
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        

## CONFUSION MATRIX

## ROC_AUC CURVE

## PRECISION_RECALL CURVE