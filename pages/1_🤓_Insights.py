from typing import List, Tuple
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# import concurrent.futures
from tqdm import tqdm

from lib.dataloader import X, y, pre_selected_features
from lib.machine_learning import predict

st.header("Business Insights")
st.write("PLACEHOLDER")
st.header("Model Insights")

@st.cache
def get_predictions() -> Tuple[np.array, List[str]]:
    pred_proba = np.zeros(len(X), dtype=np.float64)
    pred = ["" for _ in range(len(X))]
    
    # futures = []
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for ix, row in X.iterrows():
    #         futures.append((ix, executor.submit(predict, *(row[pre_selected_features].values))))
            
    # for f in tqdm(futures):
    #     ix, temp = f[0], f[1].result()
    #     pred_proba[ix] = temp[0]
    #     pred[ix] = temp[1]
    
    for ix, row in tqdm(X.iterrows()):
        temp=predict(*(row[pre_selected_features].values))
        pred_proba[ix] = temp[0]
        pred[ix] = temp[1]
    
    return pred_proba, pred

with st.spinner("Crunching numbers..."):
    pred_proba, pred = get_predictions()

## CONFUSION MATRIX
def plot_confusion_matrix() -> plt.figure:
    fig, ax = plt.subplots()
    sns.heatmap(
        pd.DataFrame(confusion_matrix(y, pred), ["No", "Yes"], ["No", "Yes"]),
        annot=True,
        fmt=".0f",
        cbar=False,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Prediction")
    ax.set_title("Confusion Matrix")
    return fig
    
st.pyplot(plot_confusion_matrix())

## ROC_AUC CURVE
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y.map({"No": 1, "Yes": 0}), pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.set_title("Receiver Operating Characteristic")
    ax.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    return fig

st.pyplot(plot_roc_curve())

## PRECISION_RECALL CURVE
def plot_precision_recall_curve() -> plt.figure:
    precision, recall, thresholds = precision_recall_curve(
        y.map({"No": 1, "Yes": 0}), pred_proba
    )
    thresholds = np.append(thresholds, 1)
    fig, ax = plt.subplots()
    ax.step(recall, precision, color="b", alpha=0.4, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("Precision-Recall curve")
    return fig

st.pyplot(plot_precision_recall_curve())