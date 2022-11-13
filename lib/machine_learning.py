import numpy as np
from dataloader import transform
import pickle

THRESHOLD = 0.44


with open("data/model/LogisticRegression.pkl", "rb") as f:
    model = pickle.load(f)

def predict(
    InternetService: str,
    Contract: str,
    tenure: int,
    TotalCharges: float,
    MultipleLines: str,
    OnlineSecurity: str,
    PaymentMethod: str,
) -> str:
    features = transform(
        InternetService,
        Contract,
        tenure,
        TotalCharges,
        MultipleLines,
        OnlineSecurity,
        PaymentMethod,
    )
    
    input_array = np.array(list(features.values())).reshape(1,-1)
    pred_proba = model.predict_proba(input_array)
    pred = np.where(pred_proba[:, 0], "No", "Yes")[0]
    
    return pred

if __name__ == "__main__":
    print(predict("DSL", "One year", 3, 232345.22, "Yes", "No", "Mailed check"))