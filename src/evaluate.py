import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/dibahk/Machine-Learning-Pipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "dibahk"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "4f087248fdd4b1542c5af6ee0c98b3e76a9cc372"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns= ["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/dibahk/Machine-Learning-Pipeline.mlflow")

    model = pickle.load(open(model_path, "rb"))

    predictions = model.predict(x)
    accuracy = accuracy_score(y, predictions)

    mlflow.log_metric("accuract", accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__ =="__main__":
    evaluate(data_path= params["data"], model_path= params["model"])