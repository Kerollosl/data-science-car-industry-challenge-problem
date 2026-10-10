"""Predict vehicle price and trim using training-only preprocessing."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,f1_score
import joblib
TARGETS=["Dealer_Listing_Price","Vehicle_Trim"]

def preprocess(frame):
    categorical=list(frame.select_dtypes(include=['object','string','bool']).columns)
    numeric=[c for c in frame if c not in categorical]
    return ColumnTransformer([
        ("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",min_frequency=5))]),categorical),
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True))]),numeric)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train",type=Path,default=Path("Training_DataSet.csv"));ap.add_argument("--test",type=Path,default=Path("Test_Dataset.csv"));ap.add_argument("--output",type=Path,default=Path("artifacts"));ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args();df=pd.read_csv(args.train).dropna(subset=TARGETS)
    features=df.drop(columns=TARGETS+['ListingID'],errors="ignore")
    # Free-form descriptions/notes can leak trim or price and create huge vocabularies.
    features=features.drop(columns=['VehSellerNotes','SellerName','VehFeats','VehPriceLabel'],errors="ignore")
    train_indices,val_indices=train_test_split(np.arange(len(df)),test_size=.15,random_state=args.seed)
    x=features.iloc[train_indices];vx=features.iloc[val_indices];models={};results={}
    for target,estimator in [(TARGETS[0],RandomForestRegressor(n_estimators=200,random_state=args.seed,n_jobs=-1)),(TARGETS[1],RandomForestClassifier(n_estimators=200,random_state=args.seed,n_jobs=-1,class_weight="balanced"))]:
        model=Pipeline([("prepare",preprocess(x)),("model",estimator)]);y=df[target].iloc[train_indices];vy=df[target].iloc[val_indices]
        model.fit(x,y);pred=model.predict(vx);models[target]=model
        if target==TARGETS[0]:results[target]={"mae":float(mean_absolute_error(vy,pred)),"rmse":float(np.sqrt(mean_squared_error(vy,pred))),"median_baseline_mae":float(mean_absolute_error(vy,np.repeat(y.median(),len(vy))))}
        else:results[target]={"accuracy":float(accuracy_score(vy,pred)),"macro_f1":float(f1_score(vy,pred,average="macro",zero_division=0)),"majority_baseline_accuracy":float((vy==y.mode().iloc[0]).mean())}
    args.output.mkdir(parents=True,exist_ok=True);joblib.dump({"models":models,"columns":list(features),"seed":args.seed},args.output/"pipelines.joblib")
    (args.output/"metrics.json").write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
    if args.test.exists():
        test=pd.read_csv(args.test);tx=test.reindex(columns=features.columns)
        output=pd.DataFrame({"ListingID":test.get("ListingID",test.index)})
        for target,model in models.items():output[target]=model.predict(tx)
        output.to_csv(args.output/"predictions.csv",index=False)
if __name__=="__main__":main()
