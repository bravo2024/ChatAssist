from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["message_length","sentiment_score","response_time_secs","user_satisfaction","intent_confidence","topic_category","language_model_score","context_relevance","escalation_flag","session_depth"]
CATEGORICAL_FEATURES = ["topic_category"]
NUMERICAL_FEATURES = ["message_length","sentiment_score","response_time_secs","user_satisfaction","intent_confidence","language_model_score","context_relevance","escalation_flag","session_depth"]
TARGET_NAME = "resolution_success"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "message_length": rng.poisson(lam=80,size=n).clip(1,500).astype(int),
        "sentiment_score": rng.uniform(-1,1,size=n).round(3),
        "response_time_secs": rng.exponential(scale=30,size=n).clip(0.5,300).round(1),
        "user_satisfaction": rng.beta(4,2,size=n).round(3),
        "intent_confidence": rng.beta(6,2,size=n).round(3),
        "topic_category": rng.choice(["billing","technical","account","general","feedback","escalation"],size=n,p=[0.20,0.25,0.15,0.20,0.10,0.10]),
        "language_model_score": rng.beta(7,2,size=n).round(3),
        "context_relevance": rng.beta(5,2,size=n).round(3),
        "escalation_flag": rng.choice([0,1],size=n,p=[0.80,0.20]),
        "session_depth": rng.poisson(lam=4,size=n).clip(1,20),
    })
    sent=df["sentiment_score"]; sat=df["user_satisfaction"]; conf=df["intent_confidence"]
    lm=df["language_model_score"]; ctx=df["context_relevance"]; esc=df["escalation_flag"]
    resp=np.clip(df["response_time_secs"]/300,0,1); depth=np.clip(df["session_depth"]/20,0,1)
    topic=df["topic_category"].map({"billing":0,"technical":0.2,"account":0.4,"general":0.5,"feedback":0.7,"escalation":1}).values
    log_odds = 2.0 + 0.3*sent + 0.5*sat + 0.4*conf + 0.3*lm + 0.4*ctx - 0.3*esc - 0.2*resp + 0.1*depth - 0.2*topic + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,75)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(resolution_success=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
