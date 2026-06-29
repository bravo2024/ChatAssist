from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="ChatAssist | Yellow.ai Conversational AI", layout="wide", page_icon="\U0001f4ac")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Yellow.ai | Conversational AI | Intent Classification")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Sessions",f"{n:,}"); c2.metric("Resolution Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f3af Intent Analysis","\U0001f4a1 Insights"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Unresolved","Resolved"],[1-data["positive_rate"],data["positive_rate"]],color=["#f43f5e","#22c55e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Resolution Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Intent Distribution & Performance")
    intent_dist=data["df"]["topic_category"].value_counts()
    colors=["#22d3ee","#f97316","#22c55e","#a78bfa","#fbbf24","#f43f5e"]
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    ax.pie(intent_dist.values,labels=intent_dist.index,autopct="%1.1f%%",colors=colors,textprops={"color":"white"})
    ax.set_title("Conversation Topics",color="white")
    st.pyplot(fig)
    col_a,col_b=st.columns(2)
    with col_a:
        res_by_topic=data["df"].groupby("topic_category")[TARGET_NAME].mean().sort_values()
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        ax.barh(res_by_topic.index,res_by_topic.values,color="#22d3ee")
        ax.set_title("Resolution Rate by Topic",color="white"); ax.set_xlabel("Resolution Rate"); ax.grid(True,alpha=.2,axis="x")
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        resolved=data["df"][data["df"]["resolution_success"]==1]["sentiment_score"]
        unresolved=data["df"][data["df"]["resolution_success"]==0]["sentiment_score"]
        ax.hist(resolved,bins=30,alpha=0.5,color="#22c55e",label="Resolved",density=True)
        ax.hist(unresolved,bins=30,alpha=0.5,color="#f43f5e",label="Unresolved",density=True)
        ax.set_title("Sentiment Score by Resolution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
with t4:
    st.subheader("Conversation Insights")
    st.latex(r"\text{Satisfaction} = \sigma(\beta_0 + \beta_{\text{conf}} \cdot \text{Conf} + \beta_{\text{ctx}} \cdot \text{Context})")
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        resolved=data["df"][data["df"]["resolution_success"]==1]["intent_confidence"]
        unresolved=data["df"][data["df"]["resolution_success"]==0]["intent_confidence"]
        ax.hist(resolved,bins=30,alpha=0.5,color="#22c55e",label="Resolved",density=True)
        ax.hist(unresolved,bins=30,alpha=0.5,color="#f43f5e",label="Unresolved",density=True)
        ax.set_title("Intent Confidence",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        resolved=data["df"][data["df"]["resolution_success"]==1]["context_relevance"]
        unresolved=data["df"][data["df"]["resolution_success"]==0]["context_relevance"]
        ax.hist(resolved,bins=30,alpha=0.5,color="#22c55e",label="Resolved",density=True)
        ax.hist(unresolved,bins=30,alpha=0.5,color="#f43f5e",label="Unresolved",density=True)
        ax.set_title("Context Relevance",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    st.subheader("Escalation Patterns")
    esc_res=data["df"].groupby("escalation_flag")[TARGET_NAME].mean()
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["No Escalation","Escalated"],esc_res.values,color=["#22c55e","#f97316"])
    ax.set_title("Resolution Rate by Escalation",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
