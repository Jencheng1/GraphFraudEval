import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import json
from datetime import datetime

# Constants
API_URL = "http://localhost:8000"

def load_available_models() -> Dict[str, List[str]]:
    response = requests.get(f"{API_URL}/api/models")
    return response.json()

def upload_document(file) -> Dict[str, Any]:
    files = {"file": file}
    response = requests.post(f"{API_URL}/api/upload", files=files)
    return response.json()

def evaluate_models(queries: List[str], model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    response = requests.post(
        f"{API_URL}/api/evaluate",
        json={"queries": queries, "model_configs": model_configs}
    )
    return response.json()

def main():
    st.set_page_config(
        page_title="GraphFraudEval - Fraud Detection Evaluation",
        page_icon="🔍",
        layout="wide"
    )

    st.title("GraphFraudEval")
    st.markdown("""
    A comprehensive framework for evaluating and analyzing fraud detection systems using GraphRAG and LLM evaluation.
    This tool helps you assess the performance of different LLM models and embedding models in fraud detection scenarios.
    """)

    # Sidebar
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Document for RAG",
        type=["txt", "pdf", "md"]
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            result = upload_document(uploaded_file)
            st.sidebar.success(f"Processed {result['num_chunks']} chunks")

    # Model selection
    available_models = load_available_models()
    
    selected_llm = st.sidebar.selectbox(
        "Select LLM Model",
        available_models["llm_models"]
    )
    
    selected_embedding = st.sidebar.selectbox(
        "Select Embedding Model",
        available_models["embedding_models"]
    )

    # Main content
    tab1, tab2 = st.tabs(["Single Query", "Batch Evaluation"])

    with tab1:
        st.header("Single Query Evaluation")
        query = st.text_area("Enter your query:")
        
        if st.button("Evaluate"):
            if not query:
                st.error("Please enter a query")
            else:
                with st.spinner("Processing..."):
                    model_config = {
                        "llm_model": selected_llm,
                        "embedding_model": selected_embedding,
                        "temperature": 0.7,
                        "max_tokens": 500,
                        "top_k": 3
                    }
                    
                    response = requests.post(
                        f"{API_URL}/api/query",
                        json={"query": query, "model_config": model_config}
                    )
                    
                    result = response.json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Response")
                        st.write(result["response"])
                        
                        st.subheader("Retrieval Context")
                        for i, context in enumerate(result["retrieval_context"], 1):
                            st.markdown(f"**Context {i}:**")
                            st.write(context)
                    
                    with col2:
                        st.subheader("Metrics")
                        st.metric("Latency", f"{result['latency']:.2f}s")
                        st.metric("Model", result["model"])
                        st.metric("Embedding Model", result["embedding_model"])
                        
                        if "llm_evaluation" in result:
                            st.subheader("LLM-as-Judge Evaluation")
                            st.markdown("**Quality Evaluation:**")
                            st.write(result["llm_evaluation"]["quality_evaluation"])
                            st.markdown("**Factuality Evaluation:**")
                            st.write(result["llm_evaluation"]["factuality_evaluation"])

    with tab2:
        st.header("Batch Evaluation")
        
        # Query input
        queries = st.text_area(
            "Enter queries (one per line):",
            height=150
        ).split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        
        # Model configurations
        st.subheader("Model Configurations")
        num_models = st.number_input("Number of models to compare", min_value=1, max_value=5, value=2)
        
        model_configs = []
        for i in range(num_models):
            st.markdown(f"### Model {i+1}")
            col1, col2 = st.columns(2)
            with col1:
                llm = st.selectbox(f"LLM Model {i+1}", available_models["llm_models"])
            with col2:
                embedding = st.selectbox(f"Embedding Model {i+1}", available_models["embedding_models"])
            
            model_configs.append({
                "llm_model": llm,
                "embedding_model": embedding,
                "temperature": 0.7,
                "max_tokens": 500,
                "top_k": 3
            })
        
        if st.button("Run Batch Evaluation"):
            if not queries:
                st.error("Please enter at least one query")
            else:
                with st.spinner("Running evaluation..."):
                    results = evaluate_models(queries, model_configs)
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    
                    # Metrics comparison
                    metrics = results["comparison_metrics"]
                    
                    # Create comparison charts
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fig_latency = px.bar(
                            x=list(metrics["average_latency"].keys()),
                            y=list(metrics["average_latency"].values()),
                            title="Average Latency by Model"
                        )
                        st.plotly_chart(fig_latency, use_container_width=True)
                    
                    with col2:
                        fig_precision = px.bar(
                            x=list(metrics["contextual_precision"].keys()),
                            y=list(metrics["contextual_precision"].values()),
                            title="Contextual Precision by Model"
                        )
                        st.plotly_chart(fig_precision, use_container_width=True)
                    
                    with col3:
                        fig_recall = px.bar(
                            x=list(metrics["contextual_recall"].keys()),
                            y=list(metrics["contextual_recall"].values()),
                            title="Contextual Recall by Model"
                        )
                        st.plotly_chart(fig_recall, use_container_width=True)
                    
                    # Additional metrics
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        fig_relevancy = px.bar(
                            x=list(metrics["contextual_relevancy"].keys()),
                            y=list(metrics["contextual_relevancy"].values()),
                            title="Contextual Relevancy by Model"
                        )
                        st.plotly_chart(fig_relevancy, use_container_width=True)
                    
                    with col5:
                        fig_answer = px.bar(
                            x=list(metrics["answer_relevancy"].keys()),
                            y=list(metrics["answer_relevancy"].values()),
                            title="Answer Relevancy by Model"
                        )
                        st.plotly_chart(fig_answer, use_container_width=True)
                    
                    with col6:
                        fig_faithfulness = px.bar(
                            x=list(metrics["faithfulness"].keys()),
                            y=list(metrics["faithfulness"].values()),
                            title="Faithfulness by Model"
                        )
                        st.plotly_chart(fig_faithfulness, use_container_width=True)
                    
                    # LLM-as-judge metrics
                    st.subheader("LLM-as-Judge Evaluation")
                    col7, col8 = st.columns(2)
                    
                    with col7:
                        fig_quality = px.bar(
                            x=list(metrics["llm_judge_quality"].keys()),
                            y=list(metrics["llm_judge_quality"].values()),
                            title="Quality Score by Model"
                        )
                        st.plotly_chart(fig_quality, use_container_width=True)
                    
                    with col8:
                        fig_factuality = px.bar(
                            x=list(metrics["llm_judge_factuality"].keys()),
                            y=list(metrics["llm_judge_factuality"].values()),
                            title="Factuality Score by Model"
                        )
                        st.plotly_chart(fig_factuality, use_container_width=True)
                    
                    # Detailed results table
                    st.subheader("Detailed Results")
                    detailed_results = []
                    for model_result in results["model_results"]:
                        for query_result in model_result["queries"]:
                            detailed_results.append({
                                "Model": model_result["model"],
                                "Embedding": model_result["embedding_model"],
                                "Query": query_result["query"],
                                "Response": query_result["response"],
                                "Latency": f"{query_result['latency']:.2f}s",
                                "Cost": f"${query_result['cost']:.4f}",
                                "Quality Evaluation": query_result.get("llm_evaluation", {}).get("quality_evaluation", ""),
                                "Factuality Evaluation": query_result.get("llm_evaluation", {}).get("factuality_evaluation", "")
                            })
                    
                    df = pd.DataFrame(detailed_results)
                    st.dataframe(df)

if __name__ == "__main__":
    main() 