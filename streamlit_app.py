import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import os
import re
from streamlit_mermaid import st_mermaid

# Config
API_URL = "http://127.0.0.1:8000"
BENCHMARK_AB_FILE = "benchmark_results.csv"
BENCHMARK_RETRIEVAL_FILE = "benchmark_retrieval.json"

st.set_page_config(page_title="RAG SRS Generator Dashboard", layout="wide", page_icon="📝")

def render_content_with_mermaid(content):
    """
    Splits markdown content by mermaid blocks and renders them using st_mermaid.
    """
    # Pattern to find ```mermaid ... ``` blocks
    # Using capturing group to keep the code content
    parts = re.split(r"```mermaid\s+(.*?)\s+```", content, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular Markdown
            if part.strip():
                st.markdown(part)
        else:
            # Mermaid Code
            # Calculate height based on line count roughly or fixed
            line_count = len(part.splitlines())
            height = max(200, line_count * 20)
            try:
                st_mermaid(part, height=height)
            except Exception:
                # Fallback code block if render fails
                st.code(part, language="mermaid")

st.title("🤖 SRS Generator with RAG - Demo & Analytics")

# Sidebar
st.sidebar.header("Configuration")
api_status = "Unknown"
try:
    # Simple check
    requests.get(f"{API_URL}/docs")
    api_status = "Online 🟢"
except:
    api_status = "Offline 🔴"
st.sidebar.write(f"API Status: **{api_status}**")

mode = st.sidebar.radio("Navigation", ["SRS Generator Demo", "Retrieval Debugger", "Analytics & Benchmarks"])

if mode == "SRS Generator Demo":
    st.header("📝 SRS Generation Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_desc = st.text_area("Project Description", height=150, 
            value="Xây dựng hệ thống quản lý kho (WMS) cho ngành bán lẻ. Yêu cầu chi tiết về quy tắc đặt mã SKU và quy trình nhập kho (PO, Receipt, Putaway).")
        
        use_rag = st.checkbox("Enable RAG (Retrieval Augmented Generation)", value=True)
        
        if st.button("Generate SRS", type="primary"):
            if api_status == "Offline 🔴":
                st.error("API is offline. Please start uvicorn backend.")
            else:
                with st.spinner("Generating SRS... (This may take 30-60s)"):
                    try:
                        payload = {"project_description": project_desc, "use_rag": use_rag}
                        response = requests.post(f"{API_URL}/generate-srs", json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            # Save to session state
                            st.session_state['last_srs'] = data.get("srs_content")
                            st.session_state['last_context'] = data.get("rag_context")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

        # Persistent Display using Session State
        if 'last_srs' in st.session_state:
            st.divider()
            st.subheader("Result")
            render_content_with_mermaid(st.session_state['last_srs'])
            
            rag_ctx = st.session_state.get('last_context')
            if use_rag and rag_ctx:
                with st.expander("📚 RAG Context (Sources Used)"):
                    st.text(rag_ctx)

    with col2:
        st.write("### AI Evaluation")
        if 'last_srs' in st.session_state:
            if st.button("Evaluate Last SRS"):
                with st.spinner("AI Judge is scoring..."):
                    try:
                        payload = {
                            "srs_content": st.session_state['last_srs'], 
                            "rag_context": st.session_state.get('last_context')
                        }
                        eval_resp = requests.post(f"{API_URL}/evaluate-srs", json=payload)
                        if eval_resp.status_code == 200:
                            res = eval_resp.json().get("evaluation_result", {})
                            score = res.get("score", {})
                            
                            # Validated Metrics
                            st.metric("Total Score", f"{score.get('total_weighted_score', 0):.2f}/10")
                            
                            st.write("#### Detailed Scores:")
                            metrics = {
                                "Faithfulness": score.get("faithfulness", {}).get("raw"),
                                "Accuracy": score.get("accuracy", {}).get("raw"),
                                "Completeness": score.get("completeness", {}).get("raw"),
                                "Consistency": score.get("consistency", {}).get("raw"),
                                "Format": score.get("format_tone", {}).get("raw"),
                            }
                            df_score = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
                            st.dataframe(df_score, hide_index=True)
                            
                            st.info(f"Group: **{res.get('group', 'N/A')}**")
                            
                            with st.expander("View Issues"):
                                st.json(res.get("comment", {}).get("issues", []))
                        else:
                            st.error("Evaluation Failed")
                    except Exception as e:
                        st.error(str(e))
        else:
            st.info("Generate an SRS first to enable evaluation.")


elif mode == "Retrieval Debugger":
    st.header("🔍 Retrieval Debugger")
    query = st.text_input("Enter Query to search Knowledge Base:", value="quy tắc nhập kho")
    top_k = st.slider("Top K", 1, 10, 5)
    
    if st.button("Search"):
        if api_status == "Offline 🔴":
            st.error("API is offline. Please start uvicorn backend.")
        else:
            with st.spinner("Searching Knowledge Base..."):
                try:
                    # Call the new /retrieve endpoint
                    # Note: We use query params for simplicity as defined in main.py
                    response = requests.post(f"{API_URL}/retrieve", params={"query": query, "top_k": top_k})
                    
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        st.subheader(f"Found {len(results)} chunks")
                        
                        for i, doc in enumerate(results):
                            score = doc.get("rerank_score", doc.get("initial_score", 0))
                            source = doc.get("metadata", {}).get("source", "Unknown")
                            content = doc.get("content", "")
                            
                            with st.expander(f"#{i+1} [{score:.4f}] {source}"):
                                st.markdown(f"**Relevance Score:** {score:.4f}")
                                st.markdown(f"**Source:** `{source}`")
                                st.text_area("Content", content, height=100)
                                st.json(doc.get("metadata", {}))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

elif mode == "Analytics & Benchmarks":
    st.header("📊 Analytics Dashboard")
    
    tab_ab, tab_ret = st.tabs(["🅰️/🅱️ RAG vs Vanilla", "🔎 Retrieval Metrics"])
    
    with tab_ab:
        st.subheader("A/B Testing Results")
        if os.path.exists(BENCHMARK_AB_FILE):
            df = pd.read_csv(BENCHMARK_AB_FILE)
            st.dataframe(df)
            
            # Compare Scores
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Mode', axis=None),
                y='Total Score',
                color='Mode',
                column='Case ID'
            ).properties(title="Total Score Comparison")
            # use_container_width=True is deprecated in some versions, but standard in others.
            st.altair_chart(chart, use_container_width=True)
            
            # Compare Quality
            chart_q = alt.Chart(df).mark_bar().encode(
                x=alt.X('Mode', axis=None),
                y='Quality Score (Shared)',
                color='Mode',
                column='Case ID'
            ).properties(title="Quality Score (Content Only)")
            st.altair_chart(chart_q, use_container_width=True)
            
        else:
            st.warning(f"No benchmark data found at {BENCHMARK_AB_FILE}. Run `benchmark_ab.py` first.")

    with tab_ret:
        st.subheader("Retrieval Performance (Hit Rate & MRR)")
        if os.path.exists(BENCHMARK_RETRIEVAL_FILE):
            with open(BENCHMARK_RETRIEVAL_FILE, "r") as f:
                ret_data = json.load(f)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Hit Rate", f"{ret_data['hit_rate']:.2%}")
            c2.metric("MRR", f"{ret_data['mrr']:.4f}")
            c3.metric("Avg Latency", f"{ret_data['avg_latency']:.4f}s")
            
            st.json(ret_data)
        else:
            st.warning(f"No retrieval data found at {BENCHMARK_RETRIEVAL_FILE}. Run `benchmark_retrieval.py` first.")
