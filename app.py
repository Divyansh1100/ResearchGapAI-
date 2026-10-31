from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from researchgap.comparison import EVALUATION_DIMENSIONS, empty_score_state, evaluation_dataframe, score_summary
from researchgap.config import DEFAULT_BASE_URL, get_model_config
from researchgap.general_llm import run_general_llm
from researchgap.pdf_processor import extract_pdf_documents
from researchgap.specialized_workflow import run_specialized_pipeline

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PREDEFINED_QUESTIONS = [
    "What are the two broad categories of artificial intelligence in healthcare described in the paper?",
    "What are the five computational task categories used to classify healthcare AI applications?",
    "Explain the difference between LLMs and LMMs.",
    "What are the ten questions proposed for critically appraising an AI-based decision support system?",
    "Explain internal validity and external validity in AI research.",
    "What are data leakage, data poisoning and dataset drift?",
    "Compare TRIPOD+AI, STARD-AI, DECIDE-AI, SPIRIT-AI and CONSORT-AI.",
    "What are the six WHO ethical principles for AI in healthcare?",
    "Explain the three illusions associated with AI in scientific research.",
    "Construct a structured critical-appraisal framework based on the uploaded paper.",
]


st.set_page_config(page_title="ResearchGapAI", page_icon="🧠", layout="wide")


def initialize_session_state() -> None:
    st.session_state.setdefault("experiment_history", [])
    st.session_state.setdefault("uploaded_documents", [])
    st.session_state.setdefault("general_result", None)
    st.session_state.setdefault("specialized_result", None)
    st.session_state.setdefault("manual_scores", empty_score_state())
    st.session_state.setdefault("research_question", PREDEFINED_QUESTIONS[0])


initialize_session_state()


def render_css() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: #0f172a; color: #e2e8f0; }
            .block-container { padding-top: 1.5rem; }
            div[data-testid="stSidebar"] { background: #111827; }
            .metric-container { background: #111827; border: 1px solid #1f2937; border-radius: 0.75rem; padding: 0.75rem; }
            .stAlert { border-radius: 0.75rem; }
            .stButton > button { border-radius: 0.6rem; font-weight: 600; }
            .stSelectbox, .stTextInput, .stTextArea, .stNumberInput { background: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_css()


def build_record(pipeline_name: str, question: str, documents: List[Dict[str, Any]], config: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "experiment_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "document_names": [doc.get("filename", "unknown") for doc in documents],
        "research_question": question,
        "pipeline_name": pipeline_name,
        "model_name": result.get("model_name") or config.model_name,
        "base_url": config.base_url,
        "embedding_model": config.embedding_model,
        "answer": result.get("answer", ""),
        "retrieved_evidence": result.get("retrieved_evidence", ""),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "execution_time": result.get("execution_time", 0),
        "token_usage": json.dumps(result.get("usage", {}), ensure_ascii=False),
        "manual_scores": json.dumps(st.session_state.get("manual_scores", empty_score_state()), ensure_ascii=False),
        "evaluator_comments": "",
    }
    return record


def add_experiment_record(pipeline_name: str, question: str, documents: List[Dict[str, Any]], config: Any, result: Dict[str, Any]) -> None:
    history = st.session_state.setdefault("experiment_history", [])
    history.insert(0, build_record(pipeline_name, question, documents, config, result))


def get_uploaded_documents() -> List[Dict[str, Any]]:
    docs = st.session_state.get("uploaded_documents", [])
    return docs if isinstance(docs, list) else []


def get_question_value() -> str:
    selected = st.session_state.get("preset_question", "Custom question")
    if selected == "Custom question":
        return st.session_state.get("custom_question", "")
    return selected


def run_general_condition() -> None:
    question = get_question_value().strip()
    documents = get_uploaded_documents()
    if not question:
        st.warning("Please enter a research question before running the general-purpose LLM pipeline.")
        return
    if not documents:
        st.warning("Upload at least one PDF before running the experiment.")
        return

    config = get_model_config()
    if not config.has_llm:
        st.warning("No real LLM API configuration is present. Add LLM_API_KEY and LLM_MODEL to run live model calls.")
        return

    combined_text = "\n\n".join(doc.get("full_text", "") for doc in documents if doc.get("full_text"))
    if not combined_text.strip():
        st.error("No readable text was extracted from the PDF files. Check the uploaded documents or confirm they are not scanned-only images.")
        return

    try:
        with st.spinner("Running the general-purpose LLM pipeline..."):
            result = run_general_llm(question, combined_text, config)
        result["pipeline_name"] = "general"
        st.session_state["general_result"] = result
        add_experiment_record("general", question, documents, config, result)
        st.success("General-purpose pipeline completed.")
    except ValueError as exc:
        st.error(str(exc))


def run_specialized_condition() -> None:
    question = get_question_value().strip()
    documents = get_uploaded_documents()
    if not question:
        st.warning("Please enter a research question before running the specialized literature-analysis pipeline.")
        return
    if not documents:
        st.warning("Upload at least one PDF before running the specialized pipeline.")
        return

    config = get_model_config()
    if not config.has_llm:
        st.warning("No real LLM API configuration is present. Add LLM_API_KEY and LLM_MODEL to run live model calls.")
        return

    try:
        with st.spinner("Running the specialized literature-analysis pipeline..."):
            result = run_specialized_pipeline(question, documents, config, top_k=5)
        st.session_state["specialized_result"] = result
        add_experiment_record("specialized", question, documents, config, result)
        st.success("Specialized pipeline completed.")
    except ValueError as exc:
        st.error(str(exc))


def run_both_and_compare() -> None:
    run_general_condition()
    run_specialized_condition()


def render_pipeline_panel(title: str, result: Dict[str, Any] | None, accent: str) -> None:
    st.markdown(f"<div class='metric-container'><h3 style='color: {accent}; margin-bottom: 0;'> {title} </h3></div>", unsafe_allow_html=True)
    if result is None:
        st.info("No execution has been run for this condition yet.")
        return

    st.markdown(f"**Model:** {result.get('model_name', 'Not available')}")
    st.markdown(f"**Execution time:** {result.get('execution_time', 'n/a')} s")
    usage = result.get("usage", {})
    if usage:
        st.markdown(f"**Token usage:** {usage}")
    st.markdown("---")
    st.subheader("Answer")
    st.write(result.get("answer") or "No answer was returned.")

    if result.get("retrieved_chunks"):
        st.subheader("Retrieved evidence")
        st.caption(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
        for index, chunk in enumerate(result["retrieved_chunks"][:5], start=1):
            st.markdown(f"**{index}. {chunk.get('source_name', 'unknown')} | p.{chunk.get('page_number', '?')}**")
            st.write(chunk.get("text", ""))
        st.caption(f"Source references: {result.get('source_references', 'n/a')}")


def render_evaluation_section() -> None:
    st.header("Manual evaluation and comparison")
    st.caption("These scores are manual evaluations entered by the user and are not scientific validation results.")
    scores = st.session_state.get("manual_scores", empty_score_state())

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Pipeline A: General-purpose LLM")
        for dim in EVALUATION_DIMENSIONS:
            value = int(scores.get("general", {}).get(dim, 3) or 3)
            scores["general"][dim] = st.slider(dim.replace("_", " ").title(), 1, 5, value=value, key=f"general_{dim}")
        general_summary = score_summary(scores.get("general", {}))
        st.metric("Total", general_summary["total"])
        st.metric("Average", general_summary["average"])

    with col_right:
        st.subheader("Pipeline B: Specialized literature-analysis")
        for dim in EVALUATION_DIMENSIONS:
            value = int(scores.get("specialized", {}).get(dim, 3) or 3)
            scores["specialized"][dim] = st.slider(dim.replace("_", " ").title(), 1, 5, value=value, key=f"specialized_{dim}")
        specialized_summary = score_summary(scores.get("specialized", {}))
        st.metric("Total", specialized_summary["total"])
        st.metric("Average", specialized_summary["average"])

    st.session_state["manual_scores"] = scores

    comparison_rows = evaluation_dataframe(scores)
    if comparison_rows:
        st.subheader("Score summary")
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)


def export_experiments() -> None:
    records = st.session_state.get("experiment_history", [])
    if not records:
        st.warning("No experiment results are available for export yet.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"researchgapai_experiments_{timestamp}.csv"
    json_path = RESULTS_DIR / f"researchgapai_experiments_{timestamp}.json"

    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    st.success(f"Experiment log exported to {csv_path} and {json_path}.")


def main() -> None:
    st.title("ResearchGapAI")
    st.caption("Comparative research prototype for general-purpose vs literature-specialized scientific analysis.")

    with st.sidebar:
        st.header("API configuration")
        st.caption("Use a valid OpenAI-compatible endpoint. Never expose API keys or secrets in exported files.")
        config = get_model_config()
        api_key = st.text_input("LLM_API_KEY", type="password", value=config.api_key, help="Your provider API key.")
        base_url = st.text_input("LLM_BASE_URL", value=config.base_url or DEFAULT_BASE_URL)
        model_name = st.text_input("LLM_MODEL", value=config.model_name)
        embedding_model = st.text_input("EMBEDDING_MODEL", value=config.embedding_model)

        if not api_key or not model_name:
            st.warning("Live model execution is not available until LLM_API_KEY and LLM_MODEL are configured.")
        else:
            st.success("LLM configuration is present.")

        if st.button("Save configuration"):
            import os
            os.environ["LLM_API_KEY"] = api_key
            os.environ["LLM_BASE_URL"] = base_url
            os.environ["LLM_MODEL"] = model_name
            os.environ["EMBEDDING_MODEL"] = embedding_model
            st.success("Configuration stored for this session. The app will use these values for live calls.")

        st.markdown("---")
        st.caption("Clinical and scientific research prototype")
        st.caption("The application uses two execution conditions only: Pipeline A and Pipeline B.")

    st.header("Dashboard / Overview")
    st.markdown(
        "This app compares a general-purpose LLM against a specialized scientific literature-analysis workflow using the same uploaded PDF and the same research question."
    )

    st.header("PDF upload and corpus management")
    uploaded_files = st.file_uploader("Upload PDF research paper(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        processed_documents = extract_pdf_documents(list(uploaded_files))
        st.session_state["uploaded_documents"] = processed_documents
        document_summary = [
            {
                "filename": doc.get("filename", "unknown"),
                "status": doc.get("status", "uploaded"),
                "page_count": doc.get("page_count", 0),
                "chunks": len(doc.get("chunks", [])),
            }
            for doc in processed_documents
        ]
        st.dataframe(pd.DataFrame(document_summary), use_container_width=True)
        for doc in processed_documents:
            if doc.get("status") in {"warning", "error"}:
                st.warning(f"{doc.get('filename','document')}: {doc.get('error','')}" )
    else:
        st.info("Upload a PDF to enable the analysis workflows.")

    st.header("Research question selection")
    preset_question = st.selectbox("Use a predefined question", ["Custom question", *PREDEFINED_QUESTIONS], index=0)
    st.session_state["preset_question"] = preset_question

    if preset_question == "Custom question":
        custom_question = st.text_area("Enter a custom research question", value=st.session_state.get("custom_question", ""), height=120)
        st.session_state["custom_question"] = custom_question
        question = custom_question
    else:
        question = preset_question
        st.text_area("Research question", value=question, height=120, disabled=True)

    selected_question = question.strip()

    st.header("Execution conditions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Run General-Purpose LLM"):
            run_general_condition()
    with col_b:
        if st.button("Run Specialized Pipeline"):
            run_specialized_condition()

    if st.button("Run Both & Compare"):
        run_both_and_compare()

    st.header("Results comparison")
    left_col, right_col = st.columns(2)
    with left_col:
        render_pipeline_panel("Pipeline A: General-purpose LLM", st.session_state.get("general_result"), "#7dd3fc")
    with right_col:
        render_pipeline_panel("Pipeline B: Specialized literature-analysis", st.session_state.get("specialized_result"), "#a78bfa")

    render_evaluation_section()

    st.header("Retrieved evidence viewer")
    if st.session_state.get("specialized_result"):
        evidence = st.session_state["specialized_result"].get("retrieved_evidence", "")
        st.text_area("Evidence used by the specialized pipeline", value=evidence, height=220, disabled=True)
        st.caption(f"Source references: {st.session_state['specialized_result'].get('source_references', 'n/a')}")
    else:
        st.info("Run the specialized pipeline to populate the evidence viewer.")

    st.header("Experiment history")
    if st.session_state.get("experiment_history"):
        experiment_df = pd.DataFrame(st.session_state["experiment_history"])
        st.dataframe(
            experiment_df[["timestamp", "pipeline_name", "model_name", "research_question", "execution_time"]],
            use_container_width=True,
        )
    else:
        st.info("No experiments have been run yet.")

    st.header("CSV / JSON export")
    if st.button("Export experiment records"):
        export_experiments()
    st.caption("Only the recorded experiment metadata is exported. API keys are never exported.")


if __name__ == "__main__":
    main()
