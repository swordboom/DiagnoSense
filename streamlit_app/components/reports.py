import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def render_reports_page():
    st.title("Evaluation Reports")
    st.markdown("---")
    
    report_path = os.path.join(REPORTS_DIR, "evaluation.md")
    
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
            
        st.download_button(
            label="Download Complete Report (Markdown)",
            data=report_content,
            file_name="evaluation_report.md",
            mime="text/markdown"
        )
        
        st.markdown(report_content)
    else:
        st.warning("Evaluation report not found. Ensure Phase 5 has been executed.")
