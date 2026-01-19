import streamlit as st
from SearchAgent import ingest_documents, answer_questions
from pypdf import PdfReader

st.set_page_config(page_title="Document QA with RAG", layout="wide")

st.title("📚 RAG Document Question-Answering System")
st.write("Upload PDF or text documents and ask questions based on them.")

# ---------------------------------------------------------
# 1. File Upload Section (PDF + TEXT)
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload documents (PDF, TXT, MD)", 
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

def extract_pdf_text(uploaded_pdf):
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            
            texts = []
            for file in uploaded_files:
                if file.type == "application/pdf":
                    extracted_text = extract_pdf_text(file)
                    texts.append(extracted_text)

                else:   # TXT or MD
                    texts.append(file.read().decode("utf-8"))

            total_chunks = ingest_documents(texts)

        st.success(f"🎉 Done! Indexed {total_chunks} chunks into the vector DB.")

# ---------------------------------------------------------
# 2. Query Section
# ---------------------------------------------------------

st.subheader("🔍 Ask a question")
query = st.text_input("Type your question here")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            answer, context = answer_questions(query)

        st.markdown("## 🧠 Answer")
        st.write(answer)

        st.markdown("## 📄 Retrieved Context")
        for c in context:
            st.markdown(f"""
            **Score:** {c.get('score_after', 0):.4f}  
            <details><summary>Show Chunk</summary>
            {c['text']}
            </details>
            """, unsafe_allow_html=True)
