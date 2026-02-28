import streamlit as st
from backend.rag import load_main_db, process_uploaded_pdf
from backend.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

# ---------------- UI ----------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Intelligent Healthcare Assistant")

# ---------------- LOAD LLM ----------------
llm = get_llm()

# ---------------- LOAD MAIN BOOK DB ----------------
if "main_db" not in st.session_state:
    st.session_state.main_db = load_main_db()

# ---------------- SIDEBAR UPLOAD ----------------
with st.sidebar:
    st.header("Upload Patient Report")
    uploaded_pdf = st.file_uploader("Upload MRI / Blood Report", type="pdf")

    if uploaded_pdf:
        st.session_state.report_db = process_uploaded_pdf(uploaded_pdf)
        st.success("Report uploaded successfully!")

# ---------------- CHAT MEMORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
query = st.chat_input("Ask a medical question")

if query:
    st.session_state.chat.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # ---------------- RETRIEVERS ----------------
    main_retriever = st.session_state.main_db.as_retriever(
        search_kwargs={"k": 5}
    )

    docs1 = main_retriever.get_relevant_documents(query)

    # If report uploaded
    if "report_db" in st.session_state:
        report_retriever = st.session_state.report_db.as_retriever(
            search_kwargs={"k": 4}
        )
        docs2 = report_retriever.get_relevant_documents(query)

        # 🔥 PRIORITY LOGIC
        if "report" in query.lower() or "mri" in query.lower() or "scan" in query.lower():
            docs = docs2 + docs1   # report first
        else:
            docs = docs1 + docs2
    else:
        docs = docs1

    # ---------------- FORMAT CONTEXT ----------------
    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            formatted.append(f"(Page {page}) {doc.page_content}")
        return "\n\n".join(formatted)

    context = format_docs(docs)

    # ---------------- PROMPT ----------------
    prompt = ChatPromptTemplate.from_template("""
You are an advanced AI medical assistant.

If a patient report is uploaded:
- Focus on report findings first
- Use medical knowledge as support

Context:
{context}

Question:
{question}

Provide:
1. Clear explanation
2. If report exists → summarize findings first
3. Severity insight
4. Suggested next steps
5. Mention page numbers if available

End with:
"I am not a doctor. Please consult a healthcare professional."
""")

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    # ---------------- LLM RESPONSE ----------------
    response = llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.write(response.content)

    st.session_state.chat.append(
        {"role": "assistant", "content": response.content}
    )