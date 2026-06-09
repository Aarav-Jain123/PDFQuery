import os
import tempfile
from pathlib import Path
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import concurrent.futures
from concurrent.futures import as_completed

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1:7b"

CHROMA_DIR = "chroma_db"

PROMPT_TEMPLATE = """
You are an advanced educational assistant.

Your job:
1. Generate keywords when explicitly asked.
2. Solve doubts with deep conceptual clarity.
3. Explain concepts step-by-step.
4. Use simple language first, then deepen understanding.
5. Use ONLY the provided context.

Rules:
- Do NOT hallucinate.
- Do NOT regenerate keywords unless asked.
- Be concise but educational.
- Focus on understanding, not verbosity.

<context>
{context}
</context>

User Question:
{input}
"""

def save_uploaded_pdf(uploaded_file):

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

def build_retrieval_chain(pdf_path):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    # Vector DB
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    # LLM
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.2
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    # QA chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


def initialize_session():
    """
    Initialize Streamlit session state.
    """

    defaults = {
        # One-time generation trigger: set True after Load PDF, cleared after run
        "generation_needed": False,
        "loaded": False,
        "keywords_generated": False,
        "retrieval_chain": None,
        "pdf_path": None,
        "pdf_name": None,
        "keywords": None,
        "mcq": None,
        "subjective_one_marks": None,
        "assertion_reason": None,
        "short_questions": None,
        "long_four_marks": None,
        "long_five_marks": None,
        "very_long_questions": None,
        "case_based_questions": None,
        "answer": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    """
    Reset app state when new PDF uploaded.
    """

    st.session_state.loaded = False
    st.session_state.retrieval_chain = None
    st.session_state.generation_needed = False
    st.session_state.keywords = None
    st.session_state.mcq = None
    st.session_state.subjective_one_marks = None
    st.session_state.assertion_reason = None
    st.session_state.short_questions = None
    st.session_state.long_four_marks = None
    st.session_state.very_long_questions = None
    st.session_state.case_based_questions = None
    st.session_state.answer = None


def generate_keywords(chain):
        query = """
        Generate 20 important keywords from the PDF.
        For each keyword:
        - give a short definition
        - keep it student friendly
        """

        result = chain.invoke({"input": query})
        return result.get("answer", "No answer generated.")
    

def generate_mcq(chain):
        query = """
        Generate 20 important multiple choice questions varying in easy to hard from the PDF.
        """

        result = chain.invoke({"input": query})
        return result.get("answer", "No keywords generated.")


def generate_subjective_one_marks(chain):
        query = """
        Generate 20 important subjective one marks questions varying in easy to hard from the PDF.
        """

        result = chain.invoke({"input": query})
        return result.get("answer", "No keywords generated.")
    

def generate_assertion_reason(chain):
            query = """
            Generate 5 important Assertion-Reason questions varying in easy to hard from the PDF.
            """

            result = chain.invoke({"input": query})
            return result.get("answer", "No keywords generated.")
    

def generate_short_questions(chain):
            query = """
            Generate 5 important subjective 2 marks questions and 5 subjective 3 marks question varying in easy to hard from the PDF.
            write "(2)" in front of 2 marks questions and "(3)" in front of 3 marks questions
            """

            result = chain.invoke({"input": query})
            return result.get("answer", "No keywords generated.")


def generate_long_questions(chain):
            query = """
            Generate 5 important subjective 4 marks questions and 5 subjective 5 marks question varying in easy to hard from the PDF.
            write "(4)" in front of 4 marks questions and "(5)" in front of 5 marks questions
            """

            result = chain.invoke({"input": query})
            return result.get("answer", "No keywords generated.")

def generate_very_long_questions(chain):
            query = """
            Generate two 8 marks question from the PDF.
            """

            result = chain.invoke({"input": query})
            return result.get("answer", "No keywords generated.")


def generate_case_based_questions(chain):
            query = """
            Generate 2 cased based questions questions from the PDF.
            """

            result = chain.invoke({"input": query})
            return result.get("answer", "No keywords generated.")


def main():

    st.set_page_config(
        page_title="PDF Query",
        page_icon="📘",
        layout="wide"
    )

    initialize_session()

    st.title("📘 PDF Query")

    st.write(
        "Upload a PDF and ask conceptual questions."
    )


    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        # Prevent duplicate reloads
        if (
            st.session_state.pdf_name
            != uploaded_file.name
        ):

            reset_session()

            pdf_path = save_uploaded_pdf(uploaded_file)

            st.session_state.pdf_path = pdf_path
            st.session_state.pdf_name = uploaded_file.name

            st.success(
                f"Uploaded: {uploaded_file.name}"
            )


    if st.button("Load PDF"):
        st.session_state.generation_needed = True
        if not st.session_state.pdf_path:
            st.error("Please upload a PDF first.")
            return
        
        try:
            
            with st.spinner(
                "Building retrieval chain..."
            ):

                chain = build_retrieval_chain(
                    st.session_state.pdf_path
                )

                st.session_state.retrieval_chain = chain
                st.session_state.loaded = True

            st.success("PDF loaded successfully, generating content...")

            if st.session_state.get("generation_needed", True):
                retrieval_chain = st.session_state.retrieval_chain
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        keywords_future = executor.submit(generate_keywords, retrieval_chain)
                        mcq_future = executor.submit(generate_mcq, retrieval_chain)
                        assertion_future = executor.submit(generate_assertion_reason, retrieval_chain)
                        subjective_future = executor.submit(generate_subjective_one_marks, retrieval_chain)
                        short_future = executor.submit(generate_short_questions, retrieval_chain)
                        long_future = executor.submit(generate_long_questions, retrieval_chain)
                        very_long_future = executor.submit(generate_very_long_questions, retrieval_chain)
                        case_future = executor.submit(generate_case_based_questions, retrieval_chain)

                        st.session_state.keywords = keywords_future.result()
                        st.session_state.mcq = mcq_future.result()
                        st.session_state.assertion_reason = assertion_future.result()
                        st.session_state.subjective_one_marks = subjective_future.result()
                        st.session_state.short_questions = short_future.result()
                        st.session_state.long_four_marks = long_future.result()
                        st.session_state.very_long_questions = very_long_future.result()
                        st.session_state.case_based_questions = case_future.result()

                        st.session_state.generation_needed = False
                except Exception as e:
                    st.session_state.generation_needed = True
                    raise


        except Exception as e:

            st.error(f"Error: {str(e)}")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.header("Keywords")
        if st.session_state.keywords:
            st.write(st.session_state.keywords)
        else:
            st.info("Keywords will appear here after loading the PDF.")

    with col2:
        st.header("Questions & Case Studies")
        if not st.session_state.loaded:
            st.info("Load a PDF to generate question lists and case studies.")
        else:
            if st.session_state.mcq:
                st.subheader("Multiple Choice Questions")
                st.write(st.session_state.mcq)

            if st.session_state.subjective_one_marks:
                st.subheader("1-Mark Subjective Questions")
                st.write(st.session_state.subjective_one_marks)

            if st.session_state.assertion_reason:
                st.subheader("Assertion-Reason Questions")
                st.write(st.session_state.assertion_reason)

            if st.session_state.short_questions:
                st.subheader("2-/3-Mark Short Questions")
                st.write(st.session_state.short_questions)

            if st.session_state.long_four_marks:
                st.subheader("4-/5-Mark Long Questions")
                st.write(st.session_state.long_four_marks)

            if st.session_state.very_long_questions:
                st.subheader("8-Mark Questions")
                st.write(st.session_state.very_long_questions)

            if st.session_state.case_based_questions:
                st.subheader("Case-Based Questions")
                st.write(st.session_state.case_based_questions)

    with col3:
        st.header("Ask from PDF")

        ask_disabled = (
            st.session_state.generation_needed
            or not st.session_state.loaded
        )

        query = st.text_area(
            "Ask a question from the PDF",
            height=200,
            placeholder=(
                "Example:\n"
                "- Explain Newton's Laws simply\n"
                "- Why does this happen?\n"
                "- Explain with analogy"
            ),
            key="ask_query",
            disabled=ask_disabled,
        )

        if ask_disabled:
            if st.session_state.generation_needed:
                st.info("Generation in progress. Please wait before asking a question.")
            else:
                st.info("Load a PDF to enable question asking.")

        if st.button("Ask Question", key="ask_button", disabled=ask_disabled):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    result = st.session_state["retrieval_chain"].invoke(
                        {"input": query}
                    )
                    st.session_state.answer = result.get(
                        "answer",
                        "No answer generated."
                    )

        if st.session_state.answer:
            st.markdown("## Answer")
            st.write(st.session_state.answer)


if __name__ == "__main__":

    main()