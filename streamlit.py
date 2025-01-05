import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.llms.ollama import Ollama
from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
# ... other necessary imports ...

# 1. Setup logic for loading documents, creating vector DB, retrievers, etc.
def init_resources():
    folder_path = "..."
    loader = PyPDFDirectoryLoader(folder_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=FastEmbedEmbeddings(),
        persist_directory='chroma_db2'
    )

    # Create multiquery retriever
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant..."""
    )
    llm = Ollama(model="llama3.2", base_url="http://localhost:11434")
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(search_kwargs={'k': 10}),
        llm,
        prompt=QUERY_PROMPT
    )

    # Ensemble retriever
    retriever2 = vector_db.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(documents=chunks)
    main_retriever = EnsembleRetriever(
        retrievers=[retriever2, keyword_retriever],
        weights=[0.4, 0.6]
    )

    # HuggingFace Pipeline QA
    model_id2 = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_id2)
    model2 = AutoModelForQuestionAnswering.from_pretrained(model_id2)
    pipe2 = pipeline("question-answering", model=model2, tokenizer=tokenizer)
    llm3 = HuggingFacePipeline(pipeline=pipe2)

    template = """
    You are an expert assistant... 
    {context}

    ---
    Answer: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": main_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm3
        | StrOutputParser()
    )

    return chain

# 2. Define the Streamlit interface
def main():
    st.title("My RAG Chatbot")

    # Initialize chain (cached, so it loads once)
    chain = init_resources()

    user_question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if user_question.strip():
            answer = chain.invoke(user_question)
            st.write(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
