import streamlit as st

# Function to handle file upload
def read_and_save_file():
    uploaded_files = st.session_state["file_uploader"]
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # You can save the file locally or read its content
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved file: {uploaded_file.name}")

# Streamlit app header and file uploader
st.header("ChatBot (PDF Version)")

st.subheader("Upload a document")
st.file_uploader(
    "Upload document",
    type=["pdf"],
    key="file_uploader",
    on_change=read_and_save_file,
    label_visibility="collapsed",
    accept_multiple_files=True,
)
