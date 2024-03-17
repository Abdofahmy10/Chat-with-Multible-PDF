import streamlit as st
import os
from helper import get_pdf_text , get_text_chunks , get_vectorstore , user_input



def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with Multible PDF 🔥 ❤ ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Hello from the other side  🚀 ")
        st.image('assets\\book.jpeg' , width=300)

        st.title("Available PDFs 🔗 ")

        # List available PDFs based on their filenames
        # pdf_filenames = st.file_uploader("Upload Multiple PDFs", accept="application/pdf", multiple=True)
        pdf_filenames = [os.path.basename(pdf) for pdf in ["path/to/pdf1.pdf", "path/to/pdf2.pdf"]]  # Get filenames
        for filename in pdf_filenames:
            st.write(filename)  # Display filenames in sidebar

        if st.button("Start process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_filenames)  # User filenames
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()