import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from transformers import pipeline, BertTokenizer
# import fitz

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess_input(input_text):
    tokens = tokenizer.tokenize(input_text)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]


    return input_ids

def extract_text_from_pdf(pdf_docs, input_text):
    all_relevant_text = []  
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        chunk_size = 1000  # Set the desired chunk size
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        relevant_text = ""
        for chunk in chunks:
            chunk_relevant_text = answer_question(input_text, chunk)
            relevant_text += chunk_relevant_text

        # relevant_text = answer_question(input_text, text)
        all_relevant_text.append(relevant_text)
    return all_relevant_text

def answer_question(question, context):
    summarization_pipeline = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    input_text = f"question: {question} context: {context}"

    input_ids = preprocess_input(input_text)
    input_text = tokenizer.decode(input_ids)
    summarized_text = summarization_pipeline(input_text, max_length=1000, min_length=100, do_sample=True)[0]['summary_text']
    return summarized_text



def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("Lets chat :books:")
    user_question = st.text_input("Ask a question about your documents:")

    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if user_question:
        with st.spinner("Processing"):
            
            pdf_docs = st.session_state.pdf_docs
           
           
            st.session_state.conversation_history.append(('user', user_question))

           
            document_texts = extract_text_from_pdf(pdf_docs,user_question)
            
            summarized_text =answer_question(user_question, document_texts)

           
            st.session_state.conversation_history.append(('bot', summarized_text))

    with st.sidebar:
        st.subheader("Upload documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            st.session_state.pdf_docs = pdf_docs
            # for pdf in pdf_docs:
            #     pdf_reader = PdfReader(pdf)
            #     text=""
            #     for page in pdf_reader.pages:
            #         text += page.extract_text()

            # st.write("Extracted text: ",text)

    # Display conversation history
    for role, message in st.session_state.conversation_history:
        if role == 'user':
            st.write("You:", message)
        elif role == 'bot':
            st.write("Bot:", message)

if __name__ == '__main__':
    main()
