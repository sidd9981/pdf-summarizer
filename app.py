import streamlit as st
import sentencepiece
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

#MODEL_AND_TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

#file loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts=""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500,
        min_length = 50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


@st.cache_data
#display pdf function
def displayPDF(file):
    #opening file from filepath
    with open(file, "rb") as f:
          base64_pdf = base64.b64decode(f.read().decode('utf-8'))
    #Embedding pdf in html
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"><iframe>'
    #displaying file
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit code
st.set_page_config(layout='wide', page_title="Summarization App")
def main():
    st.title('Document Summarization App')
    uploaded_file = st.file_uploader("Upload your Pdf here", type=['pdf'])
    if uploaded_file is not None:
        if st.button("summarize"):
            col1, col2=st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded PDF File")
            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)

if __name__=='__main__':
    main()