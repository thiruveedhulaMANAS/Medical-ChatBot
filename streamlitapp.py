import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from streamlit_chat import message

DB_FAISS_PATH = 'faiss_index'

custom_prompt_template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question by including treatment,symptoms,description,etc. 
add note at the end of the response : If you telling any medicine name then at the end add consult the doctor before taking any medication or treatment it in bold letters.
If you don't know the answer, just say that you don't know. 
Use ten sentences maximum and keep the answer concise.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 6}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    llm = GoogleGenerativeAI(google_api_key="AIzaSyAeDbOpmz4MAhBVFx3UELiYj0MRz5fK2ow", model='models/gemini-1.5-pro', temperature=0.01)
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Streamlit interface setup
st.title("HealthCare Chatbot")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query):
    result = final_result(query)
    answer = result['result']
    st.session_state['history'].append((query, answer))
    return answer

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about Disease", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Display chat history
display_chat_history()
