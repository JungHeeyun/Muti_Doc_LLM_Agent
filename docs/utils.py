import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os 

def build_ascii_tree(folder_structure, parent_path="", indent='', file_indices=[1]):
    ascii_tree = ""
    current_level_items = [item for item in folder_structure if os.path.dirname(item['orgHierarchy']) == parent_path]

    for i, item in enumerate(current_level_items):
        is_last = (i == len(current_level_items) - 1)
        connector = '└── ' if is_last else '├── '
        if item['type'] == 'Folder':
            ascii_tree += f"{indent}{connector}{os.path.basename(item['orgHierarchy'])}\n"
            ascii_tree += build_ascii_tree(folder_structure, item['orgHierarchy'], indent + ('    ' if is_last else '│   '), file_indices)
        elif item['type'] == 'File':
            current_index = file_indices[0]
            file_name = os.path.basename(item['orgHierarchy'])
            parent_folder_path = os.path.dirname(item['streamlit_path'])
            actual_file_path = os.path.join(parent_folder_path, file_name)
            ascii_tree += f"{indent}{connector}{file_name} [{item['Size (MB)']} MB] -- Index Number: {current_index}\n"
            update_file_index_map(current_index, actual_file_path, file_indices)
            if file_name.lower().endswith('.pdf'):
                build_faiss_index_from_pdf(actual_file_path, current_index)

    return ascii_tree

def update_file_index_map(current_index, actual_file_path, file_indices):
    if 'file_index_map' not in st.session_state:
        st.session_state['file_index_map'] = {}
    st.session_state['file_index_map'][current_index] = actual_file_path
    file_indices[0] += 1

def build_faiss_index_from_pdf(pdf_path, current_index):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

        if 'faiss_indices' not in st.session_state:
            st.session_state['faiss_indices'] = {}
        st.session_state['faiss_indices'][current_index] = faiss_index

    except Exception as e:
        st.error(f"Error processing {pdf_path}: {str(e)}")
        return None

    return faiss_index


def format_chat_history(messages):
    recent_messages = messages[-5:] 
    formatted_history = ""
    for message in recent_messages:
        if message['role'] == 'user':
            formatted_history += f"HumanMessage: {message['content']}\n"
        else:
            formatted_history += f"AIMessage: {message['content']}\n"
    return formatted_history