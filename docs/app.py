import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import init_app
from streamlit_option_menu import option_menu  
import pandas as pd
import os
import zipfile
import tempfile
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from utils import build_ascii_tree, format_chat_history

def file_explorer_page():
    st.subheader("CMM NLP")

    if 'OPENAI_API_KEY' not in os.environ or os.environ['OPENAI_API_KEY'] == "":
        st.warning("Please enter your OpenAI API Key in the sidebar before uploading files.", icon="⚠️")
        return
    
    st.warning("Please upload a ZIP file containing only PDF and CSV files. You must complete the embedding process before proceeding to the chatbot page.", icon="⚠️")
    file_source = st.radio("Choose the file source:", ('Upload my own file', 'Use the default demo file'))

    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()

    uploaded_file = None
    folder_structure = []
    if file_source == 'Upload my own file':
        uploaded_file = st.file_uploader("Upload a ZIP file", type="zip")
    elif file_source == 'Use the default demo file':
        default_file_path = os.path.join(os.path.dirname(__file__), 'docs', 'demo.zip')
        try:
            uploaded_file = open(default_file_path, 'rb')
        except FileNotFoundError:
            st.error("The default demo file is not found. Please upload your own file.")

    with st.spinner('Embedding the files...'):
        if uploaded_file is not None:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(st.session_state['temp_dir'])

            for root, dirs, files in os.walk(st.session_state['temp_dir']):
                if '__MACOSX' in root.split(os.sep):
                    continue

                relative_path = os.path.relpath(root, st.session_state['temp_dir'])
                if relative_path == ".":
                    relative_path = ""
                else:
                    folder_structure.append({"orgHierarchy": relative_path, "type": "Folder", "Size (MB)": "", "streamlit_path": root})

                for file in files:
                    if not file.startswith('._') and file != '.DS_Store':
                        file_path = os.path.join(relative_path, file) if relative_path else file
                        file_size = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)  # MB로 크기 변환
                        folder_structure.append({"orgHierarchy": file_path, "type": "File", "Size (MB)": f"{file_size:.2f}", "streamlit_path": os.path.join(root, file)})

            df = pd.DataFrame(folder_structure)
            ascii_tree = build_ascii_tree(folder_structure)  
            st.session_state['df'] = df.to_dict('records')
            st.session_state['ascii_tree'] = ascii_tree

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_grid_options(treeData=True, defaultColDef={"flex": 1})
            gridOptions = gb.build()

            gridOptions["autoGroupColumnDef"] = {"headerName": "Files and Folders", "minWidth": 300, "cellRendererParams": {"suppressCount": True}}
            gridOptions["columnDefs"] = [{"field": "orgHierarchy", "hide": True}, {"field": "type", "headerName": "Type"}, {"field": "Size (MB)", "headerName": "Size (MB)"}]
            gridOptions["getDataPath"] = JsCode("function(data) { return data.orgHierarchy.split('/'); }").js_code
            gridOptions["groupDefaultExpanded"] = -1 

            AgGrid(df, gridOptions=gridOptions, height=600, allow_unsafe_jscode=True, theme='material', update_mode='MODEL_CHANGED')


def chatbot_page():
    st.subheader("CMM AI Research Assistant")

    with st.chat_message("assistant"):
                        st.markdown("""Hello! I am a multi-document analysis chatbot developed by the CMM NLP team. Here is a description of what I can do:
* Document keyword search (explains how specific keywords appear on various pages)
* Document page search (it also answers specific diagrams or images on the page)
* csv query
* Parallel processing (the above functions can be processed in parallel)
* Memory (Because it remembers up to 5 past conversation records, it is possible to ask abstract questions about concepts based on the contents of past conversations.)

*Note: In the case of the csv tool, it may take a long time because it is a separate sub agent.
                                    
If you have any additional questions, please ask me and use it well.
""")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button('End Chat Session'):
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        is_user = message["role"] == "user"
        container = st.container()
        with container:
            if is_user:
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"]) 

    if prompt := st.chat_input("Send a message"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner('Wait for it...'):
                chat_history = format_chat_history(st.session_state['messages'])
                app = init_app(st.session_state['ascii_tree'], st.session_state['file_index_map'], chat_history, st.session_state['faiss_indices'])
                inputs = {"messages": [HumanMessage(content=prompt)]}
                for output in app.with_config({"run_name": "LLM with Tools"}).stream(inputs):
                    for key, value in output.items():    
                        with st.expander("Node Type: "+key):
                            st.write(value)
                        for message in value['messages']:
                            if message.content:  
                                if isinstance(message, AIMessage):
                                    st.session_state.messages.append({"role": "assistant", "content": message.content})
                                    st.markdown(message.content)
                      
def main():
    st.set_page_config(page_title="CMM", layout="wide")

    with st.sidebar:
        selected = option_menu("Main Menu", ["CMM", "Chatbot"],
            icons=['house', 'chat-left-text'], menu_icon="cast", default_index=0)
        
        if 'OPENAI_API_KEY' not in os.environ:
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                st.success("API Key stored in environment!")
        else:
            st.write("API Key is already stored in environment.")
            if st.button("Clear API Key"):
                os.environ.pop('OPENAI_API_KEY', None)
                st.warning("API Key cleared from environment!")

    if selected == "CMM":
        file_explorer_page()
    elif selected == "Chatbot":
        chatbot_page()

if __name__ == "__main__":
    main()
