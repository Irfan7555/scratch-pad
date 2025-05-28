import glob
import io
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from core import ConversationQA
from core_chat2sop import chat2sop
from streamlit_option_menu import option_menu
import streamlit as st
from dataclasses import dataclass
from typing import Literal
# from streamlit_chat import message
import os
import utils as ut
from pathlib import Path
import base64
from PIL import Image
import sqlite3
from decrypt import decrypt_message
from datetime import datetime as dt
import win32com

###
################### Testing APIM ##############
yaml_inputs = ut.load_yaml('settings_app.yml')
os.environ["OPENAI_API_KEY"] = yaml_inputs['openai_api_key']
lock_key = yaml_inputs['lock_key']
os.environ["OPENAI_API_TYPE"] = yaml_inputs['openai_api_type']
os.environ["OPENAI_API_VERSION"] = yaml_inputs['openai_api_version']
os.environ["OPENAI_API_BASE"] = yaml_inputs['openai_api_base']
os.environ["CHAT_MODEL"] = yaml_inputs['chat_model']
os.environ["CHAT_MODEL_DEPLOYMENT_NAME"] = yaml_inputs['chat_model_deployment_name']
os.environ["EMBEDDINGS_MODEL"] = yaml_inputs['embeddings_model']
###########################*****************####################################
os.environ["EMBEDDINGS_MODEL_DEPLOYMENT_NAME"] = yaml_inputs['embeddings_model_deployment_name']
# SQLITE
###################*#############################################################
# Create a connection to the SQLite database
#conn = sqlite3.connect('ntt_llm.db')
##################################################################################
##################################################################################

YAML Inputs
#########****************************************************************************#####
parent_path = Path.cwd()
im_1 = yaml_inputs['background']
im_2 = yaml_inputs['side_image']
gen_logo = yaml_inputs['gen_logo']
back_color = yaml_inputs['back_color']
hover_color = yaml_inputs['hover_color']
select_color = yaml_inputs['select_color']
icon_color = yaml_inputs['icon_color']
about_file = yaml_inputs['about_text']
dbs_path = os.path.join(os.getcwd(), yaml_inputs['vdb_path'])
model_path = os.path.join(os.getcwd(), yaml_inputs['tf_model_path'])
data_path = os.path.join(os.getcwd(), yaml_inputs['tf_matrix_path'])
know_db_path = os.path.join(os.getcwd(), yaml_inputs['know_db_path'])
##################################################################################

@st.cache_data
def get_img_as_base64(file):
    img = Image.open(file)
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64(im_1)
img2 = get_img_as_base64(im_2)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/png;base64,{img}");
background-size:auto;
}}
[data-testid="stHeader"] {{
background-image: url("data:image/png;base64,{img}");
background-size:auto;
}}
[data-testid="stSidebar"] {{
background-image: url("data:image/png;base64,{img2}");
background-size:auto;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
##################################################################################
st.markdown(page_bg_img, unsafe_allow_html=True)
logo = Image.open(gen_logo)
st.sidebar.image(logo)

with open(parent_path.joinpath(about_file), "r") as file:
    txt_1st = []
    for line in file:
        txt_1st.append(line)
###****************************************************************************###
# All CSS Fonts
##################################################################################
st.markdown(""" <style>.font {
font-size: 35px; text-align: 'center'; font-family: 'Cooper Black'; color: #FF9633;}
</style> """, unsafe_allow_html=True)
st.markdown(""" <style>.font2 {
font-size: 25px; font-family: 'Geneva'; color: #FF9633;}
</style> """, unsafe_allow_html=True)
st.markdown(""" <style>.font3 {
font-size:25px; "text-align": "center", font-family: 'Geneva'; color: #FF9633;}
</style> """, unsafe_allow_html=True)
st.markdown("""  <style>.font4 {
font-size:16px;
"text-align": "right", font-family: 'Geneva'; color: 'black';}
</style> """, unsafe_allow_html=True)
##################################################################################
####
#Misc Functions
# def send_request_subscription_key():
##################################################################################
# user_id = os.getlogin()
#
# outlook = win32com.client.Dispatch('outlook.application')
#
# mail = outlook.CreateItem(0)
#
# recipients = yaml_inputs['email_list']
#
# mail.To = ';'.join(recipients)
#
# mail.Subject = f"User: {user_id} subscription key request for Virtual Assist"
#
# mail.HTMLBody = f"User: {user_id} is requesting for new subscription key<br><br>&nbsp;<br>"
#
# mail.Send()
#
# return True

def get_path():
    vdb_path = st.text_input(label='Enter the path...')
    return vdb_path

def check_like():
    st.session_state.emoji_like = '+1'

def check_dislike():
    st.session_state.emoji_like = '-1'

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def get_text():
    input_text = st.text_input("You: ", "", key="input", max_chars=1000)
    #input_text = st.chat_input("You: ")
    print(input_text)
    return input_text

def generate_response(prompt):
    response, flag = qbot.retrieve_response(prompt)
    return response, flag

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""

    sources_list = list(source_urls)
    # sources_list.sort()
    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        if i == len(sources_list) - 1:
            sources_string += f"{i+1}. {source}"
        else:
            sources_string += f"{i+1}. {source} \n"
    return sources_string

@dataclass
class Message:
    """Class for keeping track of a chat message"""
    origin: Literal["human", "bot"]
    message: str
#########################################
# Headers
##################################################################################
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi! I am User']

if 'extras_generated' not in st.session_state:
    st.session_state['extras_generated'] = ['Hi! I am Virtual Assist. Let me know how can I assist you?']

if 'references' not in st.session_state:
    st.session_state['references'] = []

if 'emoji_like' not in st.session_state:
    st.session_state.emoji_like = "0"

if 'path' not in st.session_state:
    st.session_state['path'] = ''
#Sidebar
##################################################################################

with st.sidebar:
    choose = option_menu(menu_title="Main Menu", options=["About", "Chatbot", "Previous Responses", "Create Embed-DB", "SOP Creator", "Stop"],
                       icons=['house-fill', 'bi-chat-left-text', "bi-database", 'bi-bricks', 'file-text-fill', 'stop-circle-fill'],
                       menu_icon="app-indicator", default_index=0,
                       styles={
        "container": {"padding": "5!important", "background-color": "black"},
        "icon": {"color": icon_color, "font-size": "25px"},
        "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "hover-color": hover_color, "text-color": "white"},
        "nav-link-selected": {"background-color": select_color},
    }
    )
    if st.session_state['authenticated'] == False:
        st.header('Enter the Subscription Key')
        #req_key = st.button(label="Request Subscription Key")
        # if req_key:
        #
        #    send_request_subscription_key()

        with st.form(key='sub_key_decrypt', clear_on_submit=False):
            sub_key = st.text_input(label="Please enter your subscription Key")  # Collect user feedback
            submitted = st.form_submit_button("Submit")

            if submitted:
                corr_sub_key = sub_key.encode()
                corr_lock_key = lock_key.encode()
                user_id_rec, date_rec = decrypt_message(corr_lock_key, corr_sub_key)
                user_id_ori = os.getlogin()
                todays_date = dt.today()
                date_rec_corrected = dt.strptime(date_rec, "%Y-%m-%d")
                date_rec_new_format = date_rec_corrected.strftime("%d-%m-%Y")

                if (user_id_rec == user_id_ori) & (todays_date <= date_rec_corrected):
                    st.session_state['authenticated'] = True
                    st.write("The session has been authenticated")
                    st.write(f" Reminder: Your subscription ends on: {date_rec_new_format}")
                    time.sleep(10)
                    st.experimental_rerun()
                #else:
                    #st.warning("Invalid Subscription Key")

def on_click_callback():
    conn = sqlite3.connect('ntt_llm.db')
    c = conn.cursor()
    #Create a table with the specified columns
    c.execute('''CREATE TABLE IF NOT EXISTS new_response_table(
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            sources TEXT,
            like TEXT)  ''')
    user_input = st.session_state.user_input
    #faq_db = qbot.get_faq_data(know_db_path)
    ########*****************************#####New Addded Translation Section#######
    if lang_proc == "Non-English":
        lang_det, t_user_input = ut.detect_and_translate(user_input)
        print("Translated User Input")
        print(lang_det)
    else:
        t_user_input = user_input
    response, flag = generate_response(t_user_input)
    print(response, flag)
    ##################################
    if flag == 'from faq':
        formatted_response = "1) From FAQ"
    #####################
    if lang_proc == "Non-English":
        t_response = ut.translate_response(response, lang_det)
    else:
        t_response = response
    #####################
    st.session_state['generated'].append(t_response)
    st.session_state['extras_generated'].append(t_response)
    st.session_state['references'] = formatted_response
    st.session_state['past'].append(user_input)
    st.session_state.emoji_like = "0"
    like = st.session_state.emoji_like

    c.execute("INSERT INTO new_response_table (query, response, sources, like) VALUES (?, ?, ?, ?)",
              (t_user_input, response, formatted_response, like))
    conn.commit()
    st.success('Data inserted successfully!')
    conn.close()
    #except Exception as e:
        #conn.rollback()
        #st.error(f'Error inserting data: {e}')
        #raise e
    #finally:
        #conn.close()
    #return response

    #formatted_response = [
        #f"{sources_str}" for sources_str, source in zip(sources_strings, response["source_documents"])
    #]
    #return formatted_response
else:
    sources = set(
        [str(doc.metadata['source']).split('\\')[-1] + '  ' + 'Page#' + str(doc.metadata['page'] + 1) for doc in
         response['source_documents']])
    sources_str = create_sources_string(sources)
    formatted_response = (
        f"{sources_str}"
    )
    #####################
    if lang_proc == "Non-English":
        t_response = ut.translate_response(response.get("answer", ""), lang_det)
    else:
        t_response = response.get("answer", "")
    #####################
    st.session_state['generated'].append(t_response)
    st.session_state['extras_generated'].append(t_response)
    st.session_state['references'] = formatted_response
    st.session_state['past'].append(user_input)
    st.session_state.emoji_like = "0"
    like = st.session_state.emoji_like
    c.execute("INSERT INTO new_response_table (query, response, sources, like) VALUES (?, ?, ?, ?)",
              (t_user_input, response.get("answer", ""), formatted_response, like))
    conn.commit()
    st.success('Data inserted successfully!')
    conn.close()
    #except Exception as e:
     #   conn.rollback()
      #  st.error(f'Error inserting data: {e}')
       # raise e
    #finally:
     #   conn.close()

def custom_selectbox(options, colors):
    # Create a dictionary to map colors to their respective options
    color_map = {option: color for option, color in zip(options, colors)}

    # Create a selectbox with the options
    selected_option = st.selectbox("Select an option", options)

    #Get the selected option's color
    selected_option_color = color_map[selected_option]

    # Use HTML and CSS to display the selected option with the corresponding color
    colored_option = f'<span style="color: {selected_option_color};">{selected_option}</span>'
    st.markdown(colored_option, unsafe_allow_html=True)

load_css()
################################**
if (choose == 'About') & (st.session_state['authenticated']==True):
    st.markdown("<h1 style='text-align: center; color: black;  '>Virtual Assist</h1>",
                unsafe_allow_html=True)
    with open(parent_path.joinpath(about_file), "r") as file:
        chat2sop
        txt_1st = []
        for line in file:
            txt_1st.append(line)
        $a = f"{txt_1st[0]}"
        $b = f"{txt_1st[1]}"
        $c = f"{txt_1st[2]}"
        $d = f"{txt_1st[3]}"
        $e = f"{txt_1st[4]}"
        st.markdown(
            "<p style='text-align: justify; color: black;'><b><i>" + a + "</i></b></p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: justify; color: black; '><b>" + b + "</b></p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: justify; color: black; '><b><i>" + c + "</i></b></p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: justify; color: black; '><b>" + d + "</b></p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: justify; color: black;"><b><i>" + e + "<i></b></p>",
            unsafe_allow_html=True)

elif (choose == "SOP Creator") & (st.session_state['authenticated'] == True):
    txt_file = st.file_uploader(label="Upload text file", type='txt')
    #st.write(txt_file)
    if txt_file:
        query_content = txt_file.read().decode("utf-8")
        #with open(txt_file.name) as f:
        #
        #    content_txt = f.read()
        st.write(query_content)
        but_sop = st.button("Convert to SOP")
        if but_sop:
            st.button("Convert to SOP")
            response = chat2sop(query_content, token_limit=yaml_inputs['token_limit_chat2sop'])
            output_sop = response['choices'][0]['message']['content']
            st.markdown(output_sop)
            print(output_sop)
            doc = ut.markdown_to_docx(output_sop)
            doc_buffer = io.BytesIO()
            doc.save(doc_buffer)

            st.download_button(
                label="Download SOP as .docx",
                data=doc_buffer,
                file_name="sop_document.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
elif (choose == 'Chatbot') & (st.session_state['authenticated'] == True):
    st.title("Chatbot")
    # Initialize Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the full response
        full_response, flag = generate_response(prompt)
        if flag == 'from faq':
            response = full_response
        else:
            response = full_response.get("answer", "")
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)
elif (choose == 'Previous Responses') & (st.session_state['authenticated'] == True):
    st.title("Displaying data from the database")
    conn = sqlite3.connect('ntt_llm.db')
    c = conn.cursor()
    c.execute("SELECT * FROM new_response_table")
    data = c.fetchall()
    conn.close()
    for row in data:
        st.write(row)
elif (choose == 'Create Embed-DB') & (st.session_state['authenticated'] == True):
    st.header("Vector Database Setup")
    st.session_state['path'] = get_path()
    if st.session_state['path']:
        # Process the documents and create the database
        p = st.session_state['path']  # Use the stored path
        qbot = ConversationQA(p)
        st.write("Embeddings are created and saved to the specified directory.")
elif (choose == 'Stop'):
    st.session_state['authenticated'] = False
    st.experimental_rerun()
