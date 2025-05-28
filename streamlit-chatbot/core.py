import os
import time
from typing import Any
from pathlib import Path
import pandas as pd
from langchain.chains import RetrievalQA
# from singleton_decorator import singleton
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils import load_doc_from_dir
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import numpy as np
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from scipy.sparse import load_npz


@singleton
class ConversationQA:
    def __init__(self, search_doc_limit: int = 4):
        self.search_doc_limit = search_doc_limit
        self.chat_history = []

    def chat_bot(self, vecdb_path, model_path, data_path, know_db_path, threshold: float = 0.9):
        self.threshold = threshold
        self.tfidf_model = joblib.load(model_path)
        self.tfidf_mat = load_npz(data_path)
        self.faq = self.get_faq_data(know_db_path)
        self.vecdb_path = vecdb_path
        embeddings = OpenAIEmbeddings(
            deployment=os.environ["EMBEDDINGS_MODEL_DEPLOYMENT_NAME"],
            chunk_size=1
        )
        self.vectordb = FAISS.load_local(self.vecdb_path, embeddings=embeddings)

        system_msg_template = SystemMessagePromptTemplate.from_template(r"""
You are an AI assistant. If you are not clear and do not get a direct answer in the information then respond by saying:
"Hmm, I am not able to get this in SOP. Can you please check the SOP or rephrase your query."
If the query contains hi,Hi,Bye,bye,Hello,hello then ignore the chat history and just reply with the following fixed reply
"I am happy to assist you".
CONTEXT: {context}
Please restrict the answer only from the CONTEXT.
""")

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{question}")

        prompt_template = ChatPromptTemplate.from_messages(
            [system_msg_template, human_msg_template])

        # PROMPT = PromptTemplate(
        #    template=system_msg_template, input_variables=["context","question"]
        # )

        # self.chain_type_kwargs = {"prompt": prompt_template}

        print(os.environ)

        model = AzureChatOpenAI(
            openai_api_type=os.environ["OPENAI_API_TYPE"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_name=os.environ["CHAT_MODEL"],
            deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
        )
        self.qa = ConversationalRetrievalChain.from_llm(model,
                                                        # ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                                                        self.vectordb.as_retriever(
                                                            search_kwargs={"k": self.search_doc_limit}),
                                                        return_source_documents=True,
                                                        verbose=False,
                                                        combine_docs_chain_kwargs={"prompt": prompt_template}
                                                        )

    def create_embeddings(self, src_path):
        src_split = src_path.split('\\')
        vec_db_name = src_split[-1] + '_' + src_split[-1] + '_vecdb'
        chunked_documents = load_doc_from_dir(src_path)

        embeddings = OpenAIEmbeddings(
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_type=os.environ["OPENAI_API_TYPE"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            deployment=os.environ["EMBEDDINGS_MODEL_DEPLOYMENT_NAME"],
        )

        # Batch mechanism to create vector databases with a sleep interval
        batch_size = 15  # You can adjust the batch size as needed
        total_chunks = len(chunked_documents)
        print(total_chunks)
        chunks_processed = 0
        faiss_dbs = []

        batch = chunked_documents[chunks_processed: chunks_processed + batch_size]
        final_db = FAISS.from_documents(batch, embeddings)
        chunks_processed += batch_size
        time.sleep(5)

        while chunks_processed < total_chunks:
            # Take a batch of chunks
            batch = chunked_documents[chunks_processed: chunks_processed + batch_size]

            # Create vector database for the current batch
            db_batch = FAISS.from_documents(batch, embeddings)
            final_db.merge_from(db_batch)

            # Append the current batch vector database to the list

            # Update the number of processed chunks
            chunks_processed += batch_size
            print(chunks_processed)
            # Add a sleep interval between batches (adjust as needed)
            time.sleep(5)

        # embeddings = OpenAIEmbeddings(
        #    deployment=os.environ["EMBEDDINGS_MODEL_DEPLOYMENT_NAME"],
        #    chunk_size=1)
        # vectordb = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())

        db_path = "vector_databases" + '/' + src_split[-2] + '/' + src_split[-1] + '/' + vec_db_name
        return final_db, db_path, chunked_documents

    def create_knowledge_base(self, faq_path):
        faq = self.get_faq_data(faq_path)
        corpus = faq['FAQ'].to_list()
        tf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tf.fit_transform(corpus)
        return tf, tfidf_matrix

    def retrieve_from_knowledge_base(self, query: str):
        query_mat = self.tfidf_model.transform([query])
        pw_sim = cosine_similarity(query_mat, self.tfidf_mat)
        thresh = pw_sim.max()
        ind = pw_sim.argmax()
        ret_ans = self.faq.iloc[ind]['Answer']
