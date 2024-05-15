import unittest
import chromadb 
import subprocess
import time
import yaml
import psutil
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


class TestChromaVectorStore(unittest.TestCase):
    def setUp(self):
        try:
            # Attempt to load configuration data from config.yaml file
            with open("./config.yaml", 'r') as file:
                config_data = yaml.safe_load(file)
        except Exception as e:
            # Raise exception if config.yaml file is not found
            raise Exception(f"Not able to find the file ./config.yaml")

        self.client = chromadb.PersistentClient("./posts_db") 
        collection_name = config_data['collection_name']
        
        self.collection = self.client.get_collection(name=collection_name)
        # Initialize embedding function for sentence transformer
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        self.langchain_chroma = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=embedding_function,
            ).as_retriever(n_results=1)
    
    def test_retrieve_vector_store(self):
        # Testing whether the Chroma vector store retrieves data
        data = self.langchain_chroma.invoke("Wordpress")
        self.assertIsNotNone(data)
        print("Vector Store is Working Properly!")
        
class TestLLM(unittest.TestCase):
    def setUp(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-pro")
    
    def test_response_from_model(self):
        # Testing whether LLM returns responses
        response = self.model.invoke("Hello!")
        self.assertIsNotNone(response)
        print("LLM is generating responses!")

class TestStreamlitUI(unittest.TestCase):
    def test_streamlit_ui(self):
        process = subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        time.sleep(60)
        if process.poll() is None:
            print("Streamlit app is running. Stopping the app...")

            process_id = process.pid
            parent = psutil.Process(process_id)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            process.wait()
            print("Streamlit is working properly.")
        else:
            print("Streamlit app has already terminated.")

if __name__ == '__main__':
    unittest.main()
