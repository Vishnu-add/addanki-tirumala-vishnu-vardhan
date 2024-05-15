import requests
import re
from html import unescape
from sentence_transformers import SentenceTransformer
import chromadb
import yaml


try:
    # Attempt to load configuration data from config.yaml file
    with open("./config.yaml", 'r') as file:
        config_data = yaml.safe_load(file)
except Exception as e:
    # Raise exception if config.yaml file is not found
    raise Exception(f"Not able to find the file ./config.yaml")


# function to fetch data from WordPress site
def fetch_wordpress_data(site_url):
    """
    Fetches data from a WordPress site using its REST API.

    Args:
    site_url (str): The URL of the WordPress site.

    Returns:
    dict: JSON data retrieved from the WordPress site.
    """
    api_url = f"{site_url}/wp-json/wp/v2/posts"
    try:
        # Send GET request to WordPress API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise exception for unsuccessful responses

        # Extract and return JSON data from response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during request
        print("Error fetching WordPress data:", e)
        return None

def preprocess_text(text):
    """
    Preprocesses text by removing HTML tags, decoding special characters, and removing extra whitespaces.

    Args:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    # Remove HTML tags
    clean_text = re.sub('<.*?>', '', text)    
    # Decode special characters
    clean_text = unescape(clean_text)
    # Removing extra newline characters
    clean_text = re.sub('\n+', '\n', clean_text)    
    # Remove extra whitespaces and newline characters
    clean_text = clean_text.strip()
    
    return clean_text

def generate_embeddings(text):
    """
    Generates sentence embeddings using a pre-trained embedding model.

    Args:
    text (str): The input text.

    Returns:
    list: List of sentence embeddings.
    """
    # Load pre-trained embedding model
    model = SentenceTransformer(config_data['embedding_model'])

    # Generate embeddings for input text
    embeddings = model.encode(text)
    return embeddings.tolist()

def extract_text(post):
    """
    Extracts and preprocesses text content from a WordPress post.

    Args:
    post (dict): The WordPress post data.

    Returns:
    str: The preprocessed text content of the post.
    """
    return preprocess_text(post['content']['rendered'])

def create_vector_store_and_add_posts(wordpress_data):
    """
    Creates a vector store in Chroma database and adds WordPress posts to it.

    Args:
    wordpress_data (list): List of WordPress post data.

    Returns:
    tuple: A tuple containing the Chroma client and collection objects.
    """
    client = chromadb.PersistentClient("./posts_db") 
    collection = client.get_or_create_collection(name = config_data['collection_name'], metadata={"hnsw:space": "cosine"})
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    for post in wordpress_data:
        ids.append(str(post['id']))
        cleaned_content = extract_text(post)
        embeddings.append(generate_embeddings(cleaned_content))
        documents.append(cleaned_content)
        metadata = {}
        metadata['title'] = post['title']['rendered']
        metadata['date'] = post['date']
        metadata['modified'] = post['modified']
        metadatas.append(metadata)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return client,collection


