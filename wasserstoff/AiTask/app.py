import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from utils import fetch_wordpress_data, extract_text, generate_embeddings
import chromadb, yaml
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
try:
    # Attempt to load configuration data from config.yaml file
    with open("./config.yaml", 'r') as file:
        config_data = yaml.safe_load(file)
except Exception as e:
    # Raise exception if config.yaml file is not found
    raise Exception(f"Not able to find the file ./config.yaml")

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# Initialize Chroma database client
client = chromadb.PersistentClient("./posts_db") 
collection_name = config_data['collection_name']
collection = client.get_collection(name=collection_name)

# Initialize embedding function for sentence transformer
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Langchain Chroma retriever
langchain_chroma = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    ).as_retriever()

# Initialize Chat Google Generative AI model
model = ChatGoogleGenerativeAI(model="gemini-pro")

def update_vector_database(collection, post_id, embeddings, text):
    """
    Update the vector database with post ID, embeddings, and text.

    Args:
    collection: The collection in the database.
    post_id (str): The ID of the post.
    embeddings (list): List of embeddings generated from the post text.
    text (str): The text of the post.
    """
    collection.upsert(ids=[str(post_id)], documents=[text], embeddings=[embeddings])

def fetch_existing_posts(collection):
    """
    Fetch existing posts from the database.

    Args:
    collection: The collection in the database.

    Returns:
    list: List of existing posts.
    """
    # Fetch existing posts from the database or any other storage
    existing_posts = collection.get()
    return existing_posts

def update_embeddings_on_new_post(collection):
    """
    Update embeddings for new posts fetched from WordPress data.

    Args:
    collection: The collection in the database.
    """

    # Fetch existing posts from the database or any other storage
    existing_posts = fetch_existing_posts(collection)  
    new_posts = fetch_wordpress_data(config_data['site_url'])

    # Compare old and new posts to find the difference
    existing_post_ids = set(str(post_id) for post_id in existing_posts['ids'])
    new_posts_to_update = [post for post in new_posts if str(post['id']) not in existing_post_ids]
    
    # Update embeddings for new posts
    for post in new_posts_to_update:
        # Extract text from post
        text = extract_text(post)  
        # Generate embeddings for the post text
        embeddings = generate_embeddings(text)
        # Update vector database with post ID and embeddings
        update_vector_database(collection,post['id'], embeddings, text)

def rag_generate_response():
    """
    Generate a prompt for generating reasoning steps for answering the user query.

    Returns:
    ChatPromptTemplate: The generated prompt template.
    """
    template = """You are tasked with designing a prompt for generating reasoning steps for answering to the user_query in a website. Write a Prompt to generate a series of intermediate thoughts or reasoning steps to answer the query. Avoid providing specific solutions or examples, allowing the LLM to explore different approaches independently. Give the output as a list of steps. eg: [1,2,3,...]

    Question: {user_query}
    Previous_context : {previous_context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt
    
def develop_reasoning_steps(user_query, initial_prompt, previous_context):
    """
    Develop reasoning steps based on the user query, initial prompt, and previous context.

    Args:
    user_query (str): The query from the user.
    initial_prompt (ChatPromptTemplate): The initial prompt template.
    previous_context (str): The previous context.

    Returns:
    list: List of thought steps.
    """
    chain = (
        RunnableParallel({"user_query": RunnablePassthrough(), "previous_context": RunnablePassthrough()})
        | initial_prompt
        | model
        | CommaSeparatedListOutputParser()
    )
    thought_steps = chain.invoke({"user_query" : user_query, "previous_context" : previous_context})
    thought_steps = thought_steps[0].split('\n')
    return thought_steps

def refine_response_based_on_thought_steps(user_query, thought_steps):
    """
    Refine the response based on thought steps.

    Args:
    user_query (str): The query from the user.
    thought_steps (list): List of thought steps.

    Returns:
    str: Final refined response.
    """
    all_retrieved_content = ""
    
    for thought_step in thought_steps:
        # print(langchain_chroma.invoke(thought_step))
        retrieved_content = langchain_chroma.invoke(thought_step)
        for i in retrieved_content:
            all_retrieved_content+=i.page_content
        all_retrieved_content+="\n"
    template = """You are a helpful assistant which answers the query from the context. If the context does not provide the answer simply reply I cannot answer this and give a suggestion to refer the website. DO NOT say that 'there is no information in the context' or 'the answer from the context is this.' phrases, instead give directly the solution or answer I cannot answer this and give a suggestion to refer the website or similar kind of text based on the context.:
    
    query : {user_query}

    context : {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    reason_chain = (
        RunnableParallel({'user_query': RunnablePassthrough(), 'context': RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    final_response = reason_chain.invoke({'user_query': user_query ,'context' : all_retrieved_content})
    return final_response
    
def process_query_with_chain_of_thought(user_query, previous_context):
    """
    Process the user query using the RAG + COT approach.

    Args:
    user_query (str): The query from the user.
    previous_context (list): List of previous chat contexts.

    Returns:
    tuple: A tuple containing thought steps and final refined response.
    """
    initial_response = rag_generate_response(user_query)  # initial response is the prompt
    thought_steps = develop_reasoning_steps(user_query, initial_response, previous_context)
    final_response = refine_response_based_on_thought_steps(user_query,thought_steps)
    return thought_steps, final_response

def bot():
    """
    Streamlit application to run the conversational AI bot.
    """
    def web_bot():
        global persist_directory
        if st.button("New Chat 🤖",key="Start New Chat"):
            st.session_state.clear()
            st.session_state.app = web_bot
            st.rerun()        

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []


        # Display chat messages from history and rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"]=="assistant":
                    for i in message["thought_steps"]:
                        st.markdown("- " + i)
            
        # Respond to user input after receiving
        if user_query:= st.chat_input("What's up?"):
            # Display user messages in chat message container
            with st.chat_message("User"):
                st.markdown(user_query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

            thought_steps, final_response = process_query_with_chain_of_thought(user_query, st.session_state.messages)
        

            # Display assistant response in hcat message container
            with st.chat_message("assistant"):
                for i in thought_steps:
                    st.markdown("- " + i)
                st.markdown(final_response)

            st.session_state.messages.append({"role" : "assistant", "content": final_response, "thought_steps": thought_steps})

    if 'app' not in st.session_state:
        st.session_state.app = web_bot

    st.session_state.app()

bot()