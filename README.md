
# RAG-based Query Suggestion Chatbot
![alt text](image.png)

## Overview

The RAG-based Query Suggestion Chatbot is a project aimed at providing assistance in retrieving information from blog archives using natural language queries. It utilizes Retrieval-Augmented Generation (RAG) and Chain of Thought (COT) strategies, to generate informative responses based on user queries.

## System Architecture

The system architecture consists of the following components:

1. **Streamlit Web Interface**: This serves as the user interface for interacting with the chatbot. Users can input queries, and the bot provides responses based on the website's data.

2. **Backend Services**: These services handle the processing of user queries, including retrieving relevant blog posts, generating reasoning steps, and refining responses. The backend integrates with external APIs and databases to fetch and store data.

3. **Natural Language Processing (NLP) Models**: NLP models, such as the 'Gemini PRO' and SentenceTransformerEmbeddings, are used for generating responses and embedding text data, respectively. These models are integrated into the backend services.

4. **Database (ChromaDB)**: ChromaDB is used to store blog post data and their corresponding embeddings. This allows for efficient retrieval of relevant posts based on user queries.

## Codebases

The project consists of the following main codebases:

1. **Streamlit Interface**: Handles the frontend interface using Streamlit and integrates with backend services for processing user queries.

2. **Backend Services**: Includes modules for fetching WordPress data, generating embeddings, and processing user queries using RAG and COT approaches.

3. **Utilities (utils)**: Contains utility functions for data preprocessing, embedding generation, and database interactions.

4. **Configuration (config.yaml)**: Stores configuration data such as site URLs and database settings.

## Integration Methods

The project integrates various libraries and APIs for natural language processing, database interaction, and web development:

- **Streamlit**: Used for building the web interface and handling user interactions.
- **ChromaDB**: Provides database functionality for storing and retrieving blog post data and embeddings.
- **Langchain**: Used for streamlining the process using the modules it has.
- **YAML Configuration**: Configuration data is stored in a YAML file for easy management and customization.

## Usage Instructions

To run the files and execute the RAG-based Query Suggestion Chatbot project, follow these steps:

### Step 1: Setup Environment

Ensure that you have Python installed on your system. If not, download and install Python from the official website: [python.org](https://www.python.org/).

### Step 2: Clone the Repository

Clone the project repository to your local machine using Git:

```bash
git clone https://github.com/your/repository.git
```

### Step 3: Install Dependencies

Navigate to the project directory and install the required Python dependencies using pip:

```bash
cd addanki-tirumala-vishnu-vardhan/wasserstoff/AiTask
pip install -r requirements.txt
```

### Step 4: Update Configuration

Edit the `config.yaml` file to specify the necessary configuration settings, including the WordPress site URL, database collection name and the embedding model to use.

```yaml
# config.yaml
site_url: https://learn.wordpress.org
collection_name: post_collection
embedding_model: all-MiniLM-L6-v2
```

### Step 5: Ingest the website

Execute the ingest script to fetch the wordpress site posts and add to vector store:

```bash
python ingest.py
```
This command will take the website link from the config file and fetch the posts in the website and create a vector store and collection from it. The vector store is stored locally in `./posts_db`

### Step 6: Test the project

Execute the tests script to test whether the vector_store is retrieving the data, LLM generated the response and the streamlit application is working:

```bash
python tests.py
```
This command will test whether the vector_store is retrieving the data, LLM generated the response and the streamlit application is working. If all the three tests pass, then every component of the project is working .

### Step 7: Query 

Execute the query script to query about the website:

```bash
python query.py
```
This command will execute the query script. It takes two argument `query` and `chats`. If we do not the provide the arguments default values `query="What are WordPress tutorials"`, `chats=[("Hello", "Hey, How may i help you?")]` will be taken and the script is executed. 


### Step 8: Run the Chatbot Application

Execute the main script to launch the Streamlit web interface:

```bash
streamlit run app.py
```
This command will start the Streamlit server and provide a URL (typically `http://localhost:8501`) where you can access the web interface.

### Step 6: Interact with the Bot

Open your web browser and navigate to the provided URL (e.g., `http://localhost:8501`). You should see the interface.

Input queries in the provided text field and submit. The bot will generate responses and the Thought_steps based on the queries and display them in the interface.

### Step 7: Troubleshooting

If you encounter any issues while running the application, refer to the troubleshooting section of the README file for guidance on resolving common problems.


## Configuration Details

- **config.yaml**: The configuration file must contain the following 3 values `site_url`, `collection_name`, `embedding_model`
    ```yaml
    site_url: https://learn.wordpress.org  # website which you want to query
    collection_name: post_collection       # Name of the collection in order to save it
    embedding_model: all-MiniLM-L6-v2      # Name of the Sentence-transformers embedding model to be used while ingesting the documents
    ```
- **.env**: We are using Google's Gemini Model. So add the `GOOGLE_API_KEY` in the .env file


## Troubleshooting

If you encounter any issues while using the RAG-based Query Suggestion Chatbot, refer to the following troubleshooting steps:

1. **Check Configuration**: Ensure that the configuration settings in `config.yaml` and `.env` are accurate and properly formatted.

2. **Verify Dependencies**: Double-check that all required Python dependencies are installed and up-to-date.

3. **Database Connectivity, LLM Responses, Streamlit application**: Verify that the application can connect to the ChromaDB database, whether LLM api is working and the streamlit application is working by running the `tests.py`.

4. **Error Handling**: Review error messages and logs to identify the source of any issues and take appropriate action.


## Solutions Implemented

- **Modular Codebase**: Breaking down the project into smaller, modular components for easier development and maintenance.
- **Error Handling**: Implementing robust error handling and logging mechanisms to identify and resolve issues quickly.



## File Structures and Functions

### 1. `app.py`

This file contains the main Streamlit application code for the chatbot.

#### Functions:

- `update_vector_database(collection, post_id, embeddings, text)`: Updates the vector database with post ID, embeddings, and text.
- `fetch_existing_posts(collection)`: Fetches existing posts from the database.
- `update_embeddings_on_new_post(collection)`: Updates embeddings for new posts fetched from WordPress data.
- `rag_generate_response()`: Generates a prompt for generating reasoning steps for answering the user query.
- `develop_reasoning_steps(user_query, initial_prompt, previous_context)`: Develops reasoning steps based on the user query, initial prompt, and previous context.
- `refine_response_based_on_thought_steps(user_query, thought_steps)`: Refines the response based on thought steps.
- `process_query_with_chain_of_thought(user_query, previous_context)`: Processes the user query using the RAG + COT approach.
- `bot()`: Streamlit application to run the conversational AI bot.

### 2. `ingest.py`

This file contains the main function to fetch WordPress data, create a vector store, and add posts to it.

#### Functions:

- `main()`: Main function to fetch WordPress data, create a vector store, and add posts to it.

### 3. `query.py`

This file contains functions to process user queries using the RAG + COT approach.

#### Functions:

- `update_vector_database(collection, post_id, embeddings, text)`: Updates the vector database with post ID, embeddings, and text.
- `fetch_existing_posts(collection)`: Fetches existing posts from the database.
- `update_embeddings_on_new_post(collection)`: Updates embeddings for new posts fetched from WordPress data.
- `rag_generate_response()`: Generates a prompt for generating reasoning steps for answering the user query.
- `develop_reasoning_steps(user_query, initial_prompt, previous_context)`: Develops reasoning steps based on the user query, initial prompt, and previous context.
- `refine_response_based_on_thought_steps(user_query, thought_steps)`: Refines the response based on thought steps.
- `process_query_with_chain_of_thought(user_query, previous_context)`: Processes the user query using the RAG + COT approach.

### 4. `utils.py`

This file contains utility functions for fetching WordPress data, preprocessing text, generating embeddings, and creating a vector store.

#### Functions:

- `fetch_wordpress_data(site_url)`: Fetches data from a WordPress site using its REST API.
- `preprocess_text(text)`: Preprocesses text by removing HTML tags, decoding special characters, and removing extra whitespaces.
- `generate_embeddings(text)`: Generates sentence embeddings using a pre-trained embedding model.
- `extract_text(post)`: Extracts and preprocesses text content from a WordPress post.
- `create_vector_store_and_add_posts(wordpress_data)`: Creates a vector store in Chroma database and adds WordPress posts to it.

### 5. `tests.py`

This file contains unit tests for testing the functionality of the Chroma vector store, the language model, and the Streamlit UI.

#### Test Cases:

- `TestChromaVectorStore`: Tests whether the Chroma vector store retrieves data properly.
- `TestLLM`: Tests whether the language model generates responses.
- `TestStreamlitUI`: Tests the functionality of the Streamlit UI.
