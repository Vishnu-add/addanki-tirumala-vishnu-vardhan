from utils import fetch_wordpress_data,create_vector_store_and_add_posts
import yaml

def main():
    """
    Main function to fetch WordPress data, create vector store, and add posts to it.

    This function reads configuration data from a YAML file, fetches WordPress data using the specified site URL,
    and creates a vector store in the database with the fetched posts.

    Raises:
    Exception: If the config.yaml file is not found or if there are any other errors during execution.
    """
    try:
        # Attempt to load configuration data from config.yaml file
        with open("./config.yaml", 'r') as file:
            config_data = yaml.safe_load(file)
            print(config_data) # Printing configuration data for debugging purposes
    except Exception as e:
        # Raise exception if config.yaml file is not found
        raise Exception(f"Not able to find the file ./config.yaml")
    
    # Fetch WordPress data using the site URL specified in the configuration
    wordpress_data = fetch_wordpress_data(config_data['site_url'])

    # Create vector store in the database and add WordPress posts to it
    client, collection = create_vector_store_and_add_posts(wordpress_data)

if __name__ == "__main__":
    main()
