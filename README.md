# NLP
To run the application, first update the API keys in keys/API_keys.py. 

Then, set the admin password in your .streamlit folder, which can usually be found in your user folder. The file needs to be named as secrets.toml, and the file content should contain: ADMIN_PASSWORD = "your_password". For me, the file is located at: "C:\Users\robhe\.streamlit\secrets.toml"

Afterwards,  run the command:
streamlit run app.py in your command window. The application should now be launched.

Make sure all dependencies from the requirements.txt are installed.

The code is structured as follows:
App.py is the first streamlit layer, which creates the different pages for the non-admin and admin view. 
Home.py contains all streamlit code for displaying the chat window and sidebar, but also contains all logic to build the RAG-system and invoke it.
In the folder admin_pages, the other files contain the streamlit code to build that page, but also contain the code to log everything in databases.

The folder chains contains the files to create the RAG-chain, including both the documents and the images.

The folder LLM just initiates the LLM.

The folder datasets contains all databases, but also the files which create the classes for the telemetry database and questionnaire database. For the chat-history database, the standard database as seen in class is used.

The folder ingest contains all files to scrape the websites, including pictures, and turn this into a vector database for the documents, together with an image index file, and the downloaded image folder for the images.

The folder keys just contains the file with the keys for the LLMs.
