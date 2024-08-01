# Chat with PDF using Free LLM API and Streamlit

![ragapp-flowspotdraft](https://github.com/user-attachments/assets/1ec32f93-f8e5-4b50-8b73-6485fd6536d1)
![Demo, proof](https://github.com/user-attachments/assets/053eb907-6581-4acb-b0ec-32f9094df760)

This project allows users to upload PDF files, process them, and chat with the content using a free LLM API. The application is built using Streamlit and various Python libraries for PDF processing, embeddings, and conversational AI.


## Installation

To run this project, you'll need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/shreeharib/RAG-chatbot.git
   cd RAG-chatbot
   ```
  
2.	Install the required dependencies:
 ```bash
   pip install -r requirements.txt
   ```
3.	Set up environment variables:
	•	Create a .env file in the root directory of your project.
	•	Add your Google Generative AI API key:

   ```bash
 GOOGLE_API_KEY=your_google_api_key
   ```
# To start the application, run the following command:

   ```bash
streamlit run Finaldoc.py
   ```

# Upload PDF Files:
•Use the sidebar to upload one or more PDF files.
•Click the “Submit & Process” button to process the files.

# Ask Questions:
•Use the chat input to ask questions about the content of the uploaded PDF files.
•The application will process your question and return a detailed answer based on the content of the PDFs and the AI model.

## Demo
https://youtu.be/-3Zgm52oONk
