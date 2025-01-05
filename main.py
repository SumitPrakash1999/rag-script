import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.schema import Document
import logging
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import time
import re
import requests

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Initialize Gemini API
class ChatGemini:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        
    def generate(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API Error: {str(e)}")


# Enhanced Notion Scraper
class NotionLoader:
    def __init__(self, web_paths, headless=True):
        self.web_paths = web_paths if isinstance(web_paths, list) else [web_paths]
        self.headless = headless
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=chrome_options)
    
    def scroll_and_load_content(self, driver):
        """Scroll to load all dynamic content."""
        scroll_pause_time = 2
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def expand_collapsible_sections(self, driver):
        """Expand all collapsible sections on the page."""
        collapsible_elements = driver.find_elements(By.CLASS_NAME, "notion-toggle")
        for element in collapsible_elements:
            try:
                element.click()
                time.sleep(1)  # Wait for the section to expand
            except Exception as e:
                logging.warning(f"Failed to click collapsible section: {str(e)}")
    
    def extract_notion_content(self, driver, url):
        try:
            driver.get(url)
            
            # Wait for main content to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "notion-page-content"))
            )
            
            # Scroll to load all content
            self.scroll_and_load_content(driver)
    
            # Expand collapsible sections
            self.expand_collapsible_sections(driver)
            
            # Extract main content
            content = driver.find_element(By.CLASS_NAME, "notion-page-content").text
            
            # Extract page title
            try:
                title = driver.find_element(By.CLASS_NAME, "notion-page-block").text
            except:
                title = "Untitled Notion Page"
            
            return {
                "title": title,
                "content": content,
                "url": url
            }
            
        except Exception as e:
            logging.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def load(self):
        driver = self.setup_driver()
        documents = []
        
        try:
            for url in self.web_paths:
                self.logger.info(f"Processing URL: {url}")
                page_data = self.extract_notion_content(driver, url)
                
                if page_data:
                    doc = Document(
                        page_content=page_data["content"],
                        metadata={
                            "source": url,
                            "title": page_data["title"]
                        }
                    )
                    documents.append(doc)
                
        finally:
            driver.quit()
            
        return documents


# Global variables
documentation_links = [
    "https://www.notion.so/crustdata/Crustdata-Discovery-And-Enrichment-API-c66d5236e8ea40df8af114f6d447ab48",
    "https://www.notion.so/crustdata/Crustdata-Dataset-API-Detailed-Examples-b83bd0f1ec09452bb0c2cac811bba88c",
]

retriever = None


# Lifespan manager
@asynccontextmanager
async def lifespan(app):
    global retriever

    loader = NotionLoader(web_paths=documentation_links)
    docs = loader.load()

    for doc in docs:
        print(f"Title: {doc.metadata['title']}")
        print(f"Content Snippet: {doc.page_content[:500]}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # print(splits)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    yield


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
gemini_llm = ChatGemini(api_key=os.getenv("GEMINI_API_KEY"))

# Query schema
class QueryRequest(BaseModel):
    question: str




# Function to detect API requests in the generated answer
def detect_api_request(answer):
    """Detects and extracts API request information from the generated answer."""
    try:
        # Simple regex to find a curl command in the text
        match = re.search(r"curl '(.+?)'(.+?)--data '(.+?)'", answer, re.DOTALL)
        if match:
            endpoint = match.group(1)
            headers = dict(re.findall(r"--header '(.+?): (.+?)'", match.group(2)))
            payload = match.group(3)
            return endpoint, headers, payload
    except Exception as e:
        logging.error(f"Error detecting API request: {str(e)}")
    return None, None, None

def validate_api_request(api_request: str):
    try:
        # Extract and execute the API call from the generated response
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Token YOUR_API_KEY_HERE"
        }
        response = requests.post(
            url="https://api.crustdata.com/screener/person/search",
            headers=headers,
            json={
                "filters": [
                    {"filter_type": "CURRENT_TITLE", "type": "in", "value": ["engineer"]},
                    {"filter_type": "REGION", "type": "in", "value": ["California, United States"]}
                ],
                "page": 1
            }
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            logging.error(f"API Error: {response.text}")
            return False, response.text
    except Exception as e:
        logging.error(f"Validation Error: {str(e)}")
        return False, str(e)
        
@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        # Retrieve relevant chunks
        context_docs = retriever.get_relevant_documents(query.question)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Add examples to the context
        examples = """
        Example 1:
        Question: “How do I search for people given their current title, current company, and location?”
        Answer: You can use the api.crustdata.com/screener/person/search endpoint. Here is an example curl request to find “people with title engineer at OpenAI in San Francisco”:
        curl --location 'https://api.crustdata.com/screener/person/search' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Token $token \
        --data '{
            "filters": [
                {
                    "filter_type": "CURRENT_COMPANY",
                    "type": "in",
                    "value": [
                        "openai.com"
                    ]
                },
                {
                    "filter_type": "CURRENT_TITLE",
                    "type": "in",
                    "value": [
                        "engineer"
                    ]
                },
                {   
                    "filter_type": "REGION",
                    "type": "in",
                    "value": [
                        "San Francisco, California, United States"
                    ]
                }        
            ],
            "page": 1
        }'
        
        Example 2:
        Question: “Is there a standard you're using for the region values?”
        Answer: Yes, there is a specific list of regions listed here: https://crustdata-docs-region-json.s3.us-east-2.amazonaws.com/updated_regions.json.
        """
        
        # Construct the final prompt
        prompt = f"""
        Context:
        {context}
        
        Examples:
        {examples}
        
        Question:
        {query.question}
        
        Answer:
        """
        
        # Generate answer using Gemini API
        generated_answer = gemini_llm.generate(prompt)
        
        # Detect API request in the answer
        endpoint, headers, payload = detect_api_request(generated_answer)
        
        if endpoint and headers and payload:
            # Validate the API request
            valid_response, error_response = validate_api_request(endpoint, headers, payload)
            
            if valid_response:
                # If valid, include the API response in the answer
                return {
                    "answer": f"{generated_answer}\n\nValidated API Response:\n{valid_response}"
                }
            elif error_response:
                # Attempt to fix the API request based on error response
                suggestions = f"Error found in the API call: {error_response}. Please check the endpoint, headers, or payload."
                return {
                    "answer": f"{generated_answer}\n\n{suggestions}"
                }
        
        # Return the generated answer if no API request detected or no validation needed
        return {"answer": generated_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
