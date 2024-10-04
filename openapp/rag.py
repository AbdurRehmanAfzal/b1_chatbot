import os
import dotenv
import getpass
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MARKET_TOPICS = {
    'transaction_stats': ['sales volume', 'price trends', 'market activity'],
    'home_prices': ['average prices', 'price per sq ft', 'luxury segment pricing'],
    'market_insights': ['investment potential', 'market growth', 'future predictions'],
    'navigation': ['locations', 'communities', 'property types']
}
# conver all above function to a Class
class RAG_CLS():
    def __init__(self,pth):
        # self.llm = ChatOpenAI(model="gpt-4o-2024-08-06")
        self.pth = os.path.join(os.path.dirname(__file__), pth)
        self.llm = ChatOpenAI(model="gpt-4")
        self.documents = self.load_data()
        self.vectorstore = self.genrate_embedings()
        self.rag_chain = self.get_chain()
    
    def load_data(self):
    # Load and split the document
        with open(self.pth, "r") as file:
            text = file.read().split("##chunk##")
        documents = [Document(page_content=chunk) for chunk in text]
        return documents

    def genrate_embedings(self):
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
        chunks = text_splitter.split_documents(self.documents)

        # # Create embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        return vectorstore

    def get_chain(self):
        retriever = self.vectorstore.as_retriever()
        
        # Create a custom B1 Properties-specific prompt
        b1_prompt = """You are an AI assistant for B1 Properties, Dubai's premier luxury real estate brokerage. 
        Founded by Babak Jafari, we specialize in ultra-luxury properties in prime locations like Palm Jumeirah, 
        Jumeirah Bay, and Bluewaters Island.

        Use the following pieces of context to answer the question. If you don't know the answer, just say that 
        you don't know, don't try to make up an answer. Always maintain a tone that reflects our luxury brand 
        and expertise in the Dubai real estate market.

        Context: {context}
        
        Question: {question}
        
        Answer: Let me assist you with that."""

        # Create the RAG chain with the custom prompt
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | PromptTemplate.from_template(b1_prompt)
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_answer(self,query):
        return self.rag_chain.invoke(query)
    
    def get_answer(self, query):
        # Classify the query type
        query_type = self.classify_query(query)
        
        # Get the base response
        base_response = self.rag_chain.invoke(query)
        
        # Enhance response based on query type
        enhanced_response = self.enhance_response(base_response, query_type)
        
        return enhanced_response
    
    def classify_query(self, query):
        classification_prompt = f"""Classify this real estate query into one of these categories:
        {', '.join(MARKET_TOPICS.keys())}

        Query: {query}
        
        Category:"""
        
        response = self.llm.invoke(classification_prompt.format(query=query))
        return response
    
    def enhance_response(self, base_response, query_type):
        enhancements = {
            "Property Information": "Let me also mention that B1 Properties specializes in ultra-luxury properties in prime locations.",
            "Market Insights": "As Dubai's premier luxury brokerage, B1 Properties has unique insights into market trends.",
            "B1 Properties Services": "Our founder, Babak Jafari, ensures that every client receives exceptional service.",
            "Transaction History": "B1 Properties has achieved record-breaking sales, including a recent AED 128 million villa.",
        }

        # Since AIMessage is not defined, assume query_type is a string (or ensure it's a string)
        if not isinstance(query_type, str):  
            query_type = str(query_type)  # Convert query_type to a string, if not already

        if query_type in enhancements:
            return f"{base_response}\n\n{enhancements[query_type]}"
        
        return base_response

