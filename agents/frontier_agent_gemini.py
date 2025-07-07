# imports
import os
import re
import math
import json
from typing import List, Dict
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent

class FrontierAgentGemini(Agent):

    name = "Frontier Agent Gemini"
    color = Agent.BLUE

    MODEL = "gemini-2.5-flash"

    def __init__(self, collection):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = collection
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message
    
    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]
    
    def find_similars(self, description: str):
 
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices
    
    def get_price(self, s) -> float:

        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0
    
    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)
        retries = 8
        done = False
        reply = None
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        while not done and retries > 0:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                # Convert OpenAI-style messages to a single prompt
                prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.messages_for(description, documents, prices)]) # make the prompt the gemini way (different from openai)
                response = model.generate_content(prompt)
                reply = response.text
                done = True
            except Exception as e:
                print(f"Error: {e}")
                retries -= 1

        if reply is None:
            return "ERROR: Gemini failed after retries"
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
    