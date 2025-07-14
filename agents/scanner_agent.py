import os
import json
import google.generativeai as genai
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.deals import Deal, ScrapedDeal, DealSelection
from agents.agent import Agent
from pydantic import BaseModel
from typing import List, Dict, Self, Optional
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time

class ScannerAgent(Agent):

    MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    {"deals": [
        {
            "product_description": "Your clearly expressed summary of the product in 4-5 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a paragpraph of text for each item you choose.",
            "price": 99.99,
            "url": "the url as provided"
        },
        ...
    ]}"""
    
    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        import google.generativeai as genai
        self.log("Scanner Agent is initializing")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.genai = genai
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("all-MiniLM-L6-v2", "cpu")
        self.log("Scanner Agent is ready")
    
    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        """
        Look up deals published on RSS feeds
        Return any new deals that are not already in the memory provided
        """
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped) -> str:
        """
        Create a user prompt for Gemini based on the scraped deals provided
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt
    
    @staticmethod
    def parse_price(price_str: str) -> float:
        """
        Parse price string like "$210" to float
        """
        if isinstance(price_str, str):
            # Remove $ and any commas, convert to float
            return float(price_str.replace('$', '').replace(',', ''))
        return float(price_str)
    
    @staticmethod
    def extract_json(text: str) -> str:
        """
        Extract JSON from text that might contain markdown code blocks
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\n', '', text)
        text = re.sub(r'```\n?', '', text)
        text = text.strip()
        
        # Try to find JSON content - check for both objects {} and arrays []
        object_start = text.find('{')
        array_start = text.find('[')
        
        # Determine which comes first (or if only one exists)
        if object_start != -1 and (array_start == -1 or object_start < array_start):
            # JSON object
            start = object_start
            end = text.rfind('}') + 1
        elif array_start != -1:
            # JSON array
            start = array_start
            end = text.rfind(']') + 1
        else:
            # No JSON found, return as is
            return text
        
        if start != -1 and end != 0:
            return text[start:end]
        return text
    
    def scan_gemini(self, memory: List[str] = []) -> Optional[DealSelection]:
        """
        Call Gemini to provide a high potential list of deals with good descriptions and prices
        :param memory: a list of URLs representing deals already raised
        :return: a selection of good deals, or None if there aren't any
        """
        try:
            # First fetch the deals
            scraped = self.fetch_deals(memory)
            if not scraped:
                self.log("No new deals found to process")
                return None
            
            # Create the user prompt with the scraped deals
            user_prompt = self.make_user_prompt(scraped)
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
            
            self.log("Scanner Agent is calling Gemini")
            model = genai.GenerativeModel(self.MODEL)
            response = model.generate_content(full_prompt)
            reply = response.text
            
            print("RAW Gemini reply:\n", repr(reply))  # Debug: show raw text
            clean_text = self.extract_json(reply)
            print("Cleaned JSON text:\n", clean_text)  # Debug: show stripped version
            parsed = json.loads(clean_text)
            print("Parsed JSON type:", type(parsed))  # Debug: show type
            
            # Determine deal list location
            deals_data = None
            if isinstance(parsed, dict):
                print("Parsed keys:", parsed.keys())  # Extra debug
                for key in ["selected_deals", "deals", "promising_deals"]:
                    deals_data = parsed.get(key)
                    if deals_data:
                        break
            elif isinstance(parsed, list):
                deals_data = parsed
            else:
                raise ValueError("Parsed JSON is not a list or dict")
            
            if not deals_data:
                raise ValueError("No deals found in parsed JSON")
            
            deals = [
                Deal(
                    title=deal.get("title", ""),  # Add default for title
                    product_description=deal["product_description"],
                    price=self.parse_price(deal["price"]),
                    url=deal.get("url")  # Use .get() since URL might not exist
                )
                for deal in deals_data
            ]
            
            # Filter out deals with price <= 0
            deals = [deal for deal in deals if deal.price > 0]
            self.log(f"Scanner Agent received {len(deals)} selected deals with price>0 from Gemini")
            
            return DealSelection(deals=deals)
            
        except json.JSONDecodeError as e:
            self.log(f"❌ JSON parsing error: {e}")
            print(f"❌ JSON parsing error: {e}")
        except Exception as e:
            self.log(f"❌ Error in scan_gemini: {e}")
            print(f"❌ Error: {e}")
        return None