import requests
import pandas as pd
import sys
import argparse
import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()

API_KEY = os.getenv('SERPAPI_KEY')

profiles = []

def extract_year(text):
    match = re.search(r'(19\d{2}|200\d|201\d|202[0-3])', text)
    if match:
        return match.group()
    return None

def extract_first_author(text):
    match = re.match(r'([^,-]+)', text)
    if match:
        return match.group().strip()
    return None

# Define the function to fetch data from the SERP API
def fetch_scholar_profile(query=None, num_pages=10):
    all_papers = []
    paper_id = 0

    for page in range(num_pages):
        start = page * 10  # Adjust the start for each page
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": API_KEY,
            "start": start
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        papers = results.get("organic_results", [])
        for paper in papers:
            paper['doc_position'] = paper_id
            paper['query'] = query
            paper_info = paper['publication_info']
            authors = paper_info.get('authors', None)
            summary = paper_info.get('summary', None)
            paper['author_name'] = authors[0]['name'] if authors else extract_first_author(text=summary)
            paper['year'] = extract_year(text=summary) if summary else None
            paper_id += 1
        all_papers.extend(papers)

    return all_papers

# Define the function to convert the profile data into a CSV file
def convert_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data has been written to {filename}")

