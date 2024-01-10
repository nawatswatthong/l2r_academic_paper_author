import requests
import pandas as pd
import sys
import argparse
import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
import time

# Load .env file
load_dotenv()

BASE_URL = "https://serpapi.com/v1/scholar_profiles/"
API_KEY = os.getenv('SERPAPI_KEY')

# Define the function to fetch data from the SERP API
def fetch_scholar_profile(query=None, num_pages=10):
    all_profiles = []
    profile_id = 0
    next_page_token = None
    next_url = None

    for page in range(num_pages):
        print("PAGE :%s"  %page)
        params = {
                "engine": "google_scholar_profiles",
                "mauthors": query,
                "api_key": API_KEY,
            }

        if next_page_token:
            print(next_page_token)
            params["next_page_token"] = next_page_token
            search = requests.get(next_url, params)
            results = search.json()
        
        else:
            search = requests.get("https://serpapi.com/search.json?", params=params)
            results = search.json()
        
        if "profiles" in results and results["profiles"]:
            profiles = results.get("profiles", [])
            for profile in profiles:
                profile['id'] = profile_id
                profile['query'] = query
                profile_id += 1
            all_profiles.extend(profiles)
        else:
            break

        if "pagination" in results and "next_page_token" in results["pagination"]:
            next_page_token = results['pagination']['next_page_token']
            next_url = results['pagination']['next']
        else:
            break

    return all_profiles

# Define the function to convert the profile data into a CSV file
def convert_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data has been written to {filename}")
