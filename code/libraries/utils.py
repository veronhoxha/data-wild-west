############ UTILS ############

### IMPORTS ###

# BASICS
from datetime import datetime
import regex as re
import json
from collections import Counter

# DATA MANIPULATION
import pandas as pd
import numpy as np

# SCRAPPING
import requests
from bs4 import BeautifulSoup # xml parsing
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# NLP
import nltk
from emoji import UNICODE_EMOJI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu
from textblob import TextBlob, Word
import pkg_resources
import symspellpy
from symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# OTHERS
from tqdm import tqdm
from geopy.distance import geodesic
from IPython.display import display
from IPython.display import clear_output
import seaborn as sns

# MAPS
import folium

# GRAPH
import plotly.graph_objects as go

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


###################################################################################################

def day_schedule_periods(weekday_text):
    '''
    Converts and formats the opening and closing times of each day of the week from a structured input.
    
    Parameters:
        - weekday_text: A list of dictionaries where each dictionary contains the open and close times for a specific day of the week, with each day represented as an integer.

    Returns:
        - dict: A dictionary where each key is the name of a day of the week and the corresponding value is a string representing the formatted open and close times for that day, in 'HH:MM AM/PM' format.
    '''
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    some_d = {}

    for i in range(len(weekday_text)):
        open = weekday_text[i]['open']
        close = weekday_text[i]['close']

        open_t = datetime.strptime(open['time'], "%H%M")
        close_t = datetime.strptime(close['time'], "%H%M")

        formatted_time_open = open_t.strftime("%I:%M%p")
        formatted_time_close = close_t.strftime("%I:%M%p")

        day = weekday_text[i]['open']['day']
        some_d[days[day]] = formatted_time_open + " - " + formatted_time_close
        
    return some_d


def google_querier(gmaps, query_string):
    '''
    Function to submit a query to the Google Maps API.
    
    Parameters:
        - gmaps: API engine.
        - query_string: A string to submit to the API.
     
     Returns:
        - response_list: Dataframe with all results containing reviews.
    '''
    
    # Container for responses
    response_list = []

    # Get the response
    response = gmaps.places(query = query_string)
 
   # Check the response status and length (not empty)
    if response['status'] == 'OK' and len(response['results']) > 0:

        # For each place matching our query
        for place in response['results']:
            # Extract place ID
            place_id = place['place_id']
            # Extract place details
            place_details = gmaps.place(place_id)
            # If the response have reviews
            if 'reviews' in place_details['result']:
                # Get the location
                location = place_details['result']['geometry']['location']
                # Extract latitud and longitud
                lat = location['lat']
                lng = location['lng']
                # Extract the name
                name = place_details['result']['name']
                # For each review
                for review in place_details['result']['reviews']:
                    # Extract author
                    author_name = review['author_name']
                    # Rating
                    rating = review['rating']
                    # Review text
                    text = review['text']

                    # Add metadata if available: opening hours
                    if 'current_opening_hours' in place_details['result']:
                        weekday_text = place_details['result']['current_opening_hours']['periods']
                        m_f_schedule = day_schedule_periods(weekday_text)
                    else:
                        m_f_schedule = {}

                    # Append all details to response list
                    response_list.append({
                        'place_id': place_id,
                        'type': query_string,
                        'name': name,
                        'lat': lat,
                        'lng': lng,
                        'author_name': author_name,
                        'rating': rating,
                        'text': text,
                        'opening_hours': m_f_schedule,
                    })

    return pd.DataFrame(response_list)


def check_dataframe_results(df):
    '''
    Simple function to check the resulting dataframe.

    Parameters:
        - df: Dataframe to summarize.

    Returns:
        - df.info(): Prints dataframe shape and info.
        - df.head(): Displays the first 5 rows of data.
    '''

    print(f"Resulting dataframe has shape {df.shape}\n")
    print(df.info())

    display(df.head())


def google_nearby(gmaps, place_id: str, keys: list, location: dict, radius: int):
    '''
    Function to retrieve nearby businesses for a given location.

    Parameters:
        - place_id: Unique ID for the place of interest.
        - key: The query string for Google Maps (e.g., the text one would normally input in the search box).
        - location: A dictionary with lat, long as the center of the search.
        - radius: The radius of search from the center location.

     Returns:
        - results: Dataframe with the resulting businesses.
    '''
    
    # Results container
    results = []
    
    for k in keys:
        # Get the results
        transport_results = gmaps.places_nearby(location=location, radius=radius, type=k)

        # Parse the results
        for transport in transport_results['results']:
            transport_id = transport['place_id']
            transport_name = transport['name']
            transport_location = transport['geometry']['location']
            transport_lat = transport_location['lat']
            transport_lng = transport_location['lng']
            distance = round(distance_calc(location["lat"], location["lng"], transport_lat, transport_lng)) # rounding the value (distance in meters)

            results.append({
                'place_id': place_id,
                'transport_id': transport_id,
                'transport_name': transport_name,
                'transport_type': k,
                'transport_lat': transport_lat,
                'transport_lng': transport_lng,
                'distance_gym_transport': distance
            })

    return pd.DataFrame(results)


def distance_calc(lat1, lon1, lat2, lon2):
    '''
    Function to calculate the Euclidean distance between two points in X, Y coordinates.
    
    Parameters:
        - lat1, lon1: Latitud and longitud for point 1.
        - lat2, lon2: Latitud and longitud for point 2.

     Returns:
        - distance_in_meters: Euclidean distance in meters.
    '''

    distance_calculator = geodesic((lat1, lon1), (lat2, lon2))

# Calculate the distance in meters
    distance_in_meters = distance_calculator.meters 

    return distance_in_meters


def check_response(url):
    '''
    Function to check the response from the requests call.

    Parameters:
        - url: Simple string with the url.

    Returns:
        - html_text: Html text response.
    '''
    
    # Get the response and its status code
    response = requests.get(url)
    status_code = response.status_code
    
    # If status code not successful
    if status_code != 200:
        return False
    # Else, extract content
    else:
        html_text = response.content
        return html_text


def make_soup(html_text):
    '''
    Parses a given HTML text and creates a BeautifulSoup object for easy HTML parsing and manipulation.

    Parameters:
        - html_text: A string containing HTML content. This is the text that will be parsed and converted into a BeautifulSoup object.

    Returns:
        - BeautifulSoup object: An object that represents the parsed HTML.
    '''
    return BeautifulSoup(html_text, 'html.parser')


def get_contents(soup):
    '''
    Extracts and structures review data from a BeautifulSoup object representing parsed HTML of a reviews page.

    Parameters:
        - soup (BeautifulSoup object): A BeautifulSoup object created from HTML content of a webpage, specifically structured to contain reviews.

    Returns:
        - list: A list of dictionaries, where each dictionary represents a single review's details extracted from the HTML.
    '''

    entries = []
    keys = ['datetime', 'name', 'rating', 'title', 'review', 'event_time']
    reviews = soup.find_all('div', class_ = 'styles_reviewCardInner__EwDq2')
    
    for review in reviews:
        try:
            text = review.find('p', class_ = 'typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn').get_text()
        except AttributeError:
            continue
        
        date = (review.find('time')).attrs['datetime']
        name = review.find('span', class_ = 'typography_heading-xxs__QKBS8 typography_appearance-default__AAY17').get_text()
        rating = re.search('(\d+)(?=\s*.svg)', str(review.find('div', class_ =  'star-rating_starRating__4rrcf star-rating_medium__iN6Ty')))[0]
        title = review.find('h2', class_ = 'typography_heading-s__f7029 typography_appearance-default__AAY17').get_text()
        event = review.find('p', class_ = 'typography_body-m__xgxZ_ typography_appearance-default__AAY17').get_text()[20:]
        
        values = [date, name, rating, title, text, event]
        d = dict(zip(keys, values))
        entries.append(d)

    return entries


def trustpilot_crawler(key, verbose=False):
    '''
    Simple WebCrawler specifically for Trustpilot reviews.

    Parameters:
        - key: The key of the business to query.
        - verbose: To print messages.
        
    Returns:
        - df: A DataFrame containing all scraped review data, or an empty DataFrame if the page does not exist.
    '''
    if verbose:
        print(f"Trying to get reviews for {key}...")

    # Base URL
    base_url = f'https://dk.trustpilot.com/review/{key.lower()}.dk'

    # Check it exists
    response = check_response(base_url)
    # If it doesn't exist, return an empty DF
    if not response:
        return pd.DataFrame()

    # Pages
    i = 1
    # Container for pages
    pages = []

    # Extract responses from pages
    while response:
        if verbose:
            print(f"Trying responses for {key} - Page {i}.")
        full_url = base_url + f'?page={i}&sort=recency'
        response = check_response(full_url)

        # If not successfull response, we are finished
        if not response:
            break
        else:
            # Otherwise, parse the response and add to current DF
            soup = make_soup(response)
            d = get_contents(soup)
            # Add response to our collection of DFs
            pages.append(pd.DataFrame.from_dict(d))

        # On to next page
        i += 1

    # Add a key with the facility
    df = pd.concat(pages)
    df["enterprise"] = key

    # Return the resulting DF
    return df


def get_lat_lng(gmaps, address):
    '''
    This function retrieves the latitude and longitude of a specified address using the Google Maps API.
    
    Parameters:
        - gmaps (Google Maps Client object): An instance of the Google Maps Client, used to interact with the Google Maps API.
        - address (str): The address for which latitude and longitude coordinates are to be obtained.

    Returns:
        - tuple: A tuple containing the latitude and longitude (lat, lng) of the given address if successful.
        - bool: Returns False if the geocoding process fails or if the address is not found.
    '''
    try:
        geocode_result = gmaps.geocode(address)

        if geocode_result and len(geocode_result) > 0:
            location = geocode_result[0]['geometry']['location']
            return (location['lat'], location['lng'])
        else:
            return False
        
    except Exception as e:
        print(f"An error occurred while geocoding: {e}")
        return False
    
class KBHFacilitiesWebScraper:
    '''
    A web scraper for collecting information about various facilities in Copenhagen from a specified URL.
    
    Parameters:
        - url: The URL to scrape data from. Default is 'https://kbhkort.kk.dk/spatialmap?page=widget-view&name=motion/motionslisten'.
        - chrome_driver_path: Path to the Chrome WebDriver executable. Default is 'libraries/chromedriver-win64/chromedriver.exe'.

    Returns:
        - df_test: A DataFrame containing the scraped and processed data from the specified URL.
    '''

    def __init__(self, url: str='https://kbhkort.kk.dk/spatialmap?page=widget-view&name=motion/motionslisten', chrome_driver_path: str='libraries/chromedriver-win64/chromedriver.exe'):

        # Define URL
        self.url = url

        # Initialise a dictionary to hold the scraped data
        self.data = {'type':[], 'activity':[], 'location':[], 'website':[], 'gender':[], 'age':[], 'special':[], 'address':[]}

        options = ChromeOptions()       # Get Chrome options
        options.headless = True         # This stops an actual browser from being open and shown
        self.driver = Chrome(chrome_driver_path, options=options)  # Optional argument, if not specified will search path.
        self.driver.get(url)

        print("Driver and URL passed. Wait a second...")
        self.driver.implicitly_wait(20) # Wait for the website to load

        self.count = len(self.driver.find_elements(By.XPATH, '/html/body/div/div[4]/div/ul/li'))
        print(f'There is a total of {self.count} entries. Use the ".get()" method to get all entries.')

    
    def __crawl(self):

        # At the time of writing, there were 606 entries (2023-11-16)
        for i in tqdm(range(1, self.count+1)): 

            # Activity type/category - this is indicated by the associated icon. The icon name is collected.
            icon = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[1]/img')
            self.data['type'].append(icon.get_attribute('src'))

            # Name/title of the activity
            type = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[1]/strong[1]')
            self.data['activity'].append(type.text)

            # Location of activity - not all entries list a location
            try:
                location = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[1]/strong[2]')
                self.data['location'].append(location.text)
            except NoSuchElementException:
                self.data['location'].append(None)

            # Website
            try:
                site = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[2]/span[1]/a')
                self.data['website'].append(site.get_attribute('href'))
            except NoSuchElementException:
                self.data['website'].append(None)
            

            # Gender
            gender = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[2]/span[4]') # This works
            self.data['gender'].append(gender.text)

            # Age Group
            age = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[2]/span[5]')
            self.data['age'].append(age.text)

            # Address of activity - some entries have an extra field, so this messess with the current XPath implementation
            # Therefore, each element needs to be checked that it is indeed an address.
            # Luckily, all address entries in the table contain the prefix 'Mødested' (meeting place).
            try:
                address = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[2]/span[6]')
                if not address.text.startswith('Mødested: ') and address.text.startswith("| Særlig"): # Check that the text is indeed the address
                    self.data['special'].append(address.text)
                    try:
                        address = self.driver.find_element(By.XPATH, f'/html/body/div/div[4]/div/ul/li[{i}]/div[2]/div[2]/span[7]')
                        self.data['address'].append(address.text) #.removeprefix('Mødested: '))
                    except NoSuchElementException:
                        self.data['address'].append(None)
                else:
                    self.data['special'].append(None)
                    self.data['address'].append(address.text) #.removeprefix('Mødested: '))
            except NoSuchElementException:
                self.data['special'].append(None)
                self.data['address'].append(None)

        self.driver.quit()


    def get(self):
        
        # Crawl and store data in self.data containers
        self.__crawl()

        # Create and inspect the dataframe
        df_raw = pd.DataFrame.from_dict(self.data)

        dict_type = {'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/boldspil_26x26.png': 'ball_sports',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/dans_26x26.png': 'dance',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/fitness_26x26.png': 'fitness',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/gym_26x26.png': 'gym',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/kampsport_26x26.png': 'martial_arts',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/loeb_26x26.png': 'running',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/natur_26x26.png': 'nature',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/svoemning_26x26.png': 'swimming',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/udemotion_26x26.png': 'outdoors',
                    'https://kbhkort.kk.dk/images/ikoner/suf/sundhed/yoga_26x26.png': 'yoga'}
        dict_gender = {'Køn: Begge':'both', 'Køn: Kvinder':'women', 'Køn: Mænd':'men'}
        dict_age = {'| Aldersgruppe: Alle': 'all', '| Aldersgruppe: Seniorer': 'seniors'}
        dict_special = {'+ 65 år':'65+',
                        '+60':'60+', 
                        '+60 år':'60+', 
                        '+65':'65+', 
                        '+65 år':'65+',
                        '65+ år':'65+',
                        'Kvinder 45 +':'45+', 
                        'Kvinder 65+ år':'65+',
                        'Mænd 65 år+':'65+',
                        'PAN har fokus på inklusion af mennesker med et særligt fokus på seksuel mangfoldighed og kønsdiversitet.':'PAN har fokus på inklusion af mennesker med et særligt fokus på seksuel mangfoldighed og kønsdiversitet',
                        'mænd +65 år':'65+'}
        dict_address = {"":"None"}

        df_test["special"] = df_test["special"].apply(lambda s:s.replace('| Særlig målgruppe: ', ''))
        df_test["address"] = (df_test["address"].str.removeprefix("Mødested:")).str.strip()

        mask = df_test["location"]=='Kommunal park'
        df_test.loc[mask, ["activity", "location"]] = (df_test.loc[mask, ["location","activity"]].values)

        df_test = df_test.replace({"type": dict_type, "gender": dict_gender, "age": dict_age, "special":dict_special, "address":dict_address})

        temp = {'health':['fysio', 'hjært', 'nær', 'puls', 'hjert', 'mind', 'knæ', 'ryg', 'senior', 'stabil', 'mobil'],
                'sport':['bold', 'ball', 'tennis', 'minton', 'golf', 'cricket', 'volley'], 
                'fitness':['yoga', 'træn', 'gym', 'motion', 'fitness', 'cyk', 'løb', 'stav', 'cross', 'ro', 'kajak', 'zumba', 'pilates', 'kamp', 'svøm', 'spin', 'kondi'], 
                'recreation':['billiard', 'billard', 'dart', 'dans', 'dance', 'bowl', 'gå', 'walk', 'spil', 'petanque', 'park', 'have'] 
                }

        dict_activity = {}
        for k,v in temp.items():
            for x in v:
                dict_activity.setdefault(x,k)

        temp = df_test["activity"].copy()
        for i in range(len(temp)):
            for key in dict_activity.keys():
                if re.search(key, temp[i].lower()): #      key in temp[i].lower():
                    temp[i] = str(dict_activity[key].strip(""))
                    break

            if temp[i] not in dict_activity.values():
                #print(i, temp[i])
                temp[i] = "other"

        sorted(list(zip(np.unique(temp, return_counts=True)[0], np.unique(temp, return_counts=True)[1])), key = lambda x:x[1]), len(np.unique(temp, return_counts=True)[0])

        df_test["category"] = temp

        df_test = df_test[['type', 'activity', 'category', 'location', 'website', 'gender', 'age', 'special','address']]

        return self.df_test
    

def remove_emojis(text):
    '''
    Custom function to remove unicode characters depicting an emoji.

    Parameters:
        - text: String with potential emojis.

    Returns:
        - text: The string without emojis.
    '''

    for lang in UNICODE_EMOJI:
        for em in UNICODE_EMOJI[lang]:
            text = text.replace(em, "")

    return text
    
"""
def translate_text(input_text, model, tokenizer):
    '''
    Custom function to translate text. Can be applied to pandas dataframe column.

    Parameters:
        - input_text: The text in original language to translate.
        - model: The HuggingFace model object.
        - tokenizer: The HuggingFace tokenizer object.

    Returns:
        - translated_text: The translated text.
    '''
    
    input_ids = tokenizer.encode(input_text, max_length=512, return_tensors="pt")
    translated_ids = model.generate(input_ids, max_length=512)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return translated_text
"""

def translate(df, text_colname: str, translation_colname: str, model_name: str="Helsinki-NLP/opus-mt-da-en"):
    '''
    Custom function to translate a text column from a given dataset using a default Huggingface pretrained model.

    Parameters:
        - df: The dataframe that contains the text column to translate.
        - model_name: The HuggingFace model name. By default danish to enlish.

    Returns: 
        - df: The original dataframe with the translations in a new column.
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    max_length = 512

    translations = []

    for ix, row in df.iterrows():

        # For showing progress
        clear_output(wait=True)
        print(f"{ix / len(df):.2%}", end="\r")

        # Get the text to translate
        input_text = row[text_colname]

        # If text is longer than max tokens, split, translate each chunk and concatenate the results
        if len(input_text) > max_length:

            # Split the input text into smaller chunks
            input_chunks = [input_text[i:i+max_length] for i in range(0, len(input_text), max_length)]

            translated_chunks = []
            for chunk in input_chunks:
                input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=max_length, truncation=True)
                translated_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
                translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
                translated_chunks.append(translated_text)
            
            translated_text = ''.join(translated_chunks)

        # If less than max_length, one sweep
        else:
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)
            translated_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        translations.append(translated_text)

    # For showing progress
    print(f"{ix / len(df):.2%} - FINISHED!")
    
    # Append to dataset
    df[translation_colname] = translations

    return df


def get_reviews(gmaps, search_input):
    '''  
    Retrieves reviews for a given place using Google Maps API.
    
    Parameters:
        - gmaps: The Google Maps client instance.
        - search_input: The text query for finding the place on Google Maps.
        
    Returns: 
        - list of tuples: A list of tuples containing the author's name, review text, and rating for each review, if found. Returns None if no reviews are found or in case of an exception.
    '''
    try:
        place_result = gmaps.find_place(input=search_input, input_type='textquery')
        if place_result and len(place_result['candidates']) > 0:
            place_id = place_result['candidates'][0]['place_id']
            place_details = gmaps.place(place_id=place_id)
            reviews = place_details['result'].get('reviews', [])
            return [(review.get('author_name'), review['text'], review.get('rating')) for review in reviews]
        else:
            return None
    except Exception as e:
        return None
    
    
def review_finder(gmaps, df):
    '''
    Improves the DataFrame with review information and geographical coordinates for each entry.
    
    Parameters:
        - gmaps: The Google Maps client instance.
        - df: A DataFrame containing columns 'address' and 'location'.

    Returns:
        new_df: An updated DataFrame with additional columns for latitude ('lat'), longitude ('lng'), review author ('author'), review text ('review'), and review rating ('rating'). Rows without reviews are excluded.
    '''

    df['lat'] = None
    df['lng'] = None

    list = []

    for index, row in df.iterrows():
        address = str(row['address']).strip() 
        location = str(row['location']).strip()

        # try getting reviews based on location
        reviews = get_reviews(location)
        
        #if no reviews based on location try the address
        if not reviews:
            get_reviews(gmaps, address)
                    
        lat_lng = None
        #look first if we can find lat and lng for the address
        if address:
            lat_lng = get_lat_lng(address)
        # if there is no address or we can't find the coordiantes using address we use the location to find the lat and lng
        if not lat_lng: 
            lat_lng = get_lat_lng(location)
        
        if reviews:
            for review in reviews:
                review_author, review_text, review_rating = review 
                new_row = row.to_dict()
                if lat_lng:
                    new_row['lat'] = lat_lng[0]
                    new_row['lng'] = lat_lng[1]
                else:
                    new_row['lat'] = None
                    new_row['lng'] = None
                    
                new_row['author'] = review_author  # add author of text
                new_row['review'] = review_text  # add review text
                new_row['rating'] = review_rating  # add review rating
                list.append(new_row)
        else:
            new_row = row.to_dict()
            if lat_lng:
                new_row['lat'] = lat_lng[0]
                new_row['lng'] = lat_lng[1]
            else:
                new_row['lat'] = None
                new_row['lng'] = None
            
            new_row['author'] = None
            new_row['review'] = None
            new_row['rating'] = None
            list.append(new_row)
        
    new_df = pd.DataFrame(list)
    new_df = new_df[new_df['review'].notna() & new_df['review'].ne('')]

    return new_df

def fleiss_kappa(annotations, categories, labels):
    '''
    Custom function to calculate Fleiss' Kappa for IAA (based on https://en.wikipedia.org/wiki/Fleiss%27_kappa).
    
    Parameters:
        - annotations: A DataFrame containing annotations with a unique ID for each item.
        - categories: A list of categories to be evaluated in the annotations.
        - labels: A list of possible labels.

    Returns:
        - k: The calculated Fleiss' Kappa value, representing the IAA.
    '''
    
    overlapping_IDs = annotations["ID"].unique()

    if len(overlapping_IDs) < 2:
        raise Exception("We need at least 2 overlapping annotations to calculate IAA.")

    agreement_table = []

    # Look at each review ID
    for id in overlapping_IDs:
        # (We need to keep a list for each row)
        _ = []
        # Look at each category for that review
        for cat in categories:
            # Look at each potential label for the review and category
            for label in labels:
                # Count number of agreements
                subset = annotations.loc[annotations.ID == id, cat]
                if np.isnan(label):
                    n = len(subset[subset.isna()])
                else:
                    n = len(subset[subset == label])
                # Append the agreement count to the row
                _.append(n)
        # Append the row to the table
        agreement_table.append(_)

    # Create the table
    agreement_table = pd.DataFrame(agreement_table)

    ### Find Pi vectors
    # Apply exponent to each element and sum across rows
    Pi = np.mean((agreement_table.apply(lambda x: x**2).sum(axis=1) - agreement_table.sum(axis=1)) / (agreement_table.sum(axis=1)*(agreement_table.sum(axis=1)-1)))

    # Calculate P expected
    Pe = sum((agreement_table.sum() / agreement_table.sum().sum()) **2)

    # Final Kappa
    k = (Pi - Pe)/(1 - Pe)

    return k

def parse_label_studio_file(filepath):
    '''
    Parses a JSON file from Label Studio and converts it into a pandas DataFrame.
    
    Parameters:
        - filepath: The file path of the JSON file to be parsed.

    Returns:
        - pandas.DataFrame: A DataFrame with columns for each extracted data point and rows for each annotation instance.
    '''
    
    # Open file
    file = open(filepath, encoding="utf-8")
    # Parse JSON to dict
    jfile = json.load(file)

    # Collect reviews
    reviews = {}

    # Function to retrieve point of the label
    def point(label):

        if label == "Positive":
            return 1
        elif label == "Negative":
            return -1
        else:
            raise Exception("Not a valid label")

    # JSON file is a list of instances
    for row in jfile:

        # Extract the instance id and text review
        reviews[row["id"]] = {}
        reviews[row["id"]]["ID"] = row["data"]["ID"]
        reviews[row["id"]]["text"] = row["data"]["text"]
        
        # For each annotation of the instance
        for a in row["annotations"]:

            # Extract the result
            result = a["result"][0]["value"]["taxonomy"]
            
            # The result is a list of labels
            for label in result:
                # If the label is a list of size one, it is just sentiment
                if len(label) == 1:
                    # Assign to "Not Determined"
                    if "Not Determined" in reviews[row["id"]]:
                        reviews[row["id"]]["Not Determined"] += point(label[0])
                    else:
                        reviews[row["id"]]["Not Determined"] = point(label[0])
                # Else it is a tuple of the form "Sentiment, Object"
                else:
                    if label[1] in reviews[row["id"]]:
                        reviews[row["id"]][label[1]] += point(label[0])
                    else:
                        reviews[row["id"]][label[1]] = point(label[0])

    # Return as a Pandas DataFrame
    return pd.DataFrame.from_dict(reviews, orient="index")


def compute_BLEU(references, translations):
    '''
    Compute BLEU scores for a list of reference sentences and a list of translated sentences.

    Parameters:
        - list_of_references: List of lists, where each sublist contains reference translations for a sentence.
        - list_of_candidates: List of sentences generated by the model.

    Returns:
        - bleu_scores: List of BLEU scores for each sentence.
        - cumulative_bleu: Overall BLEU score for the entire list.
    '''
    # Ensure that the lengths of the reference and candidate lists are the same
    assert len(references) == len(translations), "Mismatched number of references and candidates"

    # Compute BLEU scores using NLTK
    bleu_scores = [corpus_bleu([refs], [cand]) for refs, cand in zip(references, translations)]

    # Compute the cumulative BLEU score for the entire list
    cumulative_bleu = corpus_bleu(references, translations)

    return bleu_scores, cumulative_bleu


def compute_WER(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and translation.

    Parameters:
        - reference: list of words (human translation)
        - hypothesis: list of words (machine translation)

    Returns:
        - wer: Word Error Rate
    """
    # We lowercase everything to avoid errors solely based in caseing
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    # Create a matrix to store the minimum edit distances
    distance_matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]

    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                distance_matrix[i][j] = j
            elif j == 0:
                distance_matrix[i][j] = i
            else:
                # Calculate the minimum edit distance
                insertion = distance_matrix[i][j - 1] + 1
                deletion = distance_matrix[i - 1][j] + 1
                substitution = distance_matrix[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1)

                distance_matrix[i][j] = min(insertion, deletion, substitution)

    # The bottom-right cell contains the minimum edit distance
    wer = distance_matrix[-1][-1]

    # Normalize by the length of the reference
    wer /= len(reference)

    return wer

class WER:
    '''
    Class to compute Word Error Rate between reference translations (human) and predictions (machine).
    It calculate some metrics like average WER for all instances, and provides ranking of best and worst instances
    for inspection.
    '''
    
    def __init__(self, texts: list, references: list, predictions: list):
        # We first make sure the shapes are equal
        assert len(texts) == len(references) and len(references) == len(predictions), print("WARNING: Reference and Predictions array should have the same length.")
        self.texts = texts
        self.references = references
        self.predictions = predictions
        self.n = len(self.references)
        pass

    def __compute(self):
        # Container
        self.WERs = []
        # For each review
        for ix in range(len(self.texts)):
            # Show progress
            self.WERs.append((self.texts[ix], self.references[ix], self.predictions[ix], compute_WER(self.references[ix], self.predictions[ix])))

        self._ranking = pd.DataFrame(self.WERs, columns=["Text", "Human", "Machine", "WER"])
        self._ranking.sort_values("WER", ascending=False, inplace=True)
        
    def mean(self):

        self.__compute()
        return sum([x[3] for x in self.WERs]) / self.n

    def ranking(self, how="full"):
        assert how in ["top", "bottom", "full"], print("How argument must be either top, bottom, both or full")

        if how=="full":
            return self._ranking
        elif how=="top":
            return self._ranking.head()
        else:
            return self._ranking.tail()


def num_to_sent(x):
    '''
    Custom function to map numerical-encoded sentiments to human-readable labels.
    
    Parameters:
        - x: The numerical sentiment value to be mapped.

    Returns:
        - str: The textual representation of the sentiment ("Positive", "Neutral", "Negative", or "None").
    '''
    
    if x == 1.0:
        return "Positive"
    elif x == 0.0:
        return "Neutral"
    elif x == -1.0:
        return "Negative"
    else:
        return "None"
        

def lemmatize_with_postag(sentence):
    '''
    Function to lemmatize each word with its POS tag.
    
    Parameters:
        - sentence: The sentence to be lemmatized.
    
    Returns:
        - str: The lemmatized sentence.
    '''
    
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    
    return " ".join(lemmatized_list)

def get_icon(activity_type):
    '''
    Returns an icon based on the activity type.
    
    Parameters:
        - activity_type: The type of activity to get the icon for.
        
    Returns: 
        - str: The name of the icon associated with the given activity type.
    '''
    
    icons = {
        "gym": "dumbbell", 
        "fitness": "heartbeat", 
        "outdoors": "tree",
        "nature": "leaf", 
        "running": "running", 
        "ball_sports": "users",
        "swimming": "swimmer", 
        "yoga": "om", 
        "dance": "music",
        "martial_arts": "fist-raised"
    }
    
    return icons.get(activity_type, "question-sign")


def get_rating_average(data, lat_field, lng_field):
    '''
    Computes the average rating for each location/address in the data and returns a function to determine the color coding based on this average rating.
    
    Parameters:
        - data: The DataFrame containing ratings and location data.
        - lat_field: The name of the column in 'data' representing latitude.
        - lng_field: The name of the column in 'data' representing longitude.
        
    Returns:
        - function: A function that takes latitude and longitude as input and returns a color.
    '''
    
    data['lat_lng'] = data[[lat_field, lng_field]].apply(tuple, axis=1)

    avg_rating_dict = data.groupby('lat_lng')['rating'].mean().to_dict()

    def get_color_for_location(lat, lng):
        lat_lng = (lat, lng)
        avg_rating = avg_rating_dict.get(lat_lng, 0)
        if avg_rating >= 4:
            return 'green'
        elif avg_rating >= 2:
            return 'orange'
        else:
            return 'red'

    return get_color_for_location


def add_kbh_markers(grouped_data, marker_cluster):
    '''
    Adds customized markers to a Folium map based on data provided in 'grouped_data'.
    
    Parameters:
        - grouped_data: Data with latitude, longitude, activity, and rating columns.
        - marker_cluster: The marker cluster object to add markers to.
    
    Returns:
        - None: This function does not return a value, it modifies `marker_cluster` in place.
    '''
    
    get_color = get_rating_average(grouped_data, 'lat', 'lng')
    
    for index, row in grouped_data.iterrows():
        lat, lng = row['lat'], row['lng']
        
        if lat is None or lng is None:
            continue
        try:
            lat_float = float(lat)
            lng_float = float(lng)
            
            color = get_color(lat_float, lng_float)
            icon = get_icon(row['type'])
            popup_content = f"Type: {row['activity']}<br> Rating: {row['rating']}"
            marker_cluster.add_child(
                folium.Marker(
                    location=[lat_float, lng_float],
                    popup=popup_content,
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                )
            )
        except ValueError:
            continue


def add_google_markers(grouped_data, marker_cluster):
    '''
    Adds customized markers to a Folium map based on data provided in 'grouped_data'.
    
    Parameters:
        - grouped_data: Data with latitude, longitude, type, and rating columns.
        - marker_cluster: The marker cluster object to add markers to.
    
    Returns:
        - None: This function does not return a value, it modifies `marker_cluster` in place.
    '''
    
    get_color = get_rating_average(grouped_data, 'lat', 'lng')
    
    for index, row in grouped_data.iterrows():
        
        lat, lng = row['lat'], row['lng']
        
        if lat is None or lng is None:
            continue
        try:
            
            lat_float = float(lat)
            lng_float = float(lng)
            
            color = get_color(lat_float, lng_float)
            popup_content = f"Type: {row['type']}<br> Rating: {row['rating']}"
            marker_cluster.add_child(
                folium.Marker(
                    location=[lat_float, lng_float],
                    popup=popup_content,
                    icon=folium.Icon(color=color, icon="dumbbell", prefix='fa')
                )
            )
        except ValueError:
            continue


def grammar_corrector(text:str) -> str:
    """
    Corrects spelling and grammar in the given text.

    Parameters:
        - text: The input text to be corrected. It can be a single string or a list of strings.

    Returns:
        - cleaned_text: The corrected text, with spelling and grammar issues fixed.
    """
    
    cleaned_text = []

    if isinstance(text, str):
        text = [text]  # Convert a single string to a list of strings for consistency.

    for line in text:
        temp_line = []
        words = line.split()
        for _, word in enumerate(words):
            # Check if the word contains a numeric character
            has_numeric = any(char.isdigit() for char in word)

            if has_numeric:
                # If the word contains a numeric character, keep the original word
                corrected_word = word
            else:
                # If the word does not contain a numeric character, perform correction
                corrected_word = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
                corrected_word = corrected_word[0].term if corrected_word and corrected_word[0] else word

            # Append the punctuation back to the corrected word if the original word had it
            if word[-1] in ['!', '?', '.']:
                corrected_word += word[-1]

            temp_line.append(corrected_word)

            # Add space between words, except for the last word
            if _ < len(words) - 1:
                temp_line.append(' ')

        cleaned_text.append(''.join(map(str, temp_line)))

    if isinstance(text, str):
        return cleaned_text[0]  # Return the corrected string.
    else:
        return cleaned_text
    

def word_counter(text_array, stopwords=set(stopwords.words('english'))):
    '''
    Counts the occurrences of each word in a given array of text, excluding stopwords.
    
    Parameters:
        - text_array: An array of strings to be analyzed.
        - stopwords: A set of words to be excluded from the count. 

    Returns:
        - word_counts: A Counter object mapping each word to its frequency in the text.
    '''
    
    # Combine all the text into a single string
    combined_text = ' '.join(text_array)

    # Tokenize the text using NLTK
    words = nltk.word_tokenize(combined_text)

    # Remove stopwords
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords]

    # Use Counter to count the occurrences of each word
    word_counts = Counter(words)

    return word_counts


def vertical_diverging_bar(x, pyvalues, nyvalues, x_label=None,y_label=None, text_font="Times New Roman", text_size=32):
    '''
    Creates a vertical diverging bar chart using Plotly.

    Parameters:
        - x: Categories along the x-axis.
        - pyvalues: Positive values corresponding to each x-axis category.
        - nyvalues: Negative values corresponding to each x-axis category.
        - x_label: Label for the x-axis. Defaults to None.
        - y_label: Label for the y-axis. Defaults to None.
        - text_font: Font family for text in the chart. Defaults to "Times New Roman".
        - text_size: Font size for text in the chart. Defaults to 32.

    Returns:
        - diverging: A Plotly Figure object representing the diverging bar chart.
    '''
    
    # Create figure
    diverging = go.Figure()

    x, py, ny = x, pyvalues.values, nyvalues.values
    colors = sns.color_palette("colorblind").as_hex()
    positive_color = colors[0]
    negative_color = colors[1]
    
    # Plot Positive values
    diverging.add_trace(go.Bar(
    y=py,
    x=x,
    text = py,    
    textfont=dict(
        family=text_font,
        size=text_size,
        color="Black"
    ),
    orientation='v',
    name="Positive",
    marker_color = positive_color,
    customdata=py,
    hovertemplate="%{y}: %{x}"
    ))

    # Plot Negative values
    diverging.add_trace(go.Bar(
    y=-ny,
    x=x,
    text = ny,    
    textfont=dict(
        family=text_font,
        size=text_size,
        color="Black"
    ),
    orientation='v',
    name="Negative",
    marker_color =  negative_color,
    customdata=ny,
    hovertemplate="%{y}: %{customdata}"
    ))

    # Format plot
    diverging.update_layout(barmode='relative',
                        height=1280,
                        width=720,
                        bargap=0.42,
                        legend_orientation='v',
                        legend_x=1,legend_y=0,
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        xaxis = dict(
                        tickfont = dict(size=text_size)),
                        font=dict(
                        family=text_font,
                        size=text_size,
                        color="Black"
                        )
                        )

    return diverging