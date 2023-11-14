
############ UTILS ############

# IMPORTS
from datetime import datetime
import pandas as pd
from geopy.distance import geodesic
import requests
import re # regular expressions
from bs4 import BeautifulSoup # xml parsing
import pandas as pd
import regex as re
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def day_schedule_periods(weekday_text):
    '''
    Author: Constantin-Bogdan Craciun

    TODO: DESCRIPTION
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
    Author: Constantin-Bogdan Craciun

    Function to submit a query to the Google Maps API.
    Arguments:
        - gmaps: API engine
        - query_string: A string to submit to the API
     
     Returns:
        - dataframe with all results containing reviews.
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
    Author: Gino F. Fazzi
    Simple function to check the resulting dataframe.

    Arguments:
     - df : Dataframe to summarize

    Output:
     - Prints dataframe shape and info.
     - Displays the first 5 rows of data.
    '''

    print(f"Resulting dataframe has shape {df.shape}\n")
    print(df.info())

    display(df.head())


def google_nearby(gmaps, place_id: str, keys: list, location: dict, radius: int):
    '''
    Author: Gino F. Fazzi (adapted from Veron Hoxha Notebook)

    Function to retrieve nearby businesses for a given location.

    Arguments:
     - place_id: Unique ID for the place of interest.
     - key: The query string for Google Maps (e.g., the text one would normally input in the search box).
     - location: A dictionary with lat, long as the center of the search.
     - radius: The radius of search from the center location.

     Returns:
      - DF with the resulting businesses.
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
    Author: Constantin-Bogdan Craciun

    Function to calculate the Euclidean distance between two points in X, Y coordinates.
    
    Arguments:
     - lat1, lon1: Latitud and longitud for point 1
     - lat2, lon2: Latitud and longitud for point 2

     Returns:
      - Euclidean distance in meters
    '''

    distance_calculator = geodesic((lat1, lon1), (lat2, lon2))

# Calculate the distance in meters
    distance_in_meters = distance_calculator.meters 

    return distance_in_meters


def check_response(url):
    '''
    Author: Christian Margo Hansen

    Function to check the response from the requests call.

    Arguments:
     - url: Simple string with the url.

    Returns:
     - html text response
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
    Author: Christian Margo Hansen

    TODO: DESCRIPTION
    '''
    return BeautifulSoup(html_text, 'html.parser')


def get_contents(soup):
    '''
    Author: Christian Margo Hansen

    TODO: DESCRIPTION
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
    Author: Christian Margo Hansen / Gino F. Fazzi

    Simple WebCrawler specifically for Trustpilot reviews.

    Arguments:
     - key: The key of the business to query.
     - verbose: To print messages.
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
    Author: Veron Hoxha

    TODO: DESCRIPTION
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