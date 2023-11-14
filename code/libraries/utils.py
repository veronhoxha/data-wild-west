
############ UTILS ############

# IMPORTS
from datetime import datetime
import pandas as pd
from geopy.distance import geodesic

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