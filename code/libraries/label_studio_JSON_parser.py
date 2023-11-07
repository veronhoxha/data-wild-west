
import json
import pandas as pd
import os


def parse_file(filepath):

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
