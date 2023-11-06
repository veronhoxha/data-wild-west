import json
import pandas as pd

filepath = "../Annotations/Gino_annotations.json"

test = pd.read_csv("../Annotations/Gino.csv", encoding="utf-8")

# Open file
file = open(filepath, encoding="utf-8")
# Parse JSON to dict
jfile = json.load(file)

new_json = []

textes = list(test.text)


for row in jfile:

    if row["data"]["text"] in list(test.text):

        id = int(test.loc[test.text == row["data"]["text"], "ID"].values[0])
        row["data"]["ID"] = id
        new_json.append(row)


#with open("new_json.json", "w") as outfile:
#    json.dump(new_json, outfile)
print(test.head())

test2 = pd.DataFrame(new_json)
print(test2.head())