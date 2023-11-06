
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

model_name = "Helsinki-NLP/opus-mt-da-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

## BELOW TOY EXAMPLE ##
"""
input_text = "Jeg drikker en øl i solen."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
translated_ids = model.generate(input_ids, max_length=512)
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
print("Translated text:", translated_text)
"""

df = pd.read_csv("./data/TrustPilot.csv", encoding="utf-16")
#df2 = pd.read_csv("./data/TrustPilot_reviews-2.csv", encoding="utf-16")
#df = pd.concat([df1, df2])
#df.drop("Unnamed: 0", axis=1, inplace=True)
#df.drop_duplicates(inplace=True)

print(df.head())
print(df.shape)
#sys.exit(0)

translations = []

for ix, row in df.iterrows():

    input_text = row.review
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    translated_ids = model.generate(input_ids, max_length=512)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    translations.append(translated_text)

    print(ix)


df["translated_reviews"] = translations

df.to_csv("./data/trustpilot_reviews_translated.csv", encoding="utf-16", index=False)