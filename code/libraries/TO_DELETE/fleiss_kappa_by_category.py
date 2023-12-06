import numpy as np
import pandas as pd

def fleiss_kappa_by_category(annotations, categories, labels):
    '''
    Custom function to calculate Fleiss' Kappa for IAA for each category separately (based on https://en.wikipedia.org/wiki/Fleiss%27_kappa).
    '''
    
    # filitering annotations to include those ID's which are repeated 5 times
    filtered_annotations = annotations.groupby("ID").filter(lambda x: len(x) == 5)
    overlapping_IDs = filtered_annotations["ID"].unique()
    
    if len(overlapping_IDs) < 2:
        raise Exception("We need at least 2 overlapping annotations to calculate IAA.")

    kappa_values = {}

    # look at each category for that review
    for category in categories:
        agreement_table = []

        # look at each review ID
        for id in overlapping_IDs:
            row = []
            # look at each potential label for the review and category
            for label in labels:
                # count number of agreements
                subset = filtered_annotations[filtered_annotations["ID"] == id]
                n = len(subset[subset[category] == label])
                row.append(n)
            # append the row to the table
            agreement_table.append(row)

        # create the table for this category
        agreement_table = pd.DataFrame(agreement_table)
        # print(agreement_table)
        # print(agreement_table.shape)
        
        # calculate Pi and Pe for each category
        Pi = np.mean((agreement_table.apply(lambda x: x**2).sum(axis=1) - agreement_table.sum(axis=1)) / (agreement_table.sum(axis=1)*(agreement_table.sum(axis=1)-1)))
        Pe = sum((agreement_table.sum() / agreement_table.sum().sum()) **2)

        # final Kappa for this category
        kappa = (Pi - Pe)/(1 - Pe)
        kappa_values[category] = kappa

    return kappa_values