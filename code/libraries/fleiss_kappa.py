
import numpy as np
import pandas as pd

def fleiss_kappa(annotations, categories, labels):
    '''
    Custom function to calculate Fleiss' Kappa for IAA (based on https://en.wikipedia.org/wiki/Fleiss%27_kappa)
    '''
    
    # filitering annotations to include those ID's which are repeated 5 times
    filtered_annotations = annotations.groupby("ID").filter(lambda x: len(x) == 5)
    overlapping_IDs = filtered_annotations["ID"].unique()

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

