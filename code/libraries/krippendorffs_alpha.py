import numpy as np
import krippendorff

### Calculating the overall Krippendorff's Alpha for all categories

def krippendorff_alpha(annotations, categories):
    '''
    Calculating Krippendorff's Alpha for IAA (based on https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) by using the python "krippendorff" library
    '''
    
    # finding the number of annotators
    max_ratings = max(annotations.groupby("ID").count().max())
    # total number of ratings in one row (5 * 5 = 25) - 5 annotators and 5 categories
    total_ratings_per_item = max_ratings * len(categories)

    # filitering annotations to include those ID's which are repeated 5 times
    filtered_annotations = annotations.groupby("ID").filter(lambda x: len(x) == 5)
    overlapping_IDs = filtered_annotations["ID"].unique()

    if len(overlapping_IDs) < 2:
        raise Exception("We need at least 2 overlapping annotations to calculate IAA.")

    data_matrix = []
    
    # look at each review ID
    for id in overlapping_IDs:
        group = annotations[annotations['ID'] == id]
        row = []
        
        for category in categories:
            # if any annotator did not rate in this category for this item, append nan
            ratings = group[category].tolist() + [np.nan] * (max_ratings - group[category].count())
            row.extend(ratings)

        # ensure each row length equals the total number of possible ratings per item
        row = row[:total_ratings_per_item]
        data_matrix.append(row)

    # matrix where each row is an item and each column is an annotator's rating
    data_matrix = np.array(data_matrix, dtype=float)
    alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ordinal')
    
    return alpha 


### Calculating Krippendorff's Alpha seperatly for each category

def krippendorff_alpha_per_category(annotations, categories):
    
    # finding the number of annotators
    max_ratings = max(annotations.groupby("ID").count().max())
    
    # filitering the annotations which are not repeated 5 times
    filtered_annotations = annotations.groupby("ID").filter(lambda x: len(x) == 5)
    overlapping_IDs = filtered_annotations["ID"].unique()

    if len(overlapping_IDs) < 2:
        raise Exception("We need at least 2 overlapping annotations to calculate IAA.")

    category_alphas = {}
    
    # iterating over each category to calculate Krippendorff's alpha separately
    for category in categories:
        data_matrix = []
        
        # look at each review ID
        for id in overlapping_IDs:
            group = annotations[annotations['ID'] == id]
            ratings = group[category].tolist()

            # appending NaNs to ensure that each item's ratings list has the same length
            ratings += [np.nan] * (max_ratings - len(ratings))
            data_matrix.append(ratings)

        data_matrix = np.array(data_matrix, dtype=float)
        # calculating Krippendorff's alpha for the current category and store it
        alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ordinal')
        category_alphas[category] = alpha

    return category_alphas