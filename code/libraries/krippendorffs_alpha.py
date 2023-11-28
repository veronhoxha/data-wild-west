import numpy as np
import krippendorff


def krippendorff_alpha(annotations, categories):
    
    max_ratings = max(annotations.groupby("ID").count().max())
    # print(max_ratings)
    # num_annotators = 5
    total_ratings_per_item = max_ratings * len(categories)
    # print(total_ratings_per_item)

    # extract the subset of overlapping annotations used to calculate the IAA
    overlapping_IDs = annotations[annotations.duplicated(subset="ID", keep=False)]["ID"].unique()
    print(overlapping_IDs)
    print(len(overlapping_IDs))

    if len(overlapping_IDs) < 2:
        raise Exception("We need at least 2 overlapping annotations to calculate IAA.")

    # Prepare the data matrix
    data_matrix = []

    for id in overlapping_IDs:
        group = annotations[annotations['ID'] == id]
        row = []
        
        for category in categories:
            # If any annotator did not rate in this category for this item, append nan
            ratings = group[category].tolist() + [np.nan] * (max_ratings - group[category].count())
            row.extend(ratings)

        # Ensure each row length equals the total number of possible ratings per item
        row = row[:total_ratings_per_item]
        data_matrix.append(row)
        # print(row)

    # matrix where each row is an item and each column is an annotator's rating.
    data_matrix = np.array(data_matrix, dtype=float)
    # print(data_matrix)
    # print(data_matrix.shape)
    alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ordinal')
    
    return alpha 


def krippendorff_alpha_per_category(annotations, categories):
    num_annotators = 5  # Assuming 5 annotators
    
    overlapping_IDs = annotations[annotations.duplicated(subset="ID", keep=False)]["ID"].unique()

    if len(overlapping_IDs) < 2:
        raise Exception("We need at least 2 overlapping annotations to calculate IAA.")

    category_alphas = {}

    for category in categories:
        data_matrix = []

        for id in overlapping_IDs:
            group = annotations[annotations['ID'] == id]
            ratings = group[category].tolist()

            # If any annotator did not rate this item, append nan to make the length consistent
            ratings += [np.nan] * (num_annotators - len(ratings))
            data_matrix.append(ratings)

        data_matrix = np.array(data_matrix, dtype=float)
        # print(data_matrix)
        # print(data_matrix.shape)
        alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ordinal')
        category_alphas[category] = alpha

    return category_alphas