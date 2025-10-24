import pandas as pd
import numpy as np
from scipy.stats import norm, randint
from sklearn.preprocessing import MinMaxScaler

datacount = 500000


def generate():
    """
    Generates synthetic dataset containing biological indicators, survey data,
    disease prevalence, and associated ages for a specified number of samples.

    Returns:
        X (numpy.ndarray): Independent variables including biological indicators,
                           survey responses, and current ages for each disease.
        y (numpy.ndarray): Dependent variables representing label ages for each disease.
        e (numpy.ndarray): Disease status indicators (0 or 1) for each disease.
    """
    np.random.seed(0)
    num_samples = datacount
    num_diseases = 60  # Number of diseases

    biosize = 65
    surveysize = 10
    biocount = 100
    surveycount = 15
    # 1. Generate biological indicator data
    # Assume each biological indicator follows a normal distribution, parameters set based on common biological indicators
    bio_data = {}
    for i in range(biocount):
        loc = 0
        scale = 1
        # Our process uses standardized data, if you want to use raw data, you can set loc and scale to other numbers, this might affect the performance of the model
        bio_data[f"bio_{i}"] = np.random.normal(loc, scale, size=num_samples)

    # Create DataFrame
    df = pd.DataFrame(bio_data)
    survey_data = {}
    # 2. Generate survey data
    # Assume survey questions are positive integers, such as number of cigarettes smoked, frequency of drinking, etc.
    for i in range(surveycount):
        lower = 0
        upper = 1
        # For demonstration purposes, we set the range to {0, 1} for simplicity
        survey_data[f"sur_{i}"] = randint.rvs(lower, upper, size=num_samples)

    df_survey = pd.DataFrame(survey_data)
    df = pd.concat([df, df_survey], axis=1)

    # 3. Generate death age
    # Assume death age follows a normal distribution, mean 80 years, standard deviation 10 years
    death_age = np.random.normal(loc=80, scale=10, size=num_samples).astype(int)
    death_age = np.clip(death_age, a_min=20, a_max=None)  # Death age at least 20 years
    df["death_age"] = death_age

    # 4. Construct disease prevalence (0-1) data and corresponding age labels
    def generate_disease_data(df, bio_features, survey_features, disease_id):
        selected_features = list(bio_features) + list(survey_features)

        # Normalize selected features for computation
        scaler = MinMaxScaler()
        df_selected = df[selected_features]
        df_scaled = scaler.fit_transform(df_selected)

        # Define weights (here simply randomly assign weights)
        n_features = df_scaled.shape[1]

        main = 0.3  # 30% of variables will use double the range that generates significant impact on disease

        # Calculate the number of variables to sample based on the proportion
        num_main_features = int(n_features * main)

        # Randomly select indices of variables controlled by 'main'
        main_indices = np.random.choice(n_features, num_main_features, replace=False)

        # Initialize weight array
        lower = -1.3
        upper = 1
        weights = np.random.uniform(lower, upper, size=n_features)
        # Numbers are set to achieve a overall 5-10% disease prevalence

        weights[main_indices] = np.random.uniform(
            lower * 2, upper * 2, size=num_main_features
        )

        linear_combination = np.dot(df_scaled, weights)
        probabilities = 1 / (1 + np.exp(-linear_combination))  # Sigmoid function

        # Generate 0-1 disease data based on probabilities
        disease = np.random.binomial(1, probabilities)
        df[f"disease_{disease_id}"] = disease

        # Generate label age
        # Assume current age is between 20 and 70 years
        current_age = np.random.randint(20, 70, size=num_samples)
        df[f"current_age_{disease_id}"] = current_age

        # If diseased, disease age = current age + random time (1 to 20 years), and less than death age
        disease_age = current_age + np.random.randint(1, 20, size=num_samples)
        disease_age = np.where(
            disease_age >= df["death_age"], df["death_age"] - 1, disease_age
        )

        # Label age: if diseased, then disease age; otherwise, death age
        label_age = np.where(
            df[f"disease_{disease_id}"] == 1, disease_age, df["death_age"]
        )
        df[f"label_age_disease_{disease_id}"] = label_age

        return df

    # 4.1 For each disease, select features and generate data separately
    for disease_id in range(1, num_diseases + 1):
        selected_bio_features = np.random.choice(
            list(bio_data.keys()), size=biosize, replace=False
        )
        selected_survey_features = np.random.choice(
            list(survey_data.keys()), size=surveysize, replace=False
        )
        df = generate_disease_data(
            df, selected_bio_features, selected_survey_features, disease_id
        )

    # 5. Select final features and labels
    # Including all biological indicators, survey data, current age for each disease, disease status, and label age
    final_features = list(bio_data.keys()) + list(survey_data.keys())

    # Add current age, disease status, and label age for each disease
    for disease_id in range(1, num_diseases + 1):
        final_features += [
            f"current_age_{disease_id}",
            f"disease_{disease_id}",
            f"label_age_disease_{disease_id}",
        ]

    # Final dataset includes all features and labels
    df_final = df[final_features + ["death_age"]]

    # 6. Save independent variables, diseases, and disease ages as three numpy arrays X, y, e
    # Independent variables X include all biological indicators, survey data, and current age for each disease
    feature_columns = list(bio_data.keys()) + list(
        survey_data.keys()
    )  # + [f'current_age_{d}' for d in range(1, num_diseases + 1)]
    X = df_final[feature_columns].to_numpy()

    # Disease y includes all disease statuses
    e = df_final[[f"disease_{d}" for d in range(1, num_diseases + 1)]].to_numpy()

    # Disease age e includes all label ages for diseases
    y = df_final[
        [f"label_age_disease_{d}" for d in range(1, num_diseases + 1)]
    ].to_numpy()

    return X, y, e
