import pickle
import pandas as pd
import json
# 1. Open the file containing the pickled model
import sys
import os
from jinja2 import Template
import json
import numpy as np
import random
import itertools
import time
import warnings

# Assuming you have the code that generates the warning here
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Assuming your script is in the tools directory, add the parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import LLM  # Your original import statement
from prompts.prompts import feature_extractor,feature_list, shap_prompt

def create_prediction_dataframe(start_date, end_date):
    # Generate a range of dates from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create an empty DataFrame with the date range as index
    empty_df = pd.DataFrame({'date': date_range})
    
    return empty_df


def get_embedding(product_description):
    import requests
    url = "https://api.together.xyz/v1/embeddings"

    payload = {
        "model": "bert-base-uncased",
        "input": product_description
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer 84ee7f7050742d032694d0f70e9b628e4293246d739df04b02721ed68502fb76"
    }

    response = requests.post(url, json=payload, headers=headers)
    json_data = json.loads(response.text)
    
    return json_data['data'][0]['embedding']



def parse_list(task_str):
        # Extracting the substring containing the Python list
        try:
            start_index = task_str.find("[")
            end_index = task_str.find("]") + 1
            python_list_string = task_str[start_index:end_index]

            # Cleaning up the string and removing unnecessary characters
            python_list_string = python_list_string.replace("assistant:", "").strip()

            # Convert the string representation of the list into a Python list
            python_list = eval(python_list_string)

            # Print the extracted Python list
            return python_list
        except Exception as e:
            return f"Oops, there is some error in parsing the task list. Are you sure you output the task in right format. Error: {e}.\nTry again. Best of luck!"


def json_parser(features_str):
        try: 
            features = json.loads(features_str)
            return features
        except Exception as e:
            return f"Oops, there seems to be an error while parsing the json. Are you sure you reponsd with valid json. Try again!\nYour output: {features_str}\nError: {e}"


def step(prompt):
    llm = LLM()
    llm.initialise_llm(api_key='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')
    return llm.step(prompt)

def feature_extractor_fn(product_detail):
    # initialize the llm
    
    output = None
    while not isinstance(output, dict):
        try:
            prompt = Template(feature_extractor)
            x = prompt.render(feature_list=feature_list, product_description=product_detail)
            response = step(x)
            output = json_parser(response)
        except:
            pass
    
    feature_list_dict = {}
    if output['product_type'] == 'pants':
         df = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_pants_dataset.csv')

    else:
         df = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_shirts_dataset.csv')

    for key, value in output.items():
              if key != 'product_type':
                   feature_list_dict[key] = []
                   feature_list_dict[key].append(f"{key}: {value}\n")
                   feature_list_dict[key].append(f"{key}: {random.choice(df[key].unique())}\n")

    result = list(itertools.product(*feature_list_dict.values()))
    combined_result = [''.join(items) for items in result]

    return combined_result


def shap(output_dict):
    prompt = Template(shap_prompt)
    x = prompt.render(results=output_dict)
    return step(x)
     
         
# print(feature_extractor_fn("I can see a shirt with tea color and no pockets and crew neck style"))


def new_product_forecasting_inference_api(product_detail, start_date, end_date):
    prediction_dict = {}

    df = create_prediction_dataframe(start_date=start_date, end_date=end_date)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Weekday'] = df['date'].dt.weekday
    df['embedding'] = None
    df['item_sold'] = 1


    with open('/home/khudi/Desktop/my_own_agent/gpt5.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    features = feature_extractor_fn(product_detail=product_detail)
    print(features)
    for i,feature in enumerate(features):
        time.sleep(1)
        embedding = get_embedding(product_description=feature)
        for i, row in df.iterrows():
            emb =  embedding + [row['Year'], row['Month'], row['Day'], 1, 1]
            df.at[i, 'embedding'] = emb
        
            # Flatten the embedding array
        embedding_columns = [f'dimension_{i}' for i in range(1,773+1)]

        for i, col in enumerate(embedding_columns):
            df[col] = df['embedding'].apply(lambda x: x[i])

        embedding_cols =  ['item_sold'] + embedding_columns
            # Split data into train and test sets
        X = df[embedding_cols]  # Feature

        

        result = loaded_model.predict(X)
        prediction_dict[feature] = np.sum(result)

    interpretaion = shap(prediction_dict)
    print(interpretaion)
    return interpretaion

    
new_product_forecasting_inference_api("I can see a shirt with tea color and no pockets and crew neck style", start_date='2024-01-01', end_date='2024-03-31')

# with open('model.pkl', 'rb') as f:
#     # 2. Load the model
#     loaded_model = pickle.load(f)

# # 3. Use the loaded model as needed
# # For example:
# result = loaded_model.predict(data)

# # 4. Close the file
# f.close()