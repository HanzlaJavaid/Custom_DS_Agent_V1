import pickle
import pandas as pd
import json
# 1. Open the file containing the pickled model
import sys
import os
from jinja2 import Template
import json
import numpy as np

# Assuming your script is in the tools directory, add the parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import LLM  # Your original import statement
from prompts.prompts import feature_extractor,feature_list

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


print(get_embedding('fancy red carr'))

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


def feature_extractor_fn(product_detail):
    # initialize the llm
    llm = LLM()
    llm.initialise_llm(api_key='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')
    output = None
    while not isinstance(output, list):
        try:
            prompt = Template(feature_extractor)
            x = prompt.render(feature_list=feature_list, product_description=product_detail)
            response = llm.step(x)
            output = parse_list(response)
        except:
            pass
    return output
         

# def inference()

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

    # features = feature_extractor_fn(product_detail=product_detail)
    # for i,feature in enumerate(features):
    embedding = get_embedding(product_description=product_detail)
    for i, row in df.iterrows():
        emb =  embedding + [row['Year'], row['Month'], row['Day'], 1, 1]
        df.at[i, 'embedding'] = emb
    
        # Flatten the embedding array
    embedding_columns = [f'dimension_{i}' for i in range(1,773+1)]
    print(embedding_columns)
    for i, col in enumerate(embedding_columns):
        df[col] = df['embedding'].apply(lambda x: x[i])

    embedding_cols =  ['item_sold'] + embedding_columns
        # Split data into train and test sets
    X = df[embedding_cols]  # Feature

    with open('/home/khudi/Desktop/my_own_agent/gpt5.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    

    result = loaded_model.predict(X)
    # print(result, "RESULTSS")
    prediction_dict[f'features_{i}'] = product_detail
    prediction_dict[f'forecast_{i}'] = np.sum(result)

    print(prediction_dict)
    return prediction_dict

    
new_product_forecasting_inference_api("I can see the shirt in grey color with no pockets", start_date='2024-01-01', end_date='2024-03-31')

# with open('model.pkl', 'rb') as f:
#     # 2. Load the model
#     loaded_model = pickle.load(f)

# # 3. Use the loaded model as needed
# # For example:
# result = loaded_model.predict(data)

# # 4. Close the file
# f.close()