import numpy as np
import pandas as pd
import math
import json
import time
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from flask import Flask, render_template
from flask_ngrok import run_with_ngrok


app = Flask(__name__, template_folder='template')

home_df=pd.read_csv('Review_Data.csv')
home_df.drop('Unnamed: 0',axis=1,inplace=True)
print(home_df.shape)

counts=home_df.userId.value_counts()
home_df_final=home_df[home_df.userId.isin(counts[counts>=12].index)]
print(home_df_final.shape)

train_data, test_data = train_test_split(home_df_final, test_size = 0.3, random_state=0)
print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)

train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 10 recommendations 
popularity_recommendations = train_data_sort.head(10) 
popularity_recommendations

output = {
        "id": "",
        "name": "",
        "description": False,
        "sub_category": [
            {
                "id": "",
                "name": ""
            }
        ],
        "parent_category": [
            {
                "id": "",
                "name": ""
            }
        ],
        "original_price": "",
        "currency_id": False,
        "price_for_user": "",
        "currency_name": "",
        "packagings": [
            {
                "Packagin_id": "",
                "name": "",
                "qty": ""
            },
            {
                "Packagin_id": "",
                "name": "",
                "qty": ""
            }
        ]
    }

myclient = pymongo.MongoClient('mongodb+srv://m001-student:m001root@sandbox.3jrnp.mongodb.net/project')
mydb = myclient['project']
mycol = mydb['productdata']

@app.route('/<user_id>')

def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    # user_recommendations = user_recommendations[cols] 
    recommended_products_id = list(user_recommendations['productId'])
    print(recommended_products_id)
    details = []
    for x in recommended_products_id:
        new_record = output.copy()
        test = mycol.find_one({'Unique Id':x})
        new_record['id'] =  test['Unique Id']
        new_record['name'] = test['Name']  
        new_record['original_price'] = test['Mrp']
        new_record['price_for_user'] = test['Selling_Price']
        details.append(new_record)
    return json.dumps(details)


if __name__ == '__main__':
    app.run(debug=True)

# Inference

# Since, it is a Popularity recommender model, so, all the new users are given the same recommendations.
# Here, we predict the products based on the popularity. It is a non-personalized recommender system.