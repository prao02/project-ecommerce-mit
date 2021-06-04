import numpy as np
import pandas as pd
import math
import json
import time
from copy import deepcopy
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from flask import Flask, render_template, request,jsonify
from flask_ngrok import run_with_ngrok

app = Flask(__name__)

home_df=pd.read_csv('Review_Data.csv')
home_df.drop('Unnamed: 0',axis=1,inplace=True)
print(home_df.shape)

counts=home_df.userId.value_counts()
home_df_final=home_df[home_df.userId.isin(counts[counts>=12].index)]
print(home_df_final.shape)

train_data, test_data = train_test_split(home_df_final, test_size = 0.3, random_state=0)
print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)

home_df_CF = pd.concat([train_data, test_data]).reset_index()

# Matrix with row per 'user' and column per 'item' 
pivot_df = home_df_CF.pivot(index = 'userId', columns ='productId', values = 'ratings').fillna(0)
print('Shape of the pivot table: ', pivot_df.shape)
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.set_index(['user_index'], inplace=True)

# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df, k = 10)
# Construct diagonal array in SVD
sigma = np.diag(sigma)
# print('Diagonal matrix: \n',sigma)

#Predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# Convert predicted ratings to dataframe
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)

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

output2 = {
    "categoryId": "",
      "categoryName": "",
      "subcategories": [
        {
          "id": "1",
          "name": "",
        },
        {
          "id": "2",
          "name": "",
        }
      ]
}

myclient = pymongo.MongoClient('mongodb+srv://m001-student:m001root@sandbox.3jrnp.mongodb.net/project')
mydb = myclient['project']
mycol = mydb['productdata']

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

@app.route('/<int:user_id>')
def recommend_items(user_id):
    # user_id = request.args.get('user_id', None)
    # debug
    print(f"got userid {user_id}")
    response = {}
    # checking if user entered a userid:
    if not user_id:
        response["ERROR"] = "no userid found, please send a userid."
        return jsonify(response)
    # index starts at 0
    else:
        user_idx = int(user_id)-1
        # Get and sort the user's ratings
        sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
        #sorted_user_ratings
        sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
        #sorted_user_predictions
        temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
        temp.index.name = 'Recommended Items'
        temp.columns = ['user_ratings', 'user_predictions']
        temp = temp.loc[temp.user_ratings == 0]
        temp = temp.sort_values('user_predictions', ascending=False)
        print('\nBelow are the recommended items for user(user_id = {}):\n'.format(user_id))
        r_products = temp.head(5).reset_index()
        r_products_id = r_products['Recommended Items']
        print(r_products_id)
        details = []
        for x in r_products_id:
            new_record = deepcopy(output)
            test = mycol.find_one({'Unique Id': x})
            new_record['id'] = test['Product Id']
            new_record['name'] = test['Name']
            new_record['sub_category'][0]['id'] = test['Category_Id']
            new_record['sub_category'][0]['name'] = test['Category']
            new_record['parent_category'][0]['id'] = test['Category_Id']
            new_record['parent_category'][0]['name'] = test['Category']
            new_record['original_price'] = test['Mrp']
            new_record['price_for_user'] = test['Selling_Price']
            new_record['packagings'][0]['qty'] = test['Qty']
            details.append(new_record)
        # print(details)
        return json.dumps(details)
@app.route('/category')
def category():
    user_id = request.args.get('user_id', None)
    de = json.loads(recommend_items(user_id))
    details2 =[]
    for x in de:
        print(x)
        new_record2 = output2.copy()
        new_record2["categoryId"] = x["parent_category"][0]['id']
        new_record2["categoryName"] = x["parent_category"][0]['name']
        details2.append(new_record2)
    return json.dumps(details2)
    
if __name__ == '__main__':
    app.run(debug=True)
