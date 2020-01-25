import pandas as pd
from pymongo import MongoClient
import json
from arango import ArangoClient
from datetime import datetime

with open('src/templates/titanic_miner_template.json') as data_file:
    arguments = json.load(data_file)

df_train = pd.read_csv('src/data/titanic_train.csv',encoding="latin-1")


con = MongoClient(arguments['outputDS']["ip"], arguments['outputDS']["port"])
mongo_db = con['titanic']

train_data_to_upload = df_train.to_dict('records')

mongo_db['train_titanic'].insert_many(train_data_to_upload)

df_train = pd.read_csv('src/data/titanic_test.csv',encoding="latin-1")

test_data_to_upload = df_train.to_dict('records')

mongo_db = con['titanic']

mongo_db['test_titanic'].insert_many(test_data_to_upload)


arango_host = 'http://' + arguments['metricDS']["ip"] + ':' + arguments['metricDS']["port"]

client = ArangoClient(hosts=arango_host)

db = client.db(arguments['metricDS']["database"], username=arguments['metricDS']["username"], password=arguments['metricDS']["password"])
# Create a new collection named "students" if it does not exist.
# This returns an API wrapper for "students" collection.
arangodb = db.collection('output_status')

now = datetime. now()
dt_string = now. strftime("%d/%m/%Y %H:%M:%S")
output = {
		"Workflow" : arguments['name'],
		"execution_time" : dt_string,
        "status" : "Sucess"
	}

arangodb.insert(output)