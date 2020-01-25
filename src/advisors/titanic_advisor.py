from arango import ArangoClient
import pandas as pd
from pymongo import MongoClient
import json
import numpy as np
from datetime import datetime

with open('ml/templates/titanic_advisor_template.json') as data_file:
    arguments = json.load(data_file)


con = MongoClient(arguments['inputDS']["ip"], arguments['inputDS']["port"])
mongo_db = con['titanic']

cursor = mongo_db['train_titanic'].find()

titanic_train = pd.DataFrame(list(cursor))
train = titanic_train.drop(['_id'],axis=1)

cursor = mongo_db['test_titanic'].find()

titanic_test = pd.DataFrame(list(cursor))
test = titanic_test.drop(['_id'],axis=1)

#
# train['Survived'].value_counts(normalize=True)
#
# train['Survived'].groupby(train['Pclass']).mean()
#
# train['Name_Len'] = train['Name'].apply(lambda x: len(x))
# train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()
#
# pd.qcut(train['Name_Len'],5).value_counts()
#
# train['Sex'].value_counts(normalize=True)
#
# train['Survived'].groupby(train['Sex']).mean()
#
# train['Survived'].groupby(train['Age'].isnull()).mean()
#
# train['Survived'].groupby(pd.qcut(train['Age'],5)).mean()
#
# train['Cabin_num'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
# train['Cabin_num'].replace('an', np.NaN, inplace = True)
# train['Cabin_num'] = train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test


def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test


def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test


def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test


def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test


def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test


test['Fare'].fillna(train['Fare'].mean(), inplace = True)



def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test


def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test



train = train
test = test
train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = drop(train, test)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini',
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print("%.4f" % rf.oob_score_)


pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']),
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])),
          axis = 1).sort_values(by='importance', ascending = False)[:20]


predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
test = test
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

# index = range(1, predictions.shape[0] + 1 ,1)
# predictions['_id'] =index
predictions = predictions.to_dict('records')


arango_host = 'http://' + arguments['outputDS']["ip"] + ':' + arguments['outputDS']["port"]

client = ArangoClient(hosts=arango_host)

arango_db = client.db(arguments['outputDS']["database"], username=arguments['outputDS']["username"], password=arguments['outputDS']["password"])

arango_db['predictions_titanic'].insert_many(predictions)

now = datetime. now()
dt_string = now. strftime("%d/%m/%Y %H:%M:%S")
output = {
		"Workflow" : arguments['name'],
		"execution_time" : dt_string,
        "status" : "Sucess"
	}

arango_db['output_status'].insert(output)






