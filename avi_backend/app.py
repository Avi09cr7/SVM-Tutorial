import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
# from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# inputs
training_data = 'data/cardio_train1.csv'
include = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
dependent_variable = include[-1]

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
clf = None


@app.route('/train-svm', methods=['POST'])
@cross_origin()
def train_svm():
    df = pd.read_csv('./data/cardio_train1.csv').iloc[:100,:]
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    json_ = request.get_json()
    json_d = dict(json_)
    json_d["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = float(json_d["test_size"]))
    if str(json_d["kernel"]) == 'linear':
        try:
            kernel = 'linear'
            c = float(json_d["c"])
            probability = True
            start = time.time()
            time.sleep(5)
            svc = SVC(kernel=kernel, C=c, probability=True)
            model = svc.fit(X_train, y_train)
            end = time.time()
            score = model.score(X_test, y_test)
            return {
                'model': 'SVM Classifier',
                'params':json_d,
                'score': "{} %".format(score *100) ,
                'time':"{} second(s)".format(int(end-start))
            }
        except Exception as e:
            print("error", e)
            return "Bad Request", 400

    elif str(json_d["kernel"]) == 'poly':
        try:
            kernel = 'poly'
            c = float(json_d["c"])
            degree = int(json_d["degree"])
            gamma = str(json_d["gamma"])
            probability = True
            start = time.time()
            svc = SVC(
                kernel=kernel, 
                C=c, 
                probability=probability, 
                degree=degree, 
                gamma = gamma
                )
            model = svc.fit(X_train, y_train)
            end = time.time()
            score = model.score(X_test, y_test)
            return {
            'model': 'SVM Classifier',
            'params':json_d,
            'score': "{} %".format(score *100) ,
            'time':"{} second(s)".format(int(end-start))
        }
        except Exception as e:
            print("error", e)
            return "Bad Request", 400

    elif str(json_d["kernel"]) == 'rbf':
        kernel = 'rbf'
        c = float(json_d["c"])
        gamma = str(json_d["gamma"])
        probability = True
        start = time.time()
        svc = SVC(
                kernel=kernel, 
                C=c, 
                probability=probability,
                gamma = gamma
                )
        model = svc.fit(X_train, y_train)
        end = time.time()
        score = model.score(X_test, y_test)
        return {
            'model': 'SVM Classifier',
            'params':json_d,
            'score': "{} %".format(score *100) ,
            'time':"{} second(s)".format(int(end-start))
        }

    elif str(json_d["kernel"]) == 'sigmoid':
        kernel = 'sigmoid'
        c = float(json_d["c"])
        gamma = str(json_d["gamma"])
        probability = True
        start = time.time()
        svc = SVC(
                kernel=kernel, 
                C=c, 
                probability=probability,
                gamma = gamma
                )
        model = svc.fit(X_train, y_train)
        end = time.time()
        score = model.score(X_test, y_test)
        return {
            'model': 'SVM Classifier',
            'params':json_d,
            'score': "{} %".format(score *100) ,
            'time':"{} second(s)".format(int(end-start))
        }


    

    pass

@app.route('/test', methods=['GET'])
@cross_origin()
def tests():
    return {
        'test':'Just kidding'
    }

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'


# @app.route('/train', methods=['GET'])
# def train():
#     # using random forest as an example
#     # can do the training separately and just update the pickles
#     from sklearn.ensemble import RandomForestClassifier as rf

#     df = pd.read_csv(training_data)
#     df_ = df[include]

#     categoricals = []  # going to one-hot encode categorical variables

#     for col, col_type in df_.dtypes.items():
#         if col_type == 'O':
#             categoricals.append(col)
#         else:
#             df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

#     # get_dummies effectively creates one-hot encoded variables
#     df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

#     x = df_ohe[df_ohe.columns.difference([dependent_variable])]
#     y = df_ohe[dependent_variable]

#     # capture a list of columns that will be used for prediction
#     global model_columns
#     model_columns = list(x.columns)
#     joblib.dump(model_columns, model_columns_file_name)

#     global clf
#     clf = rf()
#     start = time.time()
#     clf.fit(x, y)

#     joblib.dump(clf, model_file_name)

#     message1 = 'Trained in %.5f seconds' % (time.time() - start)
#     message2 = 'Model training score: %s' % clf.score(x, y)
#     return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
#     return return_message


# @app.route('/wipe', methods=['GET'])
# def wipe():
#     try:
#         shutil.rmtree('model')
#         os.makedirs(model_directory)
#         return 'Model wiped'

#     except Exception as e:
#         print(str(e))
#         return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    # try:
    #     clf = joblib.load(model_file_name)
    #     print('model loaded')
    #     model_columns = joblib.load(model_columns_file_name)
    #     print('model columns loaded')

    # except Exception as e:
    #     print('No model here')
    #     print('Train first')
    #     print(str(e))
    #     clf = None

    # app.run(host='0.0.0.0', port=port)
    app.run()
