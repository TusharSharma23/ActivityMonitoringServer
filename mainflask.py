import os.path

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
import pickle
from sklearn.cluster import KMeans
from collections import Counter

app = Flask(__name__)
script_dir = os.path.dirname(__file__)
data_dir = script_dir + "/Data"
model_dir = script_dir + '/ModelData'

@app.route('/')
def hello_world():
    return 'Hello'

# @ signifies a decorator- way to wrap a fun and modify it
@app.route('/login', methods=['POST'])
def login():
    input_data = request.get_json()
    print("Input data: " + str(input_data))
    output_data = dict()
    try:
        device_id = input_data['DeviceId']
    except KeyError:
        output_data['status'] = "Failure. Data not found."
        print("Invalid Key.")
        return jsonify(output_data)
    dir_list = os.listdir(script_dir)
    if dir_list.count("Data") == 0:
        print("Data directory missing")
        os.mkdir(data_dir)
        print("Created directory: " + data_dir)
    dir_list = os.listdir(data_dir)
    if dir_list.count("LoginData.csv") == 0:
        print("Login file missing")
        user_id = "User0001"
        data = pd.Series({"DeviceId": device_id,
                          "UserId": user_id})
        data = pd.DataFrame([data])
        data.to_csv(data_dir + '/LoginData.csv', index=False)
        print("Created file: " + data_dir + "/LoginData.csv")
        print("Added user: " + user_id)
        new_user = True
    else:
        data = pd.read_csv(data_dir + '/LoginData.csv')
        print("Opened Login file.")
        device_data = data[data["DeviceId"] == device_id]
        if len(device_data) == 0:
            print("Device not mentioned in list.")
            data_len = len(data)
            tmp = str(int(data["UserId"][data_len - 1][-len(str(data_len)):]) + 1)
            user_id = "User" + "0" * (4 - len(tmp)) + tmp
            data = data.append(pd.Series({"DeviceId": device_id,
                                          "UserId": user_id}), ignore_index=True)
            data.to_csv(data_dir + '/LoginData.csv', index=False)
            print("Device added to list.")
            print("Added user: " + user_id)
            new_user = True
        else:
            print("User already exists.")
            user_id = device_data['UserId'][device_data.index[0]]
            print("UserId: " + user_id)
            new_user = False
    output_data["status"] = "success"
    output_data["user_id"] = user_id
    output_data["new_user"] = new_user
    print("Response data: " + str(output_data))
    return jsonify(output_data)


@app.route('/lastDayDepressionState', methods=['POST'])
def status():
    input_data = request.get_json()
    print("Input data: " + str(input_data))
    output_data = dict()
    try:
        user_id = input_data['UserId']
        if not check_valid_user(user_id):
            print("User Invalid. User not mentioned in login data.")
            output_data['status'] = "Failure. Invalid User."
            return jsonify(output_data)
        status_file = pd.read_csv(data_dir + '/StatusData.csv')
    except KeyError:
        print("Invalid Key.")
        output_data['status'] = "Failure. Data not found"
        return jsonify(output_data)
    except FileNotFoundError:
        print("Status file does not exist.")
        output_data['status'] = "Error. No records found."
        return jsonify(output_data)
    state_list = status_file[status_file["UserId"] == user_id]
    if len(state_list) == 0:
        print("Status for user " + user_id + " missing")
        state = "Error. No records found."
    else:
        state = state_list['State'][state_list.index[0]]
        print("User fetched depression state: " + state)
    output_data["status"] = state
    print("Response Data: " + str(output_data))
    return jsonify(output_data)


@app.route('/lastWeekDepressionState', methods=['POST'])
def patient_status():
    input_data = request.get_json()
    print("Input data: " + str(input_data))
    output_data = dict()
    try:
        user_id = input_data['UserId']
        if not check_valid_user(user_id):
            print("User Invalid. User not mentioned in login data.")
            output_data['status'] = "Failure. Invalid User."
            return jsonify(output_data)
    except KeyError:
        print("Invalid Key.")
        output_data['status'] = "Failure. Data not found"
        return jsonify(output_data)
    dir_list = os.listdir(data_dir)
    if dir_list.count(user_id) == 0:
        print("No records found for this user.")
        output_data['status'] = "Error. No records found for this user"
    else:
        dir_list = os.listdir(data_dir + "/" + user_id)
        if len(dir_list) <= 5:
            print("Insufficient data. Total files: " + str(len(dir_list)))
            output_data['status'] = 'Error. Insufficient data.'
        else:
            print("Data sufficient. Total files: " + str(len(dir_list)))
            working_dir = data_dir + "/" + user_id
            files = os.listdir(working_dir)
            data = pd.DataFrame()
            print("Gathering Data.")
            for file in files:
                df = pd.read_csv(working_dir + "/" + file)
                data = data.append(df, ignore_index=True)
            del data['timestamp']
            print("Gathered data successfully.")
            data = data.to_numpy()
            output_data['status'] = get_patient_state(data, 'patientwise.sav')
    print("Response data: " + str(output_data))
    return jsonify(output_data)


@app.route('/uploadTodaysActivity', methods=['POST'])
def upload_activity():
    user_id = request.form['UserId']
    file = request.files.get('activity_file')
    print("Got userId: " + user_id + " and file: " + str(file.filename))
    output_data = dict()
    if not check_valid_user(user_id):
        print("User Invalid. User not mentioned in login data.")
        output_data['status'] = "Failure. Invalid User."
        return jsonify(output_data)
    df = pd.read_csv(file)
    create_user_file(user_id, df, file.filename)
    del df['timestamp']
    data = df.to_numpy()
    print("Clustering Data.")
    k_mean_model = KMeans(n_clusters=4, max_iter=500)
    k_mean_model.fit(data)
    print("Data Clustered.")
    labels = Counter(k_mean_model.labels_)
    extra_params = {'No_to_mild_Activity': labels[0],
                    'high_activity': labels[3]}
    state = get_patient_state(data, 'daywise.sav', extra_params)
    output_data['status'] = state
    update_status_file(user_id, state)
    print("Response Data: " + str(output_data))
    return jsonify(output_data)


def get_patient_state(data, model_name, extra_params=None):
    print("Collecting 14 parameters.")
    mean = np.mean(data)
    std = np.std(data)
    vari = np.var(data)
    trimmed_mean = stats.trim_mean(data, 0.20)[0]
    coff_of_variation = std / mean
    inv_coff_of_variation = mean / std
    kurtosys = stats.kurtosis(data)[0]
    skewness = stats.skew(data)[0]
    quantile1 = np.quantile(data, .01)
    quantile5 = np.quantile(data, .05)
    quantile25 = np.quantile(data, .25)
    quantile75 = np.quantile(data, .75)
    quantile95 = np.quantile(data, .95)
    quantile99 = np.quantile(data, .99)

    print("Collected 14 parameters.")

    data = {'Mean': mean,
            'Standard deviation': std,
            'Variance': vari,
            'Trimmed mean': trimmed_mean,
            'Coefficient of variation': coff_of_variation,
            'Inverse coefficient of variation': inv_coff_of_variation,
            'Kurtosis': kurtosys,
            'Skewness': skewness,
            'Quantile 1%': quantile1,
            'Quantile 5%': quantile5,
            'Quantile 25%': quantile25,
            'Quantile 75%': quantile75,
            'Quantile 95%': quantile95,
            'Quantile 99%': quantile99}
    if extra_params is not None:
        print("Adding low and high activity in data.")
        data.update(extra_params)
    return fit_model(pd.DataFrame([data]), model_name)


def fit_model(data, model_name):
    print("Fitting model to data.")
    print("Opening file: " + model_dir + '/' + model_name)
    file = open(model_dir + '/' + model_name, 'rb')
    print("Loading model.")
    model = pickle.load(file)
    print("Model loaded successfully.")
    test_data = []
    test_data.extend(data.to_numpy())
    print("Predicting value.")
    prediction = model.predict(test_data)
    file.close()
    if prediction[0][0] == 1:
        print("Predicted value: Depressed")
        return "Depressed"
    else:
        print("Predicted value: Not Depressed")
        return "Not Depressed"


def check_valid_user(user_id):
    print("Checking user validity.")
    dir_list = os.listdir(script_dir)
    if dir_list.count("Data") == 0:
        print("Data directory missing.")
        return False
    dir_list = os.listdir(data_dir)
    if dir_list.count("LoginData.csv") == 0:
        print("Login file missing.")
        return False
    else:
        data = pd.read_csv(data_dir + '/LoginData.csv')
        device_data = data[data["UserId"] == user_id]
        if len(device_data) == 0:
            print("Specified user not found.")
            return False
        else:
            print("Valid user.")
            return True


def create_user_file(user_id, data_frame, file_name):
    print("Storing user activity file.")
    dir_list = os.listdir(script_dir)
    if dir_list.count("Data") == 0:
        print("Data directory missing.")
        os.mkdir(data_dir)
        print("Data directory Created.")
    dir_list = os.listdir(data_dir)
    if dir_list.count(user_id) == 0:
        print("User directory missing.")
        os.mkdir(data_dir + "/" + user_id)
        print("User directory created.")
        path = os.path.join(data_dir + "/" + user_id, file_name)
        data_frame.to_csv(path, index=False)
        print("User activity file created.")
    else:
        dir_list = os.listdir(data_dir + "/" + user_id)
        if dir_list.count(file_name) > 0:
            print("File already exists.")
        else:
            path = os.path.join(data_dir + "/" + user_id, file_name)
            if len(dir_list) >= 15:
                print("Found more than 15 files.")
                sorted_file_name = sorted(dir_list)
                print("Removing file: " + sorted_file_name[0])
                os.remove(os.path.join(data_dir + "/" + user_id, sorted_file_name[0]))
                data_frame.to_csv(path, index=False)
                print("User file created.")
            else:
                data_frame.to_csv(path, index=False)
                print("User file created.")


def update_status_file(user_id, state):
    print("Storing predicted daily depression state.")
    dir_list = os.listdir(script_dir)
    if dir_list.count("Data") == 0:
        print("Data directory missing.")
        os.mkdir(data_dir)
        print("Data directory created.")
    dir_list = os.listdir(data_dir)
    if dir_list.count("StatusData.csv") == 0:
        print("Status file missing.")
        data = {'UserId': user_id,
                'State': state}
        data = pd.DataFrame([data])
        data.to_csv(data_dir + "/StatusData.csv", index=False)
        print("Status file created.")
    else:
        data = pd.read_csv(data_dir + "/StatusData.csv")
        user_data = data[data['UserId'] == user_id]
        if len(user_data) == 0:
            print("User state missing.")
            data = data.append(pd.Series({"UserId": user_id,
                                          "State": state}), ignore_index=True)
            data.to_csv(data_dir + '/StatusData.csv', index=False)
            print("User state added.")
        else:
            data.at[user_data.index[0], 'State'] = state
            data.to_csv(data_dir + '/StatusData.csv', index=False)
            print("User state updated.")


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
