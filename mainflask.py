import os.path
import uuid

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
import pickle
from sklearn.cluster import KMeans
from collections import Counter

from werkzeug.utils import secure_filename

app = Flask(__name__)
script_dir = os.path.dirname(__file__)
data_dir = script_dir + "/Data"
model_dir = script_dir + '/ModelData'


# @ signifies a decorator- way to wrap a fun and modify it
@app.route('/login', methods=['POST'])
def login():
    input_data = request.get_json()
    output_data = dict()
    try:
        device_id = input_data['DeviceId']
    except KeyError:
        output_data['status'] = "Failure. Data not found."
        return jsonify(output_data)
    dir_list = os.listdir()
    if dir_list.count("Data") == 0:
        os.mkdir("Data")
    dir_list = os.listdir("Data")
    if dir_list.count("LoginData.csv") == 0:
        user_id = "User0001"
        data = pd.Series({"DeviceId": device_id,
                          "UserId": user_id})
        data = pd.DataFrame([data])
        data.to_csv(data_dir + '/LoginData.csv', index=False)
        new_user = True
    else:
        data = pd.read_csv(data_dir + '/LoginData.csv')
        device_data = data[data["DeviceId"] == device_id]
        if len(device_data) == 0:
            data_len = len(data)
            tmp = str(int(data["UserId"][data_len - 1][-len(str(data_len)):]) + 1)
            user_id = "User" + "0" * (4 - len(tmp)) + tmp
            data = data.append(pd.Series({"DeviceId": device_id,
                                          "UserId": user_id}), ignore_index=True)
            data.to_csv(data_dir + '/LoginData.csv', index=False)
            new_user = True
        else:
            user_id = device_data['UserId'][device_data.index[0]]
            new_user = False
    output_data["status"] = "success"
    output_data["user_id"] = user_id
    output_data["new_user"] = new_user
    return jsonify(output_data)


@app.route('/lastDayDepressionState', methods=['POST'])
def status():
    input_data = request.get_json()
    output_data = dict()
    try:
        user_id = input_data['UserId']
        if not check_valid_user(user_id):
            output_data['status'] = "Failure. Invalid User."
            return jsonify(output_data)
        status_file = pd.read_csv(data_dir + '/StatusData.csv')
    except KeyError:
        output_data['status'] = "Failure. Data not found"
        return jsonify(output_data)
    except FileNotFoundError:
        output_data['status'] = "Error. No records found."
        return jsonify(output_data)
    state_list = status_file[status_file["UserId"] == user_id]
    if len(state_list) == 0:
        state = "Error. No records found."
    else:
        state = state_list['State'][state_list.index[0]]
    output_data["status"] = state
    return jsonify(output_data)


@app.route('/lastWeekDepressionState', methods=['POST'])
def patient_status():
    input_data = request.get_json()
    output_data = dict()
    try:
        user_id = input_data['UserId']
        if not check_valid_user(user_id):
            output_data['status'] = "Failure. Invalid User."
            return jsonify(output_data)
    except KeyError:
        output_data['status'] = "Failure. Data not found"
        return jsonify(output_data)
    dir_list = os.listdir(data_dir)
    if dir_list.count(user_id) == 0:
        output_data['status'] = "Error. No records found for this user"
    else:
        dir_list = os.listdir(data_dir + "/" + user_id)
        if len(dir_list) < 5:
            output_data['status'] = 'Error. Insufficient data.'
        else:
            working_dir = data_dir + "/" + user_id
            files = os.listdir(working_dir)
            data = pd.DataFrame()
            for file in files:
                df = pd.read_csv(working_dir + "/" + file)
                data = data.append(df, ignore_index=True)
            del data['timestamp']
            data = data.to_numpy()
            output_data['status'] = get_patient_state(data, 'patientwise.sav')
    return jsonify(output_data)


@app.route('/', methods=['POST'])
def upload_activity():
    user_id = request.form['UserId']
    file = request.files.get('activity_file')
    output_data = dict()
    if not check_valid_user(user_id):
        output_data['status'] = "Failure. Invalid User."
        return jsonify(output_data)
    df = pd.read_csv(file)
    create_user_file(user_id, df, file.filename)
    del df['timestamp']
    data = df.to_numpy()
    k_mean_model = KMeans(n_clusters=4, max_iter=500)
    k_mean_model.fit(data)
    labels = Counter(k_mean_model.labels_)
    extra_params = {'No_to_mild_Activity': labels[0],
                    'high_activity': labels[3]}
    state = get_patient_state(data, 'daywise.sav', extra_params)
    output_data['status'] = state
    update_status_file(user_id, state)
    return jsonify(output_data)


def get_patient_state(data, model_name, extra_params=None):
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
        data.update(extra_params)
    return fit_model(pd.DataFrame([data]), model_name)


def fit_model(data, model_name):
    file = open(model_dir + '/' + model_name, 'rb')
    model = pickle.load(file)
    test_data = []
    test_data.extend(data.to_numpy())
    prediction = model.predict(test_data)
    file.close()
    if prediction[0][0] == 1:
        return "Depressed"
    else:
        return "Not Depressed"


def check_valid_user(user_id):
    dir_list = os.listdir()
    if dir_list.count("Data") == 0:
        return False
    dir_list = os.listdir("Data")
    if dir_list.count("LoginData.csv") == 0:
        return False
    else:
        data = pd.read_csv(data_dir + '/LoginData.csv')
        device_data = data[data["UserId"] == user_id]
        if len(device_data) == 0:
            return False
        else:
            return True


def create_user_file(user_id, data_frame, file_name):
    dir_list = os.listdir()
    if dir_list.count("Data") == 0:
        os.mkdir("Data")
    dir_list = os.listdir("Data")
    if dir_list.count(user_id) == 0:
        os.mkdir(data_dir + "/" + user_id)
        path = os.path.join(data_dir + "/" + user_id, file_name)
        data_frame.to_csv(path, index=False)
    else:
        dir_list = os.listdir(data_dir + "/" + user_id)
        if dir_list.count(file_name) > 0:
            print("File already exists.")
        else:
            path = os.path.join(data_dir + "/" + user_id, file_name)
            if len(dir_list) >= 15:
                sorted_file_name = sorted(dir_list)
                os.remove(os.path.join(data_dir + "/" + user_id, sorted_file_name[0]))
                data_frame.to_csv(path, index=False)
            else:
                data_frame.to_csv(path, index=False)


def update_status_file(user_id, state):
    dir_list = os.listdir()
    if dir_list.count("Data") == 0:
        os.mkdir("Data")
    dir_list = os.listdir("Data")
    if dir_list.count("StatusData.csv") == 0:
        data = {'UserId': user_id,
                'State': state}
        data = pd.DataFrame([data])
        data.to_csv(data_dir + "/StatusData.csv", index=False)
    else:
        data = pd.read_csv(data_dir + "/StatusData.csv")
        user_data = data[data['UserId'] == user_id]
        if len(user_data) == 0:
            data = data.append(pd.Series({"UserId": user_id,
                                          "State": state}), ignore_index=True)
            data.to_csv(data_dir + '/StatusData.csv', index=False)
        else:
            # data.drop(user_data.index[0], inplace=True)
            # data = data.append(pd.Series({"UserId": user_id,
            #                              "State": state}), ignore_index=True)
            data.at[user_data.index[0], 'State'] = state
            data.to_csv(data_dir + '/StatusData.csv', index=False)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
