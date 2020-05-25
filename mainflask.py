import os.path
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)
script_dir = os.path.dirname(__file__)


# @ signifies a decorator- way to wrap a fun and modify it
@app.route('/login', methods=['POST'])
def index():
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
        data.to_csv(script_dir + '/Data/LoginData.csv', index=False)
        new_user = True
    else:
        data = pd.read_csv(script_dir + '/Data/LoginData.csv')
        device_data = data[data["DeviceId"] == device_id]
        if len(device_data) == 0:
            data_len = len(data)
            tmp = str(int(data["UserId"][data_len - 1][-len(str(data_len)):]) + 1)
            user_id = "User" + "0" * (4 - len(tmp)) + tmp
            data = data.append(pd.Series({"DeviceId": device_id,
                                          "UserId": user_id}), ignore_index=True)
            data.to_csv(script_dir + '/Data/LoginData.csv', index=False)
            new_user = True
        else:
            user_id = device_data['UserId'][device_data.index[0]]
            new_user = False
    output_data["status"] = "success"
    output_data["user_id"] = user_id
    output_data["new_user"] = new_user
    return jsonify(output_data)


# app.config['ALLOWED_EXTENSION'] = ['PNG', 'JPG', 'JPEG']



"""
def get_predicted_value(input_image):
    directory = script_dir + '/ModelData/'
    input_image = image.load_img(script_dir + '/static/' + input_image.filename)
    input_image = input_image.resize((128, 128))
    input_image = image.img_to_array(input_image)
    input_image = input_image / 255
    print(directory + 'Engine.h5')
    modelML = load_model(directory + 'Engine.h5')
    pred = modelML.predict(np.reshape(np.array(input_image), (1, 128, 128, 3)))
    index = pred[0].argmax(axis=0)
    print(pred)
    print(index)
    if index == 0:
        return "Non Viable tumor"
    elif index == 1:
        return "Viable tumor"
    else:
        return "Non tumor"


@app.route('/result', methods=['POST'])
def result():
    warning = ''
    typeOsteo = ''
    pred = False
    invalid = False
    input_image = request.files["image"]
    if input_image.filename != '':
        print(input_image)
        if not allowedImage(input_image.filename):
            warning = "ALERT: The file must be an Image !"
            print(warning)
            invalid = True
            #return redirect('http://localhost:5000')
        else:
            input_image.save(script_dir + '/static/' + input_image.filename)
            typeOsteo = get_predicted_value(input_image)
            pred = True
    else:
        warning = "ALERT: Please select an Image !"
    img = input_image.filename
    return render_template("homepage.html", img=img, typeOsteo=typeOsteo, pred=pred, warning=warning)
"""

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
