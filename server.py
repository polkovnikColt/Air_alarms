from flask import Flask, jsonify, request
import os

app = Flask(__name__)

prediction_data_path = "" 
prediction_script_path = ""

@app.route('/api/prediction/v1/predict', methods=['POST'])
def predict_air_alarm():
    try:
      data = request.json
      locations = data.get('locations')

      response = []
      with open(prediction_data_path) as f:
          response = f.readlines()

      if locations == "*":
          return jsonify(response), 200
      else:
          response_filtered = [data for data in response if data["location"] in locations]
          return jsonify(response_filtered), 200

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

@app.route('/api/prediction/v1/update_forecast', methods=["GET"])
def update_forecast():
    try:
        os.system(prediction_script_path)
        return jsonify({"message": "Updates"}), 200
    except Exception as e:
        return jsonify({"Error": str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True)