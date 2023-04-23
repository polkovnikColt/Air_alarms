from flask import Flask, jsonify, request
import os
import json

app = Flask(__name__)

prediction_data_path = "prediction.json" 
prediction_script_path = "./prediction.py"

@app.route('/api/prediction/v1/predict', methods=['POST'])
def predict_air_alarm():
    try:
      data = request.json
      locations = data.get('locations')

      f = open(prediction_data_path)
  
      data = json.load(f)
  
      f.close()


      if locations == "*":
          return jsonify(data), 200
      else:
          regions_forecast_filtered = []
          for alarm in data["regions_forecast"]:
            print(alarm.keys())
            if list(alarm.keys())[0] in locations:
                 regions_forecast_filtered.append(alarm)
          return jsonify({
              "last_model_train_time": data['last_model_train_time'],  
              "last_prediction_time": data["last_prediction_time"],
              "regions_forecast": regions_forecast_filtered
              }), 200

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

@app.route('/api/prediction/v1/update_forecast', methods=["GET"])
def update_forecast():
    try:
        cmd = os.path.join(os.getcwd(), "prediction.py")
        os.system('{} {}'.format('python3', cmd))
        return jsonify({"message": "Updated"}), 200
    except Exception as e:
        return jsonify({"Error": str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True)