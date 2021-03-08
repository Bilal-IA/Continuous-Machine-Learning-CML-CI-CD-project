from flask import Flask, request
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)
pickle_in = open("model.pkl","rb")
reg_model=pickle.load(pickle_in)

@app.route('/predict',methods=["Get"])
def predict_house_price():
    
    """Real Estate Price Prediction !!.
    ---
    parameters:  
      - name: transaction_date
        in: query
        type: number
        required: true
      - name: house_age
        in: query
        type: number
        required: true
      - name: distance_to_MRT_station
        in: query
        type: number
        required: true
      - name: number_of_convenience_stores
        in: query
        type: number
        required: true
      - name: latitude
        in: query
        type: number
        required: true
      - name: longitude
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    transaction_date=request.args.get("transaction_date")
    house_age=request.args.get("house_age")
    distance_to_MRT_station=request.args.get("distance_to_MRT_station")
    number_of_convenience_stores=request.args.get("number_of_convenience_stores")
    latitude=request.args.get("latitude")
    longitude_=request.args.get("longitude_")
    prediction=reg_model.predict([[transaction_date,house_age,distance_to_MRT_station,number_of_convenience_stores,latitude,longitude_]])
    print(prediction)
    return "The predicted House Price is"+str(prediction)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    
    
