from flask import Flask, request, jsonify, render_template
import numpy
import pandas as pd
import pickle

application = Flask(__name__)
app = application

# Load the pre-trained models
ridge_model = pickle.load(open('C:\\Users\\Samyak\\OneDrive\\Desktop\\DWDA\\HOUSE_PREDICTION\\Ridgeregressor.pkl', 'rb'))
Preprocessor = pickle.load(open('C:\\Users\\Samyak\\OneDrive\\Desktop\\DWDA\\HOUSE_PREDICTION\\preprocessor.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')



@app.route("/predictData",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        area=float(request.form['area'])
        bedrooms=float(request.form['bedrooms'])
        bathrooms=float(request.form['bathrooms'])
        stories=float(request.form['stories'])
        mainroad=request.form['mainroad']
        guestroom = request.form['guestroom']
        basement = request.form['basement']
        hotwaterheating = request.form['hotwaterheating']
        airconditioning = request.form['airconditioning']
        parking = float(request.form['parking'])
        prefarea = request.form['prefarea']
        furnishingstatus = request.form['furnishingstatus']
        input_features = pd.DataFrame(
                                      [[area, bedrooms, bathrooms, stories, 
                                       mainroad, guestroom, basement, hotwaterheating,
                                       airconditioning, parking, prefarea, furnishingstatus]],
                                       columns=['area', 'bedrooms', 'bathrooms', 'stories', 
                                       'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                                       'airconditioning', 'parking', 'prefarea', 'furnishingstatus'] # Replace with your actual column names
                                    )
        preprocessed_inputs=Preprocessor.transform(input_features)
        prediction=ridge_model.predict(preprocessed_inputs)

        return render_template('home.html',result=prediction[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":  # Corrected this line
    app.run(host="0.0.0.0")  # Removed the extra period
