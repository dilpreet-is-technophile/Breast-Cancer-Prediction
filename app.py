from flask import Flask, render_template,request
import numpy as np 
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/uploader',methods=['GET','POST'])
def uploader():
    if (request.method == 'POST'):
        features_file = request.files['file1']
        data=pd.read_csv(features_file)
        prediction_type=model.predict(data)
        prediction_probab=model.predict_proba(data)
        

        
        output2= prediction_type[0]
        if(output2=='M'):
            output2='Malignant'
            prediction_probab=prediction_probab[0,1]*100
        else:
            output2='Benign'
            prediction_probab=prediction_probab[0,0]*100
        output1= round(prediction_probab,2)


    return render_template("index.html",text1="The tumor for the given features is predicted as {}".format(output2)+" with the probability of {}%".format(output1))

# @app.route('/predict',methods=['POST'])
# def predict():
#     features_file=12


if __name__ == "__main__":
    app.run(debug=True)
