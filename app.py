from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np

app = Flask(__name__)
 
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [int(x) for x in request.form.values()]
    avg_per = (np.array(int_features[:4]).sum())/4
    int_features.append(avg_per)
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    if prediction ==0:
        prediction_text = "Sorry, You aren't placed. Try next year."
    else:
        prediction_text = 'Congrats, You got placed.'
    return render_template('index.html',prediction_text=prediction_text)

if __name__=='__main__':
    app.run(debug=True) 
