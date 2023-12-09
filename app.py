from flask import Flask,render_template,request
from sklearn.datasets import load_iris
import numpy as np
import model
import pickle as pkl
app=Flask(__name__,template_folder="templates")
model=pkl.load(open('model.pkl','rb'))
z=pkl.load(open('z.pkl','rb'))
L=pkl.load(open('L.pkl','rb'))
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    
    Borough=str(request.args.get("Borough"))
    Lot=request.args.get("Lot")
    JobType=str(request.args.get("JobType"))
    CommunityBoard=request.args.get("CommunityBoard")
    year=request.args.get("year")
    Borough=z.fit_transform([Borough])
    JobType=L.fit_transform([JobType])
    print(Borough)
    print(JobType)
    arr=np.array([[int(Borough[0]),int(Lot),int(JobType[0]),int(CommunityBoard),int(year)]])
    prediction=model.predict(arr)
    prediction=round(prediction[0])
    return render_template('out.html', output=prediction)
if __name__ == "__main__":
    app.run(debug=True)