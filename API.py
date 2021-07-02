from flask import Flask,request,render_template,url_for
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.neighbors
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/genrow/')
def genrow():
    return render_template('genrow.html')


@app.route('/genrowaction/',methods=['POST'])
def submit():
    if request.method=='POST':
        global rownum
        global trainingData
        global testData
        global targetTrainingData
        global targetTestdata

        rownum= int(request.form['rownum'])
        featuresData,targetData  = sklearn.datasets.make_blobs(n_samples=rownum,centers= 2, n_features=2)
        print(featuresData,type(featuresData),featuresData.shape)
        print(targetData,type(targetData),targetData.shape)
        trainingData,testData,targetTrainingData,targetTestdata = sklearn.model_selection.train_test_split(featuresData,targetData,test_size=0.25)
        print(trainingData,testData,targetTrainingData,targetTestdata)

        return render_template('datasets.html',trainingLen=len(trainingData),trainingData=trainingData,targetTrainingData=targetTrainingData,
                               testLen = len(testData),testData=testData,targetTestdata=targetTestdata
                               )


@app.route('/process/',methods=['POST'])
def process():
    if request.method=='POST':
        algo = request.form['algo']
        if algo=='knn':
            global rownum
            global trainingData
            global testData
            global targetTrainingData
            global targetTestdata
            global result
            scaler = sklearn.preprocessing.StandardScaler().fit(trainingData)
            print(type(scaler))
            print('Mean =',scaler.mean_)
            print('Variance = ',scaler.scale_)
            #print('trainingData',trainingData)
            trainingData = scaler.transform(trainingData)
            testData = scaler.transform(testData)
            print('Scaled Feature training data ')
            print('trainingData',trainingData)
            print('testData',testData)
            knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
            knn.fit(trainingData,targetTrainingData)
            result = knn.predict(testData)
            print('result ',result)
            print('actual',targetTestdata)
            return render_template('results.html' ,resultLen=len(result),testData=testData,targetTestdata=targetTestdata,result=result)



rownum = None
trainingData = None
testData= None
targetTrainingData= None
targetTestdata= None
result = None
if __name__ == "__main__":
    app.debug=True
    app.run()









