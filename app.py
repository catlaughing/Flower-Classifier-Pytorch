from flask import Flask, request, render_template
from web_predict import predict
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html',value='Shadieq')

    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        probs,name = predict(image)
        return render_template('result.html',name=name.capitalize(),probs=probs[0])
    
if __name__ == '__main__':
    app.run(debug=True)