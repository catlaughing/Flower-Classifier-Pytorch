from flask import Flask, request, render_template
from web_predict import predict
import os
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        print(request.files)
        target = os.path.join(APP_ROOT,'static/img/uploaded')
        if 'file' not in request.files:        
            print('file not uploaded')
            return render_template('index.html')
            # return
 
        file = request.files['file']
        filename = file.filename
        dest = 'static/img/uploaded/'+filename
        print(dest)
        image = file.read()
        probs,name = predict(image)
        file.save(dest)
        return render_template('result.html',name=name.capitalize(),probs=probs[0],image_file=dest)

# @app.route('/uploads/<filename>')
# def send_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)
if __name__ == '__main__':
    app.run(debug=True)