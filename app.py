from flask import Flask, request, render_template, send_from_directory
from web_predict import predict
import os
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/img/uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        if file and allowed_file(file.filename):
            filename = file.filename
            dest = 'static/img/uploaded/'+filename
            file.stream.seek(0)
            file.save(dest)
            print(dest)
            file.stream.seek(0)
            image = file.read()
            probs,name = predict(image)
            return render_template('result.html',name=name.capitalize(),probs=probs[0],image_file=dest)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)