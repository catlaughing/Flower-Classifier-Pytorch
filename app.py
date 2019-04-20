from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html',value='Shadieq')

    if request.method == 'POST':
        print(request.file)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        
        return render_template('result.html')
    
if __name__ == '__main__':
    app.run(debug=True)