from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World"

if __name__ == '__main__':
    print("Starting test flask app...")
    app.run(debug=True, use_reloader=False, port=5002)
