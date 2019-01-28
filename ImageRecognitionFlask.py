from flask import Flask

app = Flask(__name__)

@app.route('/index')

def run():
    return "Flask running"
