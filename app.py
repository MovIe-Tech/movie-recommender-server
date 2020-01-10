from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    dic = {
        'foo': 'bar',
        'ほげ': 'ふが'
    }
    return jsonify(dic)
    

if __name__ == '__main__':
    app.run()