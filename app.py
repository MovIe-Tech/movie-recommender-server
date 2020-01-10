from flask import Flask, jsonify, request
from main.find import find

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    input = request.args.get('input')
    return jsonify(find(input))


if __name__ == '__main__':
    app.run()
