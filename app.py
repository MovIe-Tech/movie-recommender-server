from flask import Flask, jsonify, request
from find import find

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    text = request.args.get('input')
    return jsonify(find(text))


if __name__ == '__main__':
    app.run()
