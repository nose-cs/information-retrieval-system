from functools import cache

from flask import Flask, jsonify
from main import get_extended_boolean_model

app = Flask(__name__)

@app.route('/search/<query>')
def get_results(query):
    return jsonify(results = [doc.doc_title for doc in get_extended_boolean_model().query(query)])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)