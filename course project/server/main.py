import pandas as pd
import dill
import text_cleaner
import flask
import os
import json

clean_series = text_cleaner.clean_series

model_path = 'server/saved/pipeline.dill'

global model
with open(model_path, 'rb') as f:
    model = dill.load(f)
# print(model)

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def general():
    return 'Toxicometer. Example usage: /predict?text=["Good morning!", "Will take a look."]'


@app.route('/predict', methods=['GET'])
def predict():
    data = { 'success': False }
    text = ''

    if 'text' in flask.request.args:
        text = json.loads(flask.request.args['text'])
    else:
        return flask.jsonify(data)

    pred = None
    try:
        pred = model.predict_proba(pd.Series(text))
    except AttributeError as e:
        return flask.jsonify(data)

    data['predictions'] = pred[:, 1].tolist()
    data['success'] = True
        
    return flask.jsonify(data)


if __name__ == '__main__':
	print('Starting Flask server...')
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
