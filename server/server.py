from flask import Flask, make_response, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from pathlib import Path
import sys
path_root = Path(__file__).joinpath("..","..").resolve()
sys.path.append(str(path_root))
print(sys.path)
import pandas as pd
from functools import wraps

from dataclasses import dataclass
import numpy as np
import pandas as pd
import os


@dataclass
class SongInfo:
    title: str
    artist: str



def read(feature, h=0, delimiter="\t"):
    file_path = os.path.join(
        '.', f"id_{feature}_mmsr.tsv"
    )
    return pd.read_csv(file_path, delimiter=delimiter, header=h)

def read_feather(feature):
    file_path = os.path.join(
        '.', f"{feature}.feather"
    )
    return pd.read_feather(file_path)


app = Flask(__name__)
CORS(app)
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'x-api-key'
    }
}
api = Api(app, authorizations=authorizations, security='apikey')


API_KEY="6908c5e16f04b08a"

def require_apikey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            response = make_response(jsonify({"message": "API key is missing or incorrect"}), 401)
            return response
    return decorated_function

# Setup Flask-RESTx
song_model = api.model('Song', {
    'id': fields.String(required=True, description='The song id')
})


@api.route('/<retrieval_system>')
class SongSearch(Resource):
    @api.expect(song_model)
    @require_apikey
    def post(self, retrieval_system):
        df = read("information", 0)
        genres = read("genres", 0)
        genres["genre"] = genres["genre"].apply(eval).apply(set)
        df = df.merge(genres, on="id", how="left")
        yt_link = read("url", 0)
        df = df.merge(yt_link, on="id", how="left")

        precomputed_results = read_feather(retrieval_system)
        
        song_id = request.json['id']
        precomputed_results = precomputed_results[song_id]

        precomputed_results
        if isinstance(precomputed_results, pd.Series):
            precomputed_results = precomputed_results.reset_index()
            precomputed_results.columns = ['id', 'similarity']  # Rename columns appropriately

        # Step 2: Merge with df
        result_df = df.merge(precomputed_results, on='id', how='inner')
        result_df
        return result_df.to_json(orient='records')

@api.route('/allSongs')
class SongSearch(Resource):
    @require_apikey
    def get(self):
        df = read("information", 0)
        genres = read("genres", 0)
        genres["genre"] = genres["genre"].apply(eval).apply(set)
        df = df.merge(genres, on="id", how="left")
        yt_link = read("url", 0)
        df = df.merge(yt_link, on="id", how="left")
        
        precomputed_results = read("cached_results", 0, ",")
        df = df.merge(precomputed_results, on="id", how="left")
        res = df.drop(["tfidf", "resnet", "ivec256"], axis=1)
        return res.to_json(orient='records')

if __name__ == '__main__':

    
    app.run(debug=True)