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
        
        precomputed_results = read("cached_results", 0, ",")
        df = df.merge(precomputed_results, on="id", how="left")
        data = request.get_json()
        
        related_ids_str = df.loc[df['id'] == data['id'], retrieval_system].values[0]
        related_ids = list(related_ids_str.split(';'))
        # Filter the DataFrame to get rows with the related IDs
        related_songs_df = df[df['id'].isin(related_ids)]


        # Convert the result to a JSON-compatible format
        related_songs_df.drop(["tfidf", "resnet", "ivec256"], axis=1, inplace=True)
        return related_songs_df.to_json(orient='records')

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