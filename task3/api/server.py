from flask import Flask, request
from flask_restx import Api, Resource, fields
import pandas as pd

app = Flask(__name__)
api = Api(app)

# Define your data model
class SongInfoModel:
    def __init__(self, title, artist):
        self.title = title
        self.artist = artist

    def retrieve(self):
        return df[(df['title'] == self.title) & (df['artist'] == self.artist)]

# Setup Flask-RESTx
song_model = api.model('Song', {
    'title': fields.String(required=True, description='The song title'),
    'artist': fields.String(required=True, description='The song artist')
})

@api.route('/search')
class SongSearch(Resource):
    @api.expect(song_model)
    def post(self):
        data = request.get_json()
        song = SongInfoModel(data['title'], data['artist'])
        
        result = song.retrieve()

        # Convert the result to a JSON-compatible format
        return result.to_dict(orient='records')

if __name__ == '__main__':
    df = pd.DataFrame({
        'id': [],  # Fill with your data
        'artist': [],
        'title': [],
        'album': []
    })
    
    app.run(debug=True)