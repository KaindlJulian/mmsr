from flask import Flask, request
from flask_restx import Api, Resource, fields
from pathlib import Path
import sys
path_root = Path(__file__).joinpath("..\\..").resolve()
sys.path.append(str(path_root))
print(sys.path)
import pandas as pd
from task1.retrieval_system import RetrievalSystem, SongInfo
from task1.similarity_measure import (
    cosine_similarity,
    dot_product,
    manhattan_distance,
    euclidean_distance,
    random_similarity,
)
from utils import read, embed_and_merge
from task3.methods_task3 import early_fusion

app = Flask(__name__)
api = Api(app)

g_variable ="Tstest"

# Setup Flask-RESTx
song_model = api.model('Song', {
    'title': fields.String(required=True, description='The song title'),
    'artist': fields.String(required=True, description='The song artist')
})

@api.route('/earlyf')
class SongSearch(Resource):
    @api.expect(song_model)
    def post(self):
        data = request.get_json()
        
        (rs_cos_early_fusion_1, feature_name1, test) = early_fusion("bert", "musicnn", df)

        sample_song = SongInfo(title=data['title'], artist=data['artist'])
        result =  rs_cos_early_fusion_1.retrieve(sample_song)

        # Convert the result to a JSON-compatible format
        print(result.columns)
        return result.to_json(orient='records')

@api.route('/tfidf')
class SongSearch(Resource):
    @api.expect(song_model)
    def post(self):
        data = request.get_json()

        sample_song = SongInfo(title=data['title'], artist=data['artist'])
        result =  rt_dict['tfidf'].retrieve(sample_song)

        # Convert the result to a JSON-compatible format
        print(result.columns)
        return result.to_json(orient='records')

if __name__ == '__main__':
    df = read("information", 0)
    genres = read("genres", 0)
    # convert genre to actual list via eval
    genres["genre"] = genres["genre"].apply(eval).apply(set)
    df = df.merge(genres, on="id", how="left")
    visual_feature = "resnet"
    stats = read(visual_feature, 0)
    df = embed_and_merge(df, stats, visual_feature)
    yt_link = read("url", 0)
    df = df.merge(yt_link, on="id", how="left")

    for audio_feature in ["mfcc_bow", "blf_spectral", "ivec256", "musicnn"]:
        stats = read(audio_feature, 0)
        df = embed_and_merge(df, stats, audio_feature)

    for text_feature in ["lyrics_bert", "lyrics_word2vec", "lyrics_tf-idf"]:
        stats = read(text_feature, 0)
        df = embed_and_merge(df, stats, text_feature.split("_")[1])
    df = df.drop(5)
    df = df.reset_index()
    rt_dict = {'tfidf': RetrievalSystem(
        df=df,
        sim_metric=cosine_similarity,
        sim_feature="tf-idf",
    )}
    app.run(debug=True)