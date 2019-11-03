import pandas as pd


def get_emo_nrc_lexicon():
    filepath = "./data/NRC_emotion_lexicon_list.txt"
    return pd.read_csv(
        filepath,  names=["word", "emotion", "association"],  sep='\t').pivot(index='word', columns='emotion', values='association').reset_index()
