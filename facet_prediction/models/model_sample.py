from facet_prediction.models.abstract_model import BaseModel
import pandas as pd
from facet_prediction.config import *
import json

class RandomForest(BaseModel):
    def __init__(self, data):
        self.data = data

    def train(self, samples):
        print("Something")

    def predict(self, samples):
        print("Something")

if __name__ == '__main__':
    f = open(ROOT_DIR + '/dataset/preprocessed_data/encoded_data.json', )
    train = json.load(f)
    f.close()
    train = pd.DataFrame(train)

    f = open(ROOT_DIR + '/dataset/preprocessed_data/test_encoded_data.json', )
    test = json.load(f)
    f.close()
    test = pd.DataFrame(test)

    print("Done")


