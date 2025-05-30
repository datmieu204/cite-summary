from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from facet_prediction.config import *
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


def get_section_title_by_sid(doc, sid):
    for section in doc['sections']:
        for sent in section['sents']:
            if sent['sid'] == sid:
                return section["text"]
    print(f"ERROR when find {sid} in {doc['ID']}")

def get_all_section_title_by_sids(doc, sids):
    sections = []
    for sid in sids:
        section = get_section_title_by_sid(doc, sid)
        sections.append(section)
    return sections


def get_important_features(name, X_train, y_train, X_test, num=100):
    selection_name = 'KBest_for_{}'.format(name)
    model_select = SelectKBest(score_func=chi2, k=num)
    model_select.fit(X_train, y_train)
    X_train_selected = model_select.transform(X_train)
    try:
        X_test_selected = model_select.transform(X_test)
    except:
        X_test_selected = -1

    with open(ROOT_DIR + '/facet_prediction/dataset/encoders/{}.bin'.format(selection_name), 'wb+') as f:
        pickle.dump(model_select, f)
    return model_select, X_train_selected, X_test_selected

def label_encoder_for_dataframe(df, test_df, column):
    encoder_name = 'label_encoder_for_{}'.format(column)
    print(encoder_name)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df[column])
    y_test = label_encoder.transform(test_df[column])
    with open(ROOT_DIR + '/facet_prediction/dataset/encoders/{}.bin'.format(encoder_name), 'wb+') as f:
        pickle.dump(label_encoder, f)
    return label_encoder, y_train, y_test


def tfidf_for_dataframe(df, test_df, column):
    encoder_name = 'tfidf_encoder_for_{}'.format(column)
    print(encoder_name)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', dtype=np.float32)
    X_train = vectorizer.fit_transform(df[column])
    try:
        X_test = vectorizer.transform(test_df[column])
    except:
        X_test = -1
    with open(ROOT_DIR + '/facet_prediction/dataset/encoders/{}.bin'.format(encoder_name), 'wb+') as f:
        pickle.dump(vectorizer, f)
    return vectorizer, X_train, X_test

def tf_for_dataframe(df, test_df, column):
    encoder_name = 'tfidf_encoder_for_{}'.format(column)
    print(encoder_name)
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', dtype=np.float32)
    X_train = vectorizer.fit_transform(df[column])
    try:
        X_test = vectorizer.transform(test_df[column])
    except:
        X_test = -1
    with open(ROOT_DIR + '/facet_prediction/dataset/encoders/{}.bin'.format(encoder_name), 'wb+') as f:
        pickle.dump(vectorizer, f)
    return vectorizer, X_train, X_test

