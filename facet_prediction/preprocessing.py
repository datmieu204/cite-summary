import math
import os
import pandas as pd
import json

from sklearn.calibration import LabelEncoder

"""
Feature Position
"""

# Read the data from the folder
def read_data(folder_path):
    data = {}
    for file_path in os.listdir(folder_path):
        if file_path.endswith(".json"):
            file_id = file_path.split('.')[0]
            file_path = os.path.join(folder_path, file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                data[file_id] = json.load(file)
    return data


# Get the position of the sentence in the document
def get_position(document_data, sentence_sid):
    sentence = []

    total_sentences = sum(len(section['sents']) for section in document_data['sections'])

    for section in document_data['sections']:
        for sent in section['sents']:
            if sent['sid'] == sentence_sid:
                section_number = int(section['number'])
                sentence_section_number = sent['ssid']
                section_id = f"{section_number} / {len(document_data['sections'])}"
                sentence_in_section_ratio = f"{sent['ssid']} / {len(section['sents'])}"
                sentence_in_document_ratio = f"{sent['sid']} / {total_sentences}"
                sentence.append(section_id)
                sentence.append(sentence_in_section_ratio)
                sentence.append(sentence_in_document_ratio)
    return sentence, section_number, sentence_section_number

def modify_position_arv(sentence, section_arv):
    sum_sec = 0

                    

def preprocess_data_output_1(citations_data, documents_data):
    df = pd.DataFrame()

    samples = []

    for citation_id, citation_data in citations_data.items():
        for citation in citation_data['citations']:
            refer_id = citation['refer_ID'] 
            refer_sids = citation['refer_sids'] # list
            refer_text = citation['refer_text']
            cite_id = citation['cite_ID']
            cite_sids = citation['cite_sids'] # list
            cite_text = citation['cite_text']
            cite_marker_sids = citation['cite_maker_sids']
            labels = citation['label']

            refer_document = documents_data.get(refer_id)
            cite_document = documents_data.get(cite_id)

            if not refer_document or not cite_document:
                continue
                
            # Update the average position.
            #-----------------------------------------------------------------------------------------------------------------------------------------------------
            refer_section_array = []
            cite_section_array = []

            refer_sum_sen_sec = 0
            refer_sum_sen_doc = 0

            for refer_sid in refer_sids:
                # ERROR refer_sid = 0
                if refer_sid == 0:
                    refer_sid = 1
                refer_sentence, refer_section_number, refer_sentence_section = get_position(refer_document, refer_sid)
                refer_section_array.append(refer_section_number)

                refer_sum_sen_sec += int(refer_sentence_section)
                refer_sum_sen_doc += int(refer_sid)
           
            refer_section_arv = round(sum(refer_section_array)/len(refer_section_array))
            refer_sentence_section_arv = round(refer_sum_sen_sec/len(refer_sids))
            refer_sentence_document_arv = round(refer_sum_sen_doc/len(refer_sids))

            refer_sentence_1 = refer_sentence[0].split('/')
            refer_sentence_1[0] = str(refer_section_arv)
            refer_sentence[0] = '/'.join(refer_sentence_1)

            refer_sentence_2 = refer_sentence[1].split('/')
            refer_sentence_2[0] = str(refer_sentence_section_arv)
            refer_sentence[1] = '/'.join(refer_sentence_2)

            refer_sentence_3 = refer_sentence[2].split('/')
            refer_sentence_3[0] = str(refer_sentence_document_arv)
            refer_sentence[2] = '/'.join(refer_sentence_3)


            cite_sum_sen_sec = 0
            cite_sum_sen_doc = 0

            for cite_sid in cite_sids:
                cite_sentence, cite_section_number, cite_sentence_section = get_position(cite_document, cite_sid)

                cite_section_array.append(cite_section_number)
               
                cite_sum_sen_sec += int(cite_sentence_section)
                cite_sum_sen_doc += int(cite_sid)

            cite_section_arv = round(sum(cite_section_array)/len(cite_section_array))
            cite_sentence_section_arv = round(cite_sum_sen_sec/len(cite_sids))
            cite_sentence_document_arv = round(cite_sum_sen_doc/len(cite_sids))

            cite_sentence_1 = cite_sentence[0].split('/')
            cite_sentence_1[0] = str(cite_section_arv)
            cite_sentence[0] = '/'.join(cite_sentence_1)

            cite_sentence_2 = cite_sentence[1].split('/')
            cite_sentence_2[0] = str(cite_sentence_section_arv)
            cite_sentence[1] = '/'.join(cite_sentence_2)

            cite_sentence_3 = cite_sentence[2].split('/')
            cite_sentence_3[0] = str(cite_sentence_document_arv)
            cite_sentence[2] = '/'.join(cite_sentence_3)

            #-----------------------------------------------------------------------------------------------------------------------------------------------------

            if ',' in labels:
                # Convert string to list
                labels_list = eval(labels)
                for label in labels_list:
                    samples.append([refer_id, refer_sids, refer_section_array, cite_id, cite_sids, cite_section_array] + refer_sentence + cite_sentence + [label])
            else:
                samples.append([refer_id, refer_sids, refer_section_array, cite_id, cite_sids, cite_section_array] + refer_sentence + cite_sentence + [labels])

    df = pd.DataFrame(samples, columns=['refer_id', 'refer_sid', 'refer_section', 'cite_id', 'cite_sid', 'cite_section',
                                    'refer_section_in_sections', 'refer_sentence_in_section_ratio', 'refer_sentence_in_document_ratio', 
                                    'cite_section_in_sections', 'cite_sentence_in_section_ratio', 'cite_sentence_in_document_ratio', 'label'])

    return df


def preprocess_data_output_2(df):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['label'.lower()])
    df['label'] = encoded_labels
    return df

if __name__ == "__main__":
    citations_data = read_data("/citesumm/source/citations")
    documents_data = read_data("/citesumm/source/documents")
    output_folder = '/citesumm/facet_prediction/dataset'
    os.makedirs(output_folder, exist_ok=True)

    df_output1= preprocess_data_output_1(citations_data, documents_data)
    print(df_output1)
    df_output1.to_csv(os.path.join(output_folder, 'output1_1.csv'), index=False)
   
    df_output2 = preprocess_data_output_2(df_output1)
    # print(df_output2)
    # df_output2.to_csv(os.path.join(output_folder, 'output2.csv'), index=False)
