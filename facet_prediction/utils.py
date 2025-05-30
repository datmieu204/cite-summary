import json
import re
import pandas as pd
from facet_prediction.config import *

def save_json(track, data, path):
    json_object = json.dumps(data, indent=4)

    if os.path.exists(path):
        with open(path, "r") as outfile:
            data = outfile.read()
            if data != json_object:
                print(path, "in " + track + " file exists and not same")
                # path = path.replace(".json", "_other.json")

    with open(path, "w") as outfile:
        outfile.write(json_object)

def remove_multi_space_from_string(s):
    return re.sub(' +', ' ', s)

def get_sent_from_sid(sid, doc):
    for section in doc['sections']:
        for sent in section['sents']:
            if sent['sid'] == sid:
                return sent
    print("Can not find {} in {}".format(sid, doc['ID']))

def read_all_docs(is_test=False):
    if not(is_test):
        folder_path = ROOT_DIR + "/source/documents"
    else:
        folder_path = ROOT_DIR + "/source/test_documents"

    data = {}
    for file_path in os.listdir(folder_path):
        if file_path.endswith(".json"):
            file_id = file_path.split('.')[0]
            file_path = os.path.join(folder_path, file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                data[file_id] = json.load(file)
    return data

nom_label_dict = {
    ' Results_Citation' : "result",
    'Aim_Citation': "aim",
    'Aim_citation': "aim",
    'Hypothesis_Citation': "hypothesis",
    'Implication_Citation': "implication",
    'Method Citation': "method",
    'Method citation': "method",
    'Method citation |': "method",
    'Method_CItation': "method",
    'Method_Citation': "method",
    'Method_Citation |': "method",
    'Result Citation': "result",
    'Result_Citation': "result",
    'Results_Citation': "result",
    'Result_citation': "result",
    "method_Citation": "method",
    "Method_citation": "method"
}
def load_citation_data_as_df(directory, is_test=False):
    data = []
    citation_list = os.listdir(directory)
    citation_list = sorted(citation_list)
    for filename in citation_list:
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                citations = json_data.get("citations", [])

                for citation in citations:
                    labels = []
                    for label in citation["label"]:
                        labels.append(nom_label_dict[label])

                    if is_test:
                        if len(labels)==1:
                            pass
                        else:
                            labels = [labels]

                    for label in labels:
                        refer_id = citation.get("refer_ID", "")
                        refer_sids = citation.get("refer_sids", "")
                        refer_text = citation.get("refer_text", "")
                        cite_id = citation.get("cite_ID", "")
                        cite_sids = citation.get("cite_sids", "")
                        cite_text = citation.get("cite_text", "")
                        cite_maker_sids = citation.get("cite_maker_sids", "")
                        number = citation.get("Number", "")
                        data.append(
                            {"refer_id": refer_id,
                             "cite_id": cite_id,
                             "refer_sids": refer_sids,
                             "refer_text": refer_text,
                             "cite_sids":cite_sids,
                             "cite_text": cite_text,
                             "cite_maker_sids": cite_maker_sids,
                             "number": number,
                             "label": label})
    data = pd.DataFrame(data)
    return data





