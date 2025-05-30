import numpy as np

from utils import *
from config import *
import warnings
import  pickle
from tqdm import tqdm
from bs4 import GuessedAtParserWarning, BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings('ignore', category=GuessedAtParserWarning)
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)



def save_sample(sample, path, is_test=False, name=None):
    if is_test:
        file_path = path + "/{}.json".format(name)
    else:
        file_path = path + "/{}.json".format(sample['ID'])
    save_json(track="save_sample", data=sample, path=file_path)

def save_document_json(document, ID, is_test=False):
    if is_test:
        file_path = ROOT_DIR + "/source/test_documents" + "/{}.json".format(ID)
    else:
        file_path = ROOT_DIR + "/source/documents" + "/{}.json".format(ID)
    save_json(track="save_document_json", data=document, path=file_path)

def get_document_xml(source, ID):
    def read_file(source):
        f = open(source, "rb")
        byte_ = f.read()
        text = BeautifulSoup(byte_)
        f.close()
        return text
    try:
        text = read_file(source)
    except:
        try:
            source = source.replace(".txt", ".xml")
            text = read_file(source)
        except:
            source = source + ".xml"
            text = read_file(source)

    document = dict()
    document['ID'] = ID
    document['sections'] = []

    section = dict()
    section["text"] = "abstract"
    section["number"] = 0
    section["sents"] = []

    for tag in text.find_all(True):
        if tag.name in ["html", "body", "paper"]:
            continue
        elif tag.name.lower() == "abstract":
            if section["text"] == "abstract":
                continue
            else:
                print("ABSTRACT NOT IN BEGINING")
        elif tag.name.lower() == "section":
            if len(section["sents"])>0: document['sections'].append(section)
            section = dict()
            try: section["text"] = remove_multi_space_from_string(tag.attrs["title"]).lower()
            except: section["text"] = ""
            try: section["number"] = remove_multi_space_from_string(tag.attrs["number"]).lower()
            except: section["number"] = ""
            section["sents"] = []

        elif tag.name.lower() == "s":
            sent = dict()
            sent["text"] = tag.getText()
            try:
                sent["sid"] = int(tag.attrs["sid"])
            except:
                sent["sid"] = None
                print("Error in load sid from", source)

            try:
                sent["ssid"] = int(tag.attrs["ssid"])
            except:
                sent["ssid"] = None
                if DEBUG:
                    print(f"Error in load ssid with {sent['sid']} in {source}", source)

            sent["kind_of_tag"] = "s"
            section["sents"].append(sent)

        elif tag.name.lower() == "subsection":
            sent = dict()
            sent["text"] = tag.getText()
            sent["sid"] = ""
            sent["ssid"] = ""
            sent["kind_of_tag"] = "subsection"
        else:
            if DEBUG:
                print("Find a noise tag:", tag, "from", source)

    if tag.name.lower() != "section": document['sections'].append(section)
    return document


def get_text(s):
    return BeautifulSoup(s).find("html").getText()



# Preprocessing for train
if CREATE_TRAIN:
    with open(os.path.join(ROOT_DIR, 'source/cl_sum.pkl'), 'rb') as file:
        clcite_data = pickle.load(file)

    for ID in tqdm(clcite_data):
        doc = clcite_data[ID]
        if DEBUG:
            print(ID)

        refer_doc = get_document_xml(ROOT_DIR + "/source/raw_data/scisumm-corpus/Training-Set-2019/Task1/From-Training-Set-2018/{}/Reference_XML/".format(ID) + doc['citing_sent_list'][0]['Reference Article'], ID)
        save_document_json(refer_doc, ID)

        sample = dict()
        sample["ID"] = ID
        sample["citations"] = []

        for sent in doc['citing_sent_list']:
            cite = dict()
            cite['Number'] = int(sent['Citance Number'])
            cite['refer_ID'] = ID
            cite['refer_sids'] = [int(i) for i in re.findall(r'\d+', sent['Reference Offset'])]
            cite['refer_text'] = get_text(sent['Reference Text'])
            cite['cite_ID'] = sent['Citing Article'].split(".")[0]
            cite['cite_maker_sids'] = [int(i) for i in re.findall(r'\d+', sent['Citation Marker Offset'])]
            cite['cite_sids'] = [int(i) for i in re.findall(r'\d+', sent['Citation Offset'])]
            cite['cite_text'] = get_text(sent['Citation Text'])

            cite_doc = get_document_xml(
                ROOT_DIR + "/source/raw_data/scisumm-corpus/Training-Set-2019/Task1/From-Training-Set-2018/{}/Citance_XML/".format(
                    ID) + sent['Citing Article'], cite['cite_ID'])
            save_document_json(cite_doc, cite['cite_ID'])

            cite['label'] = []
            for label in sent['Discourse Facet'].split(","):
                if label != "":
                    cite['label'].append(label.replace("'", "").replace("[","").replace("]",""))
            sample['citations'].append(cite)
        save_sample(sample=sample, path=ROOT_DIR + "/source/citations")


# Preprocessing for test
if CREATE_TEST:
    test_files = os.listdir(ROOT_DIR + "/source/raw_data/scisumm-corpus/Test-Set-2018-Gold/Task1")

    for file in test_files:
        data = pd.read_csv(ROOT_DIR + "/source/raw_data/scisumm-corpus/Test-Set-2018-Gold/Task1/" + file)

        if DEBUG:
            print(file)

        refer_ID = file.split("_")[0]
        refer_doc = get_document_xml(ROOT_DIR + "/source/raw_data/scisumm-corpus/Test-Set-2018/{}/Reference_XML/{}.xml".format(refer_ID, refer_ID), refer_ID)
        save_document_json(refer_doc, refer_ID, is_test=True)

        sample = dict()
        sample["ID"] = refer_ID
        sample["citations"] = []

        for index, row in tqdm(data.iterrows()):
            if str(row["Discourse Facet"]) == "nan" :
                continue

            cite = dict()
            cite['Number'] = int(row['Citance Number'])
            cite['refer_ID'] = refer_ID
            try:
                cite['refer_sids'] = [int(i) for i in re.findall(r'\d+', row['Reference Offset'])]
            except:
                cite['refer_sids'] = [row['Reference Offset']]

            cite['refer_text'] = get_text(row['Reference Text'])
            cite['cite_ID'] = row['Citing Article'].split(".")[0]
            cite['cite_maker_sids'] = [0]
            cite['cite_sids'] = [0]
            cite['cite_text'] = get_text(row['Citation Text'])

            if cite['cite_ID'] not in MISSING_DOCUMENTS:
                cite_doc = get_document_xml(ROOT_DIR + "/source/raw_data/scisumm-corpus/Test-Set-2018/{}/Citance_XML/{}.xml".format(refer_ID, cite['cite_ID']), cite['cite_ID'])
                save_document_json(cite_doc, cite['cite_ID'], is_test=True)

            cite['label'] = []
            for label in row['Discourse Facet'].split(","):
                if label != "":
                    cite['label'].append(label.replace("'", "").replace("[","").replace("]",""))

            sample['citations'].append(cite)

        save_sample(sample=sample, path=ROOT_DIR + "/source/test_citations", is_test=True, name=file.split(".")[0])
