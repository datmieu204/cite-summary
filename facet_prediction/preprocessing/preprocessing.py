from text_encoding import *
from facet_prediction.utils import *
from pos_encoding import pos_avg_ratio_caculating
from tqdm import  tqdm

documents = read_all_docs()
test_documents = read_all_docs(is_test=True)


def preprocess_text(text):
    text = remove_multi_space_from_string(text)
    text = text.lower()
    return text

def preprocess_df(df):
    df["refer_text"] = df["refer_text"].apply(preprocess_text)
    df["cite_text"] = df["cite_text"].apply(preprocess_text)
    return df

def create_pos_features(df, is_test=False):
    print("create_pos_features")
    for index, row in tqdm(df.iterrows()):
        refer_id = row['refer_id']
        cite_id = row['cite_id']
        refer_sids = row['refer_sids']
        cite_sids = row['cite_sids']

        if is_test:
            refer_doc = test_documents.get(refer_id)
            cite_doc = test_documents.get(cite_id)
        else:
            refer_doc = documents.get(refer_id)
            cite_doc = documents.get(cite_id)

        if not refer_doc:
            if refer_id in MISSING_DOCUMENTS:
                pass
            else:
                print(f"ERROR when find refer_doc and cite_doc in sample {index}, ID {refer_id}")
        else:
            refer_sec_poss, refer_sec_pos_avg, refer_sec_pos_ratio_avg, refer_ssid_avg, refer_ssid_ratio_avg, refer_sid_avg, refer_sid_ratio_avg = pos_avg_ratio_caculating(refer_sids, refer_doc)
            df.at[index, "refer_sec_poss"] = "_".join([str(val) for val in refer_sec_poss])
            df.at[index, "refer_sec_pos_avg"] = refer_sec_pos_avg
            df.at[index, "refer_sec_pos_avg_nom"] = refer_sec_pos_ratio_avg
            df.at[index, "refer_sid_avg"] = refer_sid_avg
            df.at[index, "refer_sid_avg_nom"] = refer_sid_ratio_avg
            df.at[index, "refer_ssid_avg"] = refer_ssid_avg
            df.at[index, "refer_ssid_avg_nom"] = refer_ssid_ratio_avg

        if not cite_doc:
            if cite_id in MISSING_DOCUMENTS:
                pass
            else:
                print(f"ERROR when find refer_doc and cite_doc in sample {index}, ID {cite_id}")
        else:
            cite_sec_poss, cite_sec_pos_avg, cite_sec_pos_ratio_avg, cite_ssid_avg, cite_ssid_ratio_avg, cite_sid_avg, cite_sid_ratio_avg = pos_avg_ratio_caculating(cite_sids, cite_doc)
            df.at[index, "cite_sec_poss"] = "_".join([str(val) for val in cite_sec_poss])
            df.at[index, "cite_sec_pos_avg_nom"] = cite_sec_pos_ratio_avg
            df.at[index, "cite_sid_avg"] = cite_sid_avg
            df.at[index, "cite_sid_avg_nom"] = cite_sid_ratio_avg
            df.at[index, "cite_ssid_avg"] = cite_ssid_avg
            df.at[index, "cite_ssid_avg_nom"] = cite_ssid_ratio_avg
    return df

def create_section_text_features(df, is_test=False):
    for index, row in tqdm(df.iterrows()):
        refer_id = row['refer_id']
        cite_id = row['cite_id']
        refer_sids = row['refer_sids']
        cite_sids = row['cite_sids']

        if is_test:
            refer_doc = test_documents.get(refer_id)
            cite_doc = test_documents.get(cite_id)
        else:
            refer_doc = documents.get(refer_id)
            cite_doc = documents.get(cite_id)

        if not refer_doc:
            if refer_id in MISSING_DOCUMENTS: pass
            else: print(f"ERROR when find refer_doc and cite_doc in sample {index}, ID {refer_id}")
        else:
            refer_sections = get_all_section_title_by_sids(refer_doc, refer_sids)
            df.at[index, "refer_sections"] = ", ".join(refer_sections)

        if not cite_doc:
            if cite_id in MISSING_DOCUMENTS: pass
            else: print(f"ERROR when find refer_doc and cite_doc in sample {index}, ID {cite_id}")
        else:
            cite_sections = get_all_section_title_by_sids(cite_doc, cite_sids)
            df.at[index, "cite_sections"] = ", ".join(cite_sections)
    return df

def text_embedding(df, test_df):
    refer_encoder, refer_text_encoded, test_refer_text_encoded = tfidf_for_dataframe(df, test_df, "refer_text")
    cite_encoder, cite_text_encoded, test_cite_text_encoded = tfidf_for_dataframe(df, test_df, "cite_text")
    label_encoder, labels, test_labels = label_encoder_for_dataframe(df, test_df, "label")

    refer_section_encoder, refer_section_encoded, test_refer_section_encoded = tfidf_for_dataframe(df, test_df, "refer_sections")
    cite_section_encoder, cite_section_encoded, test_cite_section_encoded = tfidf_for_dataframe(df, test_df, "cite_sections")

    refer_text_selection_model, refer_text_encoded, test_refer_text_encoded = get_important_features("refer_text_encoded", X_train=refer_text_encoded, y_train=labels, num=NUM_IMPORTANT_FEATURES_TEXT, X_test=test_refer_text_encoded)
    cite_text_selection_model, cite_text_encoded, test_cite_text_encoded = get_important_features("cite_text_encoded", X_train=cite_text_encoded, y_train=labels, num=NUM_IMPORTANT_FEATURES_TEXT, X_test=test_cite_text_encoded)
    refer_section_selection_model, refer_section_encoded, test_refer_section_encoded = get_important_features("refer_text_encoded", X_train=refer_section_encoded, y_train=labels, num=NUM_IMPORTANT_FEATURES_SECTION_TEXT, X_test=test_refer_section_encoded)
    cite_section_selection_model, cite_section_encoded, test_cite_section_encoded = get_important_features("cite_text_encoded", X_train=cite_section_encoded, y_train=labels, num=NUM_IMPORTANT_FEATURES_SECTION_TEXT, X_test=test_cite_section_encoded)

    refer_text_selected = refer_text_selection_model.get_feature_names_out(refer_encoder.get_feature_names_out())
    cite_text_selected = cite_text_selection_model.get_feature_names_out(cite_encoder.get_feature_names_out())

    refer_section_selected = refer_section_selection_model.get_feature_names_out(refer_section_encoder.get_feature_names_out())
    cite_section_selected = cite_section_selection_model.get_feature_names_out(cite_section_encoder.get_feature_names_out())

    for data, file_name in zip([refer_text_selected, cite_text_selected, refer_section_selected, cite_section_selected], ['refer_text_selected', 'cite_text_selected', 'refer_section_selected', 'cite_section_selected']):
        path = ROOT_DIR + "/facet_prediction/dataset/important_token_list/{}.json".format(file_name)
        arr = list(data)
        save_json("save_important_token_list", arr, path)

    df_dict = dict()
    for key in df.columns:
        df_dict[key] = df[key]
    df_dict['label'] = labels.tolist()
    df_dict['refer_text_tfidf'] = refer_text_encoded.toarray().tolist()
    df_dict['cite_text_tfidf'] = cite_text_encoded.toarray().tolist()
    df_dict['refer_section_encoded'] = refer_section_encoded.toarray().tolist()
    df_dict['cite_section_encoded'] = cite_section_encoded.toarray().tolist()
    df_encoded = pd.DataFrame(df_dict)

    test_dict = dict()
    for key in test_df.columns:
        test_dict[key] = test_df[key]
    test_dict['label'] = test_labels.tolist()
    test_dict['refer_text_tfidf'] = test_refer_text_encoded.toarray().tolist()
    test_dict['cite_text_tfidf'] = test_cite_text_encoded.toarray().tolist()
    test_dict['refer_section_encoded'] = test_refer_section_encoded.toarray().tolist()
    # test_dict['cite_section_encoded'] = cite_section_encoded.toarray().tolist()
    test_encoded = pd.DataFrame(test_dict)

    return df_encoded, test_encoded


if __name__ == '__main__':
    train_directory = ROOT_DIR +  "/source/citations"
    train_output_directory = ROOT_DIR + "/facet_prediction/dataset/preprocessed_data"
    train_data = load_citation_data_as_df(train_directory, is_test=False)
    train_preprocessed = preprocess_df(train_data)

    test_directory = ROOT_DIR + "/source/test_citations"
    test_output_directory = ROOT_DIR + "/facet_prediction/dataset/preprocessed_data"
    test_data = load_citation_data_as_df(test_directory, is_test=True)
    test_preprocessed = preprocess_df(test_data)

    # Encode features và labels
    train_preprocessed = create_pos_features(train_preprocessed)
    train_preprocessed = create_section_text_features(train_preprocessed)

    test_preprocessed = create_pos_features(test_preprocessed, is_test=True)
    test_preprocessed = create_section_text_features(test_preprocessed, is_test=True)

    train_preprocessed.to_csv(os.path.join(train_output_directory, "train_preprocessed_data.csv"), index=False)
    test_preprocessed.to_csv(os.path.join(test_output_directory, "test_preprocessed_data.csv"), index=False)

    train_encoded, test_encoded = text_embedding(train_preprocessed, test_preprocessed)

    train_encoded.to_csv(os.path.join(train_output_directory, "encoded_data.csv"), index=False)
    test_encoded.to_csv(os.path.join(test_output_directory, "test_encoded_data.csv"), index=False)

    data_json = train_encoded.to_dict()
    save_json("save_data", data_json, ROOT_DIR + "/facet_prediction/dataset/preprocessed_data/encoded_data.json")

    test_json = test_encoded.to_dict()
    save_json("save_data", test_json, ROOT_DIR + "/facet_prediction/dataset/preprocessed_data/test_encoded_data.json")




# def encode_to_ratio(df_encoded):
#     df_encoded['refer_text_ratio'] = df_encoded['refer_text_encoded'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
#     df_encoded['cite_text_ratio'] = df_encoded['cite_text_encoded'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
#     df_features = df_encoded[['refer_id', 'cite_id', 'refer_text_ratio', 'cite_text_ratio', 'label', 'number']]
#     df_features.to_csv(os.path.join(output_directory, "feature.csv"), index=False)
