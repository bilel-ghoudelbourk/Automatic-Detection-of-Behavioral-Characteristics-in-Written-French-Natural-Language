import os
import pandas as pd
import json

class DataExtractor:
    @staticmethod
    def extract_text_from_all_txt(folder_path):
        texts = {}
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        txt_files.sort()

        for filename in txt_files:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts[filename] = file.read()
        return texts

    @staticmethod
    def extract_annotations_from_output(folder_path):
        annotations = {}
        output_files = [f for f in os.listdir(folder_path) if f.endswith(".output")]
        output_files.sort()

        for filename in output_files:
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None,
                               names=['label', 'term', 'line_num', 'start_word', 'end_word', 'start_char', 'end_char'])
            annotations[filename] = data.to_dict('records')
        return annotations

    @staticmethod
    def load_label_to_id(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            label_to_id = json.load(file)
        return label_to_id

    @staticmethod
    def save_label_to_id(label_to_id, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(label_to_id, file, ensure_ascii=False, indent=4)
