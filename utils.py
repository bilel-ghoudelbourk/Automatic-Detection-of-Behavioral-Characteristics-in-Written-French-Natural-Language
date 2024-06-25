import json

def load_label_to_id(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        label_to_id = json.load(file)
    return label_to_id

def save_label_to_id(label_to_id, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(label_to_id, file, ensure_ascii=False, indent=4)
