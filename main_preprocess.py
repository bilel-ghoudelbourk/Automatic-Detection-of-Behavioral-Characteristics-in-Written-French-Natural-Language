import os
from sklearn.model_selection import train_test_split
from extractor import DataExtractor
from utils import save_label_to_id

# Folder paths
data_folder = 'txt'
train_folder = './data/train'
val_folder = './data/val'
test_folder = './data/test'
label_to_id_path = 'data/label_to_id.json'

# Extract data
extractor = DataExtractor()
all_texts = extractor.extract_text_from_all_txt(data_folder)
all_annotations = extractor.extract_annotations_from_output(data_folder)

# Split data
keys1 = list(all_texts.keys())
keys2 = list(all_annotations.keys())
texts = [all_texts[key] for key in keys1]
annotations = [all_annotations[key] for key in keys2]

train_texts, temp_texts, train_annotations, temp_annotations = train_test_split(
    texts, annotations, test_size=0.3, random_state=42)
val_texts, test_texts, val_annotations, test_annotations = train_test_split(
    temp_texts, temp_annotations, test_size=0.5, random_state=42)

# Create directories if not exist
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Save data
def save_data(dataset, annotations, folder):
    for i, (text, annotation) in enumerate(zip(dataset, annotations)):
        text_filename = os.path.join(folder, f'text_{i}.txt')
        with open(text_filename, 'w', encoding='utf-8') as file:
            file.write(text)

        annotation_filename = os.path.join(folder, f'annotations_{i}.output')
        with open(annotation_filename, 'w', encoding='utf-8') as file:
            for annot in annotation:
                file.write(f"{annot['label']}\t{annot['term']}\t{annot['line_num']}\t{annot['start_word']}\t{annot['end_word']}\t{annot['start_char']}\t{annot['end_char']}\n")

save_data(train_texts, train_annotations, train_folder)
save_data(val_texts, val_annotations, val_folder)
save_data(test_texts, test_annotations, test_folder)

# Generate and save label to id mapping
priority_labels = [f"_sk{i}" for i in range(1, 17)]
other_labels = set(annotation['label'] for annotation_list in all_annotations.values() for annotation in annotation_list if annotation['label'] not in priority_labels)
all_labels = priority_labels + sorted(other_labels)
all_labels.append('O')  # Add 'O' for non-annotated tokens

label_to_id = {label: idx for idx, label in enumerate(all_labels)}
save_label_to_id(label_to_id, label_to_id_path)
