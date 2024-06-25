from extractor import DataExtractor
from tokenizer import TextTokenizer
from dataset import NERDataset
from trainer import ModelTrainer
from utils import load_label_to_id

# Folder paths
train_folder = './data/train'
val_folder = './data/val'
test_folder = './data/test'
label_to_id_path = 'data/label_to_id.json'
output_dir = './model/results'

# Load label to id mapping
label_to_id = load_label_to_id(label_to_id_path)

# Initialize tokenizer
tokenizer = TextTokenizer()

# Extract data
extractor = DataExtractor()
train_texts = list(extractor.extract_text_from_all_txt(train_folder).values())
train_annotations = list(extractor.extract_annotations_from_output(train_folder).values())
val_texts = list(extractor.extract_text_from_all_txt(val_folder).values())
val_annotations = list(extractor.extract_annotations_from_output(val_folder).values())
test_texts = list(extractor.extract_text_from_all_txt(test_folder).values())
test_annotations = list(extractor.extract_annotations_from_output(test_folder).values())

# Create datasets
train_dataset = NERDataset(train_texts, train_annotations, tokenizer, label_to_id, max_length=512)
val_dataset = NERDataset(val_texts, val_annotations, tokenizer, label_to_id, max_length=512)
test_dataset = NERDataset(test_texts, test_annotations, tokenizer, label_to_id, max_length=512)

# Train model
trainer = ModelTrainer(label_to_id, output_dir=output_dir)
trainer.train(train_dataset, val_dataset)

# Evaluate model
eval_results = trainer.evaluate(test_dataset)
print(eval_results)

# Confusion matrix and incorrect predictions
eval_pred = trainer.get_predictions_and_labels(trainer, test_dataset)
trainer.compute_confusion_matrix(eval_pred, test_dataset, tokenizer)

predictions, labels = eval_pred
incorrect_predictions = trainer.display_incorrect_predictions(predictions, labels, test_dataset, tokenizer)

for line in incorrect_predictions:
    print(line)
