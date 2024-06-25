import torch
from transformers import CamembertConfig, CamembertForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_metric


class ModelTrainer:
    def __init__(self, label_to_id, model_name='camembert-base', output_dir='./results', num_labels=None):
        self.label_to_id = label_to_id
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_labels = num_labels if num_labels else len(label_to_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        config = CamembertConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
        model = CamembertForTokenClassification.from_pretrained(self.model_name, config=config)
        model.to(self.device)
        return model

    def train(self, train_dataset, val_dataset, num_epochs=25, batch_size=16, logging_steps=10):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=logging_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        self.save_model()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    @staticmethod
    def compute_metrics(eval_pred):
        accuracy_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        f1_metric = load_metric("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        predictions = predictions.flatten()
        labels = labels.flatten()

        indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        true_predictions = [pred for pred, label in zip(predictions, labels) if label in indexs]
        true_labels = [label for label in labels if label in indexs]

        return {
            "accuracy": accuracy_metric.compute(predictions=true_predictions, references=true_labels),
            "precision": precision_metric.compute(predictions=true_predictions, references=true_labels,
                                                  average="macro"),
            "recall": recall_metric.compute(predictions=true_predictions, references=true_labels, average="macro"),
            "f1": f1_metric.compute(predictions=true_predictions, references=true_labels, average="macro")
        }

    def evaluate(self, test_dataset):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=16,
            do_train=False,
            do_eval=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            eval_dataset=test_dataset
        )

        eval_results = trainer.evaluate()
        return eval_results

    def compute_confusion_matrix(self, eval_pred, dataset, tokenizer):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        predictions = predictions.flatten()
        labels = labels.flatten()

        ignored_index = -100
        true_predictions = [pred for pred, label in zip(predictions, labels) if label != ignored_index]
        true_labels = [label for label in labels if label != ignored_index]

        unique_labels = sorted(set(true_labels))
        cm = confusion_matrix(true_labels, true_predictions, labels=unique_labels)
        report = classification_report(true_labels, true_predictions, target_names=[str(i) for i in unique_labels])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        print(cm)
        print(report)

        return {"confusion_matrix": cm}

    def display_incorrect_predictions(self, predictions, labels, dataset, tokenizer):
        idx_to_label = {v: k for k, v in self.label_to_id.items()}
        incorrect_predictions = []

        for i in range(len(predictions)):
            input_ids = dataset[i]['input_ids']
            for j in range(len(predictions[i])):
                if labels[i][j] != -100 and labels[i][j] != predictions[i][j]:
                    term = tokenizer.decode([input_ids[j]])
                    true_label = idx_to_label[labels[i][j]]
                    pred_label = idx_to_label[predictions[i][j]]
                    incorrect_predictions.append(f"{term}\t{true_label}\t{pred_label}")

        return incorrect_predictions

    def get_predictions_and_labels(self, trainer, dataset):
        predictions, labels, _ = trainer.predict(dataset)
        predictions = np.argmax(predictions, axis=-1)
        return predictions, labels
