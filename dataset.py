import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, texts, annotations, tokenizer, label_to_id, max_length=512):
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.data = self.process_data(texts, annotations)

    def process_data(self, texts, annotations):
        data = []
        for text, annotation in zip(texts, annotations):
            lines = text.split('\n')
            file_tokens = []

            for line_num, line in enumerate(lines):
                if line.strip():
                    tokenized_text = self.tokenizer.tokenize_text(line)
                    tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'])
                    offsets = tokenized_text['offset_mapping']

                    line_tokens = []
                    for idx in range(len(tokens)):
                        if tokens[idx] not in [",", ".", "!", "?", ";", ":", "(", ")"]:
                            start_char, end_char = offsets[idx]
                            label_id = self.find_label_id(start_char, end_char - 1, line_num, annotation)
                            if label_id in self.label_to_id.values():
                                token_info = {
                                    'input_id': tokenized_text['input_ids'][idx:idx + 1],
                                    'attention_mask': [1],
                                    'label_id': [label_id]
                                }
                                line_tokens.append(token_info)

                                if idx + 1 < len(tokens) and tokens[idx + 1] not in [",", ".", "!", "?", ";", ":", "(",
                                                                                     ")"]:
                                    start_char_next, end_char_next = offsets[idx + 1]
                                    combined_label_id = self.find_label_id(start_char, end_char_next - 1, line_num,
                                                                           annotation)
                                    if combined_label_id != self.label_to_id['O']:
                                        combined_info = {
                                            'input_id': tokenized_text['input_ids'][idx:idx + 2],
                                            'attention_mask': [1, 1],
                                            'label_id': [combined_label_id, combined_label_id]
                                        }
                                        line_tokens.append(combined_info)

                    file_tokens.extend(line_tokens)
            data.append(file_tokens)
        return data

    def find_label_id(self, start, end, line_num, annotations):
        valid_labels = [ann['label'] for ann in annotations if
                        ann['line_num'] == line_num and start >= ann['start_char'] and end <= ann['end_char']]
        valid_labels = [label for label in valid_labels if label in self.label_to_id]
        if valid_labels:
            return self.label_to_id[min(valid_labels, key=lambda label: self.label_to_id[label])]
        return self.label_to_id['O']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids = torch.cat([torch.tensor(info['input_id']) for info in item])
        attention_masks = torch.cat([torch.tensor(info['attention_mask']) for info in item])
        labels = torch.cat([torch.tensor(info['label_id']) for info in item])

        input_ids = torch.nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]),
                                            value=self.tokenizer.tokenizer.pad_token_id)
        attention_masks = torch.nn.functional.pad(attention_masks, (0, self.max_length - attention_masks.shape[0]),
                                                  value=0)
        labels = torch.nn.functional.pad(labels, (0, self.max_length - labels.shape[0]), value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
