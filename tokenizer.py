from transformers import CamembertTokenizerFast

class TextTokenizer:
    def __init__(self, model_name='camembert-base'):
        self.tokenizer = CamembertTokenizerFast.from_pretrained(model_name)

    def tokenize_text(self, text):
        return self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    
    def decode_token(self, token_id):
        return self.tokenizer.decode(token_id)
