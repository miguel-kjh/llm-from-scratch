import os
import re

from utils import PETRAIN_DATA_FOLDER

class Tokenizer:

    def create_vocab(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

class SimpleTokenizer(Tokenizer):

    def __init__(self, dataset):
        self.dataset = dataset
        self.regular_expression = r'([,.:;?_!"()\']|--|\s)'

        self.str_to_int = self.create_vocab()
        self.int_to_str = {i: word for word, i in self.str_to_int.items()}

    def tokenize(self, text: str) -> list:
        preprocessed = re.split(self.regular_expression, text)
        return [item.strip() for item in preprocessed if item.strip()]


    def create_vocab(self) -> dict:
        tokens = self.tokenize(self.dataset)
        all_tokens = sorted(set(tokens))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        return {word: i for i, word in enumerate(all_tokens)}
    
    def encode(self, text: str) -> list:
        tokens = self.tokenize(text)
        tokens = [token if token in self.str_to_int else "<|unk|>" for token in tokens]
        return [self.str_to_int[token] for token in tokens]
    
    def decode(self, tokens_ids: list) -> str:
        text = " ".join([self.int_to_str[i] for i in tokens_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text



# main
if __name__ == '__main__':
    
    FILE_PATH = os.path.join(PETRAIN_DATA_FOLDER, "the-verdict.txt")

    with open(FILE_PATH, "r") as f:
        dataset = f.read()

    tokenizer = SimpleTokenizer(dataset)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
    
