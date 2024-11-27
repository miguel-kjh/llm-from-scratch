import os
import re
import tiktoken
import torch

from utils import PETRAIN_DATA_FOLDER
from data_prepare.GPTDataset import create_dataloader

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

def main1():
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

def main2():
    tokenizer = tiktoken.get_encoding('gpt2')

    text = "Veuillez résumer le texte suivant en quelques phrases en mettant en évidence les points les plus importants."
    integers = tokenizer.encode(text)
    print(integers, len(integers))
    for i in integers:
        print("token: ", tokenizer.decode([i]))
    print(tokenizer.decode(integers))
    

def main3():
    FILE_PATH = os.path.join(PETRAIN_DATA_FOLDER, "the-verdict.txt")

    with open(FILE_PATH, "r") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding('gpt2')
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    enc_sample = enc_text[50:]
    context_size = 4 # determna cuanto token se va a considerar en el contexto (X)
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:     {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

def main4():
    FILE_PATH = os.path.join(PETRAIN_DATA_FOLDER, "the-verdict.txt")

    with open(FILE_PATH, "r") as f:
        raw_text = f.read()

    dataloader = create_dataloader(
        raw_text, 
        max_length=4, 
        stride=4, 
        batch_size=8, 
        shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

def embeddings():
    input_ids = torch.tensor([0])
    vocab_size = 6
    embedding_dim = 3
    torch.manual_seed(0)
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    print(embedding_layer.weight)
    print(embedding_layer(input_ids))

    import numpy as np
    matriz = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

    # Matriz de máscara (vector columna para seleccionar la segunda fila)
    mascara = np.array([[0],  # No selecciona la primera fila
                        [1],  # Selecciona la segunda fila
                        [0]])  # No selecciona la tercera fila

    # Multiplicación matricial para obtener la fila seleccionada
    resultado = np.dot(mascara.T, matriz)
    print(resultado)

def positions():
    FILE_PATH = os.path.join(PETRAIN_DATA_FOLDER, "the-verdict.txt")

    with open(FILE_PATH, "r") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    token_embeddings = token_embedding_layer(inputs)
    print("\nToken embeddings shape:\n", token_embeddings.shape)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    

# main
if __name__ == '__main__':
    #main1()
    #main2()
    #main3()
    #main4()
    #embeddings()
    positions()