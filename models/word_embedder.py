import numpy as np

class WordEmbedder:

    def __init__(self, file: str, delim: str = ' '):
        print("Loading word embeddings...")
        self.embeddings = load_embeddings(file, delim)
        self.dimension = len(list(self.embeddings.values())[0])
        for embedding in self.embeddings.values():
            assert len(embedding) == self.dimension

    def embed_word(self, word: str) -> list[float]:
        return self.embeddings[word.lower()]\
            if word.lower() in self.embeddings else\
            [0.0] * self.dimension

    def embed_sentence(self, sentence: str, length: int = -1) -> list[list[float]]:
        words = sentence.split(' ')
        length = length if length != -1 else len(words)
        words = words[:length] + ([''] * (length - len(words)))

        #return [[self.embed_word(word)[i]
        #    for word in words]
        #    for i in range(self.dimension)]

        return np.transpose([self.embed_word(word) for word in words])

    def embed_dataset(self, dataset: list[str], length: int = -1) -> list[list[list[float]]]:
        length = length if length != -1 else\
            max(len(sentence.split(' ')) for sentence in dataset)
        return [self.embed_sentence(sentence, length) for sentence in dataset]

def load_embeddings(file: str, delim: str) -> dict[list[float]]:
    file = open(file, 'r', encoding='utf-8')
    lines = file.read().split('\n')
    embeddings = {
        line.split(delim)[0].lower():
            [float(d) for d in line.split(delim)[1:]]
        for line in lines
        if line.strip() != ''
    }
    file.close()
    return embeddings