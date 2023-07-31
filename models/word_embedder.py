import numpy as np

class WordEmbedder:
    """
    A mechanism for loading pre-generated word vectors from a file,
    and using them to embed words and sentences in a latent space.
    """

    def __init__(self, file: str, delim: str = ' '):
        """
        Load a set of latent word embeddings from a file.
        @param file: The name of the file to load word vectors from.
        @param delim: The character used in the file to separate tokens.
        """

        print(f"Loading word embeddings from {file}.")

        self.embeddings = load_embeddings(file, delim)

        # Ensure that all embeddings have the same dimensionality.
        self.dimension = len(list(self.embeddings.values())[0])
        for embedding in self.embeddings.values():
            assert len(embedding) == self.dimension

        # Load the list of words to skip.
        self.stopwords = load_stopwords('data/stopwords.txt')

    def embed_word(self, word: str) -> list[float]:
        """
        Embed a single word in a latent space
        If the word isn't in the dictionary, an all-zeroes vector is provided.
        @param word: The word to be embedded.
        @return: A latent representation of the given word.
        """
        return self.embeddings[word.lower()]\
            if word.lower() in self.embeddings else\
            [0.0] * self.dimension

    def embed_sentence(self, sentence: str, length: int = -1, border: int = 0) -> list[list[float]]:
        """
        Embed an entire sentence in a latent space.
        Individual words are assumed to be separated by spaces.
        A matrix of embeddings is created,
        in transposed form for compatibility with PyTorch convolutional layers.
        @param sentence: The sentence to be embedded.
        @param length: The number of words in a sentence (default: unconstrained).
        @param border: The number of empty words to include on each side.
        @return: A matrix of latent representations of each word.
        """
        words = [word.lower()
            for word in sentence.split(' ')
            if word.lower() not in self.stopwords]

        # Determine the number of words in a sentence.
        length = length if length != -1 else len(words)
        # Pad/trim the sentence to/from the required length.
        words = ([''] * border) + words[:length]\
            + ([''] * (length - len(words))) + ([''] * border)
        # Find the embedding corresponding to each word.
        embeddings = [self.embed_word(word) for word in words]
        # Transpose the embeddings for PyTorch compatibility.
        return np.transpose(embeddings)

    def embed_dataset(self, dataset: list[str], length: int = -1, border: int = 0) -> list[list[list[float]]]:
        """
        Embed a collection of sentences in a latent space.
        @param dataset: The collection of sentences to be embedded.
        @param length: The number of words in a sentence (default: maximum of any given).
        @param border: The number of empty words to include on each side.
        @return: A tensor of latent representations of each sentence.
        """
        # Determine the number of words in each sentence.
        length = length if length != -1 else\
            max(len(sentence.split(' ')) for sentence in dataset)
        # Embed each sentence in the dataset.
        return [self.embed_sentence(sentence, length, border) for sentence in dataset]

def load_embeddings(file: str, delim: str) -> dict[list[float]]:
    # It is necessary to specify a UTF-8 encoding because some words have special characters.
    file = open(file, 'r', encoding='utf-8')
    lines = file.read().split('\n')
    embeddings = {
        # The first token on each line is the word itself.
        line.split(delim)[0].lower():
            # All subsequent tokens are the latent representation.
            [float(d) for d in line.split(delim)[1:]]
        for line in lines
        # Ignore empty lines.
        if line.strip() != ''
    }
    file.close()
    return embeddings

def load_stopwords(file: str, delim: str = '\n') -> set[str]:
    file = open(file, 'r', encoding='utf-8')
    stopwords = set([word.lower() for word in file.read().split(delim)])
    file.close()
    return stopwords