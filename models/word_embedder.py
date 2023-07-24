import numpy as np

from typing import List, Dict

from sqlitedict import SqliteDict

class WordEmbedder:
    """
    A mechanism for loading pre-generated word vectors from a file,
    and using them to embed words and sentences in a latent space.
    """

    def __init__(self, file: str, delim: str = ' ', db_backend=True, set_up=False):
        """
        Load a set of latent word embeddings from a file.
        @param file: The name of the file to load word vectors from.
        @param delim: The character used in the file to separate tokens.
        """
        print(f"Loading word embeddings from {file}.")
        # Load the embedding data from a file.

        if db_backend:
            self.embeddings = SqliteDict("/cluster/scratch/guphilip/embeddings.sqlite")

            if set_up:
                load_embeddings_sql(file, delim, self.embeddings)

        else:
            self.embeddings = load_embeddings(file, delim)
            # Determine the dimensionality of the embedding.
            
        self.dimension = len(next(self.embeddings.values()))
        # Ensure that all embeddings have the same dimensionality.
        
        #for embedding in self.embeddings.values():
        #    assert len(embedding) == self.dimension

    def embed_word(self, word: str) -> List[float]:
        """
        Embed a single word in a latent space
        If the word isn't in the dictionary, an all-zeroes vector is provided.
        @param word: The word to be embedded.
        @return: A latent representation of the given word.
        """
        return self.embeddings[word.lower()]\
            if word.lower() in self.embeddings else\
            [0.0] * self.dimension

    def embed_sentence(self, sentence: str, length: int = -1) -> List[List[float]]:
        """
        Embed an entire sentence in a latent space.
        Individual words are assumed to be separated by spaces.
        A matrix of embeddings is created,
        in transposed form for compatibility with PyTorch convolutional layers.
        @param sentence: The sentence to be embedded.
        @param length: The number of words in a sentence (default: unconstrained).
        @return: A matrix of latent representations of each word.
        """
        words = sentence.split(' ')
        # Determine the number of words in a sentence.
        length = length if length != -1 else len(words)
        # Pad/trim the sentence to/from the required length.
        words = words[:length] + ([''] * (length - len(words)))
        # Find the embedding corresponding to each word.
        embeddings = [self.embed_word(word) for word in words]
        # Transpose the embeddings for PyTorch compatibility.
        return np.transpose(embeddings)

    def embed_dataset(self, dataset: List[str], length: int = -1) -> List[List[List[float]]]:
        """
        Embed a collection of sentences in a latent space.
        @param dataset: The collection of sentences to be embedded.
        @param length: The number of words in a sentence (default: maximum of any given).
        @return: A tensor of latent representations of each sentence.
        """
        # Determine the number of words in each sentence.
        length = length if length != -1 else\
            max(len(sentence.split(' ')) for sentence in dataset)
        # Embed each sentence in the dataset.
        return [self.embed_sentence(sentence, length) for sentence in dataset]

def load_embeddings(file: str, delim: str) -> Dict[str, List[float]]:
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

def load_embeddings_sql(file: str, delim: str, embeddings_db: SqliteDict) -> Dict[str, List[float]]:

    # It is necessary to specify a UTF-8 encoding because some words have special characters.
    file = open(file, 'r', encoding='utf-8')
    
    #lines = file.read().split('\n')

    counter = 0
    for line in file:

        if line.strip() != '':
            token = line.split(delim)[0].lower()
            latent_repr = [float(d) for d in line.split(delim)[1:]]
            embeddings_db[token] = latent_repr

            counter += 1
            if counter % 100 == 0:
                embeddings_db.commit()
    
    file.close()

if __name__ == "__main__":

    embeddings_file = "/cluster/scratch/guphilip/glove.twitter.27B.200d.txt"
    WordEmbedder(embeddings_file, set_up=True)
    print("Finished")