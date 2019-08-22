from gensim.models import Word2Vec
import logging
from gensim.models import KeyedVectors
logging.basicConfig(filename="embed.log", level=logging.DEBUG)
from gensim.models.word2vec import PathLineSentences
import copy
import numpy as np
import pickle
from tqdm import tqdm

class EmbeddingsLearner(object):

    def __init__(self, path_to_docs, pretrained_model = None):
        if(pretrained_model):
            self.load_pretrained(pretrained_model)
        else:
            self.sentences = PathLineSentences(path_to_docs)
            self.model = Word2Vec(self.sentences, size=150, window=10, min_count=1, workers=20)
            self.word_vectors_all = []
            self.word_vectors_kb = []

    def load_pretrained(self, pretrained_path):
        self.model = Word2Vec.load(pretrained_path)
        self.word_vectors_all = self.model.wv

    def train(self, ep=10):
        self.model.train(self.sentences, total_examples=self.model.corpus_count, epochs=ep)
        self.word_vectors_all = self.model.wv
        print('Total vocabulary size: ', len(self.word_vectors_all.vocab))
        return self.model

    def save(self, model_path, kb_vecs_file, all_vocab_file):

        self.model.save(model_path)
        self.save_vectors(kb_vecs_file, all_vocab_file)

    def save_vectors(self, kb_vecs_file, all_vocab_file):
        # pickle.dump(self.word_vectors_kb, open(kb_vecs_file, "wb"), protocol=4) # protocol four to be able to handle large data
        # pickle.dump(self.word_vectors_all, open(all_vocab_file, "wb"), protocol=4)

        self.word_vectors_kb.save(kb_vecs_file)
        self.word_vectors_all.save(all_vocab_file)

    def restrict_w2v(self, restricted_word_set):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []
        w2v_original = self.word_vectors_all
        self.model.wv.most_similar(positive='canada') # for some reason without this line the norms are all None objects
        w2v = copy.deepcopy(w2v_original)

        for i in range(len(w2v.vocab)):
            word = w2v.index2entity[i]
            vec = w2v.vectors[i]
            vocab = w2v.vocab[word]
            vec_norm = w2v.vectors_norm[i]
            if word in restricted_word_set:
                vocab.index = len(new_index2entity)
                new_index2entity.append(word)
                new_vocab[word] = vocab
                new_vectors.append(vec)
                new_vectors_norm.append(vec_norm)

        # CONVERT THEM TO NUMPY ARRAYS !!!
        w2v.vocab = new_vocab
        w2v.vectors = np.array(new_vectors)
        w2v.index2entity = np.array(new_index2entity)
        w2v.index2word = np.array(new_index2entity)
        w2v.vectors_norm = np.array(new_vectors_norm)

        self.word_vectors_kb = w2v
        return w2v

    def split_entities_from_vocab_vectors(self, restricted_word_set):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []
        w2v_original = self.word_vectors_all
        self.model.wv.most_similar(
            positive='has')  # for some reason without this line the norms are all None objects
        w2v = copy.deepcopy(w2v_original)
        count_not_found = 0

        for entity in tqdm(restricted_word_set):
            if w2v.vocab.get(entity):
                index = w2v.vocab.get(entity).index
                vec = w2v.vectors[index]
                vocab = w2v.vocab[entity]
                vec_norm = w2v.vectors_norm[index]
                vocab.index = len(new_index2entity)
                new_index2entity.append(entity)
                new_vocab[entity] = vocab
                new_vectors.append(vec)
                new_vectors_norm.append(vec_norm)
            else:
                count_not_found += 1
                #print('Not found entity: ', entity)
                with open('log_embed_split.txt', 'a') as the_file:
                    the_file.write('\n')
                    the_file.write('Not found ' + entity + '\n')
                continue

        # CONVERT THEM TO NUMPY ARRAYS !!!
        w2v.vocab = new_vocab
        w2v.vectors = np.array(new_vectors)
        w2v.index2entity = np.array(new_index2entity)
        w2v.index2word = np.array(new_index2entity)
        w2v.vectors_norm = np.array(new_vectors_norm)

        self.word_vectors_kb = w2v
        print('KB vecs: ', len(new_vectors))
        print('Not found kb vectors: ', count_not_found)
        return w2v


if __name__ == '__main__':
    embed = EmbeddingsLearner('../preprocessing/preprocessed/conll/test_preprocessed_conll.txt')
    # model = embed.train(10)
    # print('Model initialized and trained.')
    #
    # with open("wiki_entities.txt", "rb") as fp:  # Entities ids gathered during preprocessing.
    #     kb_entities = pickle.load(fp)
    #
    # embed.split_entities_from_vocab_vectors(kb_entities)
    # embed.save('models_data/word2vec_test.model', 'kb_vecs_test_pickle', 'all_vocab_vecs_test_pickle')
    #
    # print('Model trained and saved.')

    # print('Loading pretrained vectors...')
    # kb_vectors_loaded = KeyedVectors.load("kb_vecs_test_pickle.kv", mmap='r')
    # print(kb_vectors_loaded.vocab)
    # kb_vectors_loaded.similar_by_vector('germany')