from gensim.models import KeyedVectors
from gensim.models import Word2Vec

class EmbeddingsModel(object):

    def __init__(self, out_dir):
        vocab_vecs = out_dir + '/embeddings/full_vocab_vectors.kv'
        kb_vecs = out_dir + '/embeddings/kb_vectors.kv'

        self.vocab_vectors = KeyedVectors.load(vocab_vecs, mmap='r')
        self.kb_vectors = KeyedVectors.load(kb_vecs, mmap='r')
        print('Embeddings model initialized! ')
        print('Vocab size: ', len(self.vocab_vectors.index2word))
        print('KB size: ',len(self.kb_vectors.index2word))


    def get_entity(self, mention, k=500, use_vector_avg=True):
        orig_mention = mention.lower()

        multi_word = (len(orig_mention.split(' ')) > 1)

        if (multi_word):
            mention = orig_mention.replace(' ', '_')
        else:
            mention = orig_mention

        mention = "kb_" + mention

        if mention in self.vocab_vectors:
            mention_vec = self.vocab_vectors[mention]
            predictions = self.kb_vectors.similar_by_vector(mention_vec, topn=k)
            #print(predictions)

            if len(predictions)>1:
                top_prediction = predictions[0][0]
                predictions_list = [pred[0] for pred in predictions]
            else:
                top_prediction = predictions[0]
                predictions_list = [top_prediction]

            return top_prediction, predictions
        else:
            if use_vector_avg:
                if multi_word:
                    words = orig_mention.split()
                    words_vectors = []
                    for w in words:
                        try:
                            words_vectors.append(self.vocab_vectors[w])
                        except:
                            continue
                    if words_vectors:
                        mention_vec = self.vector_avg(words_vectors)
                        predictions = self.kb_vectors.similar_by_vector(mention_vec, topn=k)
                        top_prediction = predictions[0][0]
                        return top_prediction, predictions
                    else:
                        return [], []
                else:
                    return [], []
            else:
                return [], []


    def vector_avg(self, components):
        return (sum(components)) / len(components)

if __name__ == '__main__':
    model = Word2Vec.load("full_embed_model/word2vec_full.model")
    model.most_similar(positive ='greece')
    # search = EmbeddingsModel()
    # predictions, predictions_list = search.get_entity('kb_bucharest')
    # print(predictions_list)
    # print(predictions)

