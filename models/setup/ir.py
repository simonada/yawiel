import numpy as np
import collections
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords
from numpy.linalg import norm
from memory_profiler import profile
import time
import traceback

class InfoRetrievalSetup:
    """
        Index creation
            - dictionary: entities vocabulary from all entities
            - input: list of all KB entities
            - logic: iterate over all tokens contained in the entity, update dictionary
                        for each token, keep track of tokens already updated
            - output: dictionary (k,v) where k= token, v= list of entities
                        the token is in


    """

    def __init__(self, index_file_path= None):
        if index_file_path:
         self.loadIndex(index_file_path)

    def loadIndex(self, index_file):
        self.index = pickle.load(open(index_file, "rb"))
        print('Index initialized.')

    def build_index(self, entities_list):

        print('Constructing index...')
        entities_list = set(entities_list)
        token_entities_index = collections.defaultdict(list)
        # helper mapping structures
        # entities maps
        ids_to_entities = collections.defaultdict(list)
        entities_to_ids = collections.defaultdict(int)
        # vocab tokens maps
        ids_to_tokens = collections.defaultdict(list)
        tokens_to_ids = collections.defaultdict(int)

        if not isinstance(entities_list, list):
            entities_list = list(entities_list) # make sure we're considering only the unique cases

        for i in range(len(entities_list)):
            existing = set()
            entity_id = i
            entity_text = entities_list[i]
            ids_to_entities[entity_id] = entity_text
            entities_to_ids[entity_text] = entity_id
            entity_text = entity_text.split('_')

            if ('category:' in entity_text):
                continue

            for token in entity_text:
                if (token not in existing) and not (token == 'kb'): # idea is to consider only the first occurence of the term in the entity

                    token_entities_index[token].append(entity_id)
                    existing.add(token)

        # Build a token=>int map for the vocabulary and replace the token term with its id
        token_id = 0
        token_entities_index_numeric = collections.defaultdict(list)
        for token, posting_list in token_entities_index.items():
            tokens_to_ids[token] = token_id
            ids_to_tokens[token_id] = token
            token_entities_index_numeric[token_id] = posting_list
            token_id += 1

        self.index = token_entities_index_numeric
        self.tokens_to_ids = tokens_to_ids
        self.ids_to_tokens = ids_to_tokens
        self.ids_to_entities = ids_to_entities
        self.entities_to_ids = entities_to_ids
        self.N = len(entities_list)
        self.entities = entities_list

        print('Index length...', len(self.index))
        return token_entities_index_numeric, ids_to_entities, entities_to_ids, ids_to_tokens, tokens_to_ids

    def store_ir_datastructures(self, output_folder_path):
        pickle.dump(self.index, open(output_folder_path + "/tokens_entities_index.p", "wb"))
        pickle.dump(self.ids_to_entities, open(output_folder_path + "/ids_to_entities_map.p", "wb"))
        pickle.dump(self.entities_to_ids, open(output_folder_path + "/entities_to_ids_map.p", "wb"))
        pickle.dump(self.ids_to_tokens, open(output_folder_path + "/ids_to_tokens_map.p", "wb"))
        pickle.dump(self.tokens_to_ids, open(output_folder_path + "/tokens_to_ids_map.p", "wb"))
        pickle.dump(self.term_idfs_dict, open(output_folder_path + "/term_idf_scores_dict.p", "wb"))
        pickle.dump(self.norms_idfs_dict, open(output_folder_path + "/entities_norms_dict.p", "wb"))
        pickle.dump(self.most_frq_terms, open(output_folder_path + "/most_frq_terms_in_entities_dict.p", "wb"))
        pickle.dump(self.norms_tfidfs_dict, open(output_folder_path + "/entities_norms_tfidf_dict.p", "wb"))

    def loadIndex(self, index_file):
        index = pickle.load(open(index_file, "rb"))
        return index

    def storeIndex(self, path, index):
        pickle.dump(index, open(path + "index.p", "wb"))

    def build_idf_mapping(self, index = None, N = None):
        if index:
            self.index = index
        if N:
            self.N = N
        terms_idfs = collections.defaultdict(int)
        for term_id in self.index:
            terms_idfs[term_id] = self.compute_idf(term_id)
        self.term_idfs_dict = terms_idfs
        return terms_idfs

    def compute_entities_norms_idf(self):
        norms_dict = collections.defaultdict(float)

        for entity in self.entities:
            entity_id = self.entities_to_ids[entity]
            entities_text = [elem for elem in entity.split('_') if not elem == 'kb']
            idf_vector = []
            for token in entities_text:
                token_id = self.tokens_to_ids[token]
                idf_vector = np.append(idf_vector, self.term_idfs_dict[token_id])
            entity_l2_norm = norm(idf_vector)
            norms_dict[entity_id] = entity_l2_norm
        self.norms_idfs_dict = norms_dict
        return norms_dict

    def compute_entities_norms_tfidf(self, entities, entities_to_ids, tokens_to_ids, tfidf_dict):
        norms_dict = collections.defaultdict(float)
        print(len(tfidf_dict))
        stop_words = set(stopwords.words('english'))

        for entity in entities:
            entity_id = entities_to_ids[entity]
            entities_text = [elem for elem in entity.split('_') if not elem == 'kb']
            tfidf_vector = []
            for token in entities_text:
                if not token:
                    continue
                if token in stop_words:
                    continue
                token_id = tokens_to_ids[token]
                query_tuple = (entity_id, token_id)
                try:
                    tfidf_vector = np.append(tfidf_vector, tfidf_dict[query_tuple])
                except:
                    print('Tuple not found! ', entity, ' ', token)
            entity_l2_norm = norm(tfidf_vector)
            norms_dict[entity_id] = entity_l2_norm
        self.norms_tfidfs_dict = norms_dict
        return norms_dict

    def build_most_frq_term_entity(self, entities_to_ids):
        frq_terms_dict = collections.defaultdict(int)
        for entity, ent_id in entities_to_ids.items():
            entity = entity.strip()
            if not entity:
                continue
            else:
                entity_text = entity.replace('kb_','').replace('_',' ')
                most_frq_term = self.get_most_freq_term(entity_text)
                frq_terms_dict[ent_id] = most_frq_term
        self.most_frq_terms = frq_terms_dict
        return frq_terms_dict


    """
        New data structure: {(doc_id, term_id): score}    
    """

    def compute_term_weights_matrix_compact(self, ids_to_entities, vocab_index, tokens_to_ids):
        print('Constructing term-weights matrix...')
        self.index = vocab_index
        self.N = len(ids_to_entities)
        tfidf_matrix = collections.defaultdict(list)

        for token, entities in tqdm(vocab_index.items()):
            for entity_id in entities:
                entity = ids_to_entities[entity_id]
                print('token-entity ', token, entity)
                entities_text = ' '.join(str(elem) for elem in entity.split('_') if not elem == 'kb')
                tf_idf_score = self.compute_tf_idf(token, entities_text, entity_id)
                token_id = tokens_to_ids[token]
                tfidf_matrix[(entity_id, token_id)] = tf_idf_score

        return tfidf_matrix

    def compute_term_weights_matrix_compact_store(self, most_frq_terms, idf_map, ids_to_tokens, ids_to_entities, vocab_index, tokens_to_ids, store_file):
        print('Constructing and storing term-weights matrix...')
        self.index = vocab_index
        self.N = len(ids_to_entities)
        self.ids_to_entities = ids_to_entities
        self.ids_to_tokens = ids_to_tokens
        self.tokens_to_ids = tokens_to_ids
        stop_words = set(stopwords.words('english'))

        for token, entities in tqdm(vocab_index.items()):
            idf_score = idf_map[token]
            token = self.ids_to_tokens[token]

            if not token :
                continue
            if token in stop_words:
                continue
            for entity_id in entities:
                most_frq = most_frq_terms[entity_id]
                entity = ids_to_entities[entity_id]
                entities_text = entity.replace('kb_','').replace('_',' ')
                tf_idf_score = self.compute_tf_idf(idf_score, most_frq, token, entities_text)
                token_id = tokens_to_ids[token]
                with open(store_file + ".txt", "a") as f:
                    f.write(str(entity_id) + "#" + str(token_id) + "#" + str(tf_idf_score) + "\n")

        tfidf_dict = self.read_tw_matrix(store_file+ ".txt")
        pickle.dump(tfidf_dict, open(store_file +".p", "wb"))
        return tfidf_dict

    def print_time(self, elapsed_time):
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def read_tw_matrix(self, filename):
        with open(filename, "r") as f:
            dict = {}
            for line in f:
                values = line.split("#")
                dict[(int(values[0]), int(values[1]))] = float(values[2])
        return(dict)


    def compute_tf_idf(self, idf_score, most_frq, term, doc):
        # if (self.isInDoc(term, docID) == False):
        #     return 0
        # else:
        return self.compute_tf(term, doc, most_frq) * idf_score

    def isInDoc(self, term, docID):
        if docID in self.index[term]:
            return True
        return False

    def compute_tf(self, term, doc, most_frq):
        raw_frq = self.raw_frequency(term, doc)
        if (raw_frq > 0):
            return (1 + np.log10(raw_frq)) / (1 + np.log10(most_frq))
        else:
            return 0

    def compute_idf(self, term):
        return np.log10(self.N / self.get_docs_count(term))

    def get_docs_count(self, term):
        term_id = self.tokens_to_ids[term]
        return len(self.index[term_id])

    def raw_frequency(self, term, doc):
        count = 0
        if isinstance(doc, str):
            for word in doc.split():
                if term == word:
                    count = count + 1

        return count

    def get_most_freq_term(self, doc):
        if len(doc) == 1:
            return 1
        elif len(doc) == 0:
            return 0

        doc_freq = dict([word, self.raw_frequency(word, doc)] for word in doc.split())
        value, count = collections.Counter(doc_freq).most_common(1)[0]

        return count

    def store_term_weigths_matrix(self, path, matrix):
        pickle.dump(matrix, open(path + "saved_weights_matrix.p", "wb"))

    def retrieve_term_weights_matrix(self, path):
        matrix = pickle.load(open(path + "saved_weights_matrix.p", "rb"))
        return matrix

if __name__ == '__main__':
    with open("kb_full_wikidump.pickle", "rb") as fp:  # Entities ids gathered during preprocessing.
        kb_entities = pickle.load(fp)

    print(len(kb_entities))

    # Data Structures Construction
    ir_prep = InfoRetrievalSetup()

    # token_entities_index, ids_to_entities, entities_to_ids, ids_to_tokens, tokens_to_ids = ir_prep.build_index(kb_entities)
    #
    # idf_maps = ir_prep.build_idf_mapping()
    # entity_norms = ir_prep.compute_entities_norms()
    #
    # ir_prep.store_ir_datastructures()
    #
    # pickle.dump(idf_maps, open("ir_data/term_idf_scores_dict.p", "wb"))
    # pickle.dump(entity_norms, open("ir_data/all_entities_norms_dict.p", "wb"))

    #TFIDF

    # ids_to_tokens = pickle.load(open('ir_data/all_ids_to_tokens_map.p', "rb"))
    # ids_to_entities = pickle.load(open('ir_data/all_ids_to_entities_map.p', "rb"))
    # print(ids_to_tokens[2404095])
    # print(ids_to_entities[7340919])

    # tfidf_scores = pickle.load(open('ir_data/wiki_all_compact_tf_idf_dict.p', "rb"))
    # print(tfidf_scores)
    # tokens_to_ids = pickle.load(open('ir_data/all_tokens_to_ids_map.p', "rb"))
    # entities_to_ids = pickle.load(open('ir_data/all_entities_to_ids_map.p', "rb"))
    # tfidf_norms = ir_prep.compute_entities_norms_tfidf(kb_entities, entities_to_ids, tokens_to_ids, tfidf_scores)
    # pickle.dump(tfidf_norms, open("ir_data/all_entities_norms_tfidf_dict.p", "wb"))
    # print(len(tfidf_norms))
    # print(tfidf_scores[(917504, 18)])
    # print(tfidf_scores[(4268036, 8789)])
    # print(tfidf_scores[(917504, 18)])
    # print(tfidf_scores[(6053891, 2986)])



    # token_entities_index = pickle.load(open('ir_data/all_tokens_entities_index.p', "rb"))
    # tokens_to_ids = pickle.load(open('ir_data/all_tokens_to_ids_map.p', "rb"))
    # ids_to_tokens = pickle.load(open('ir_data/all_ids_to_tokens_map.p', "rb"))
    # ids_to_entities = pickle.load(open('ir_data/all_ids_to_entities_map.p', "rb"))
    # idf_map = pickle.load(open('ir_data/term_idf_scores_dict.p', "rb"))
    # entities_to_ids = pickle.load(open('ir_data/all_entities_to_ids_map.p', "rb"))
    # most_frq_terms = pickle.load(open('ir_data/most_frq_terms_in_entities_dict.p', "rb"))

    # most_frq_terms = ir_prep.build_most_frq_term_entity(entities_to_ids)
    # pickle.dump(most_frq_terms, open("ir_data/most_frq_terms_in_entities_dict.p", "wb"))
    # ir_prep.compute_term_weights_matrix_compact_store(most_frq_terms, idf_map, ids_to_tokens, ids_to_entities, token_entities_index, tokens_to_ids, 'ir_data/full_doc_term_tfidf_scores.txt')


    # CONLL
    #    # print(len(set(kb_entities)))
    # ir_prep = InfoRetrievalSetup()
    #
    with open("evalData/kb_entities_list_conll.p", "rb") as fp:  # Entities ids gathered during preprocessing.
        conll_entities = pickle.load(fp)
    print(len(conll_entities))
    #
    # token_entities_index, ids_to_entities, entities_to_ids, ids_to_tokens, tokens_to_ids = ir_prep.build_index(conll_entities)
    #
    # idf_map = ir_prep.build_idf_mapping()
    # entity_norms = ir_prep.compute_entities_norms_idf()
    #
    # ir_prep.store_ir_datastructures()
    #
    # pickle.dump(idf_map, open("evalData/conll_term_idf_scores_dict.p", "wb"))
    # pickle.dump(entity_norms, open("evalData/conll_entities_norms_dict.p", "wb"))
    #
    # most_frq_terms = ir_prep.build_most_frq_term_entity(entities_to_ids)
    # pickle.dump(most_frq_terms, open("evalData/conll_most_frq_terms_in_entities_dict.p", "wb"))
    # ir_prep.compute_term_weights_matrix_compact_store(most_frq_terms, idf_map, ids_to_tokens, ids_to_entities, token_entities_index, tokens_to_ids, 'evalData/conll_doc_term_tfidf_scores')
    # tfidf_scores = pickle.load(open('evalData/conll_doc_term_tfidf_scores.p', "rb"))
    #
    # tfidf_norms = ir_prep.compute_entities_norms_tfidf(conll_entities, entities_to_ids, tokens_to_ids, tfidf_scores)
    # pickle.dump(tfidf_norms, open("evalData/conll_entities_norms_tfidf_dict.p", "wb"))