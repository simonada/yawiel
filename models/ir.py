import numpy as np
import operator
import collections
import pickle
from models import tfidf
import json
import os

class IRModel:

    def __init__(self, model, output_dir):
        output_dir = output_dir + '/ir'

        self.model_to_use = model
        self.index = self.loadPickle(output_dir + '/tokens_entities_index.p')
        self.tokens_to_ids = self.loadPickle(output_dir + '/tokens_to_ids_map.p')
        self.index_values = list(self.index.keys())
        self.entities_lookup = self.loadPickle(output_dir + '/ids_to_entities_map.p')
        self.entities_ids_lookup = self.loadPickle(output_dir + '/entities_to_ids_map.p')
        self.term_weights_matrix = self.loadPickle(output_dir + '/doc_term_tfidf_scores.p')
        self.idf_matrix = self.loadPickle(output_dir + '/term_idf_scores_dict.p')
        self.entities_norms = self.loadPickle(output_dir + '/entities_norms_dict.p')
        self.entities_norms_tfidf = self.loadPickle(output_dir + '/entities_norms_tfidf_dict.p')

        print('IR Model initialized. Index size: ', len(self.index))



    def loadPickle(self, file):
        return pickle.load(open(file, "rb"))

    def getIndex(self):
        return self.index

    def get_entity(self, mention):
        if self.model_to_use == 'tfidf':
            return self.retrieve_compact_tfidf_alternative(mention)
        elif self.model_to_use == 'idf':
            return self.retrieve_idf_only(mention)
        else:
            return self.retrieve_index_only(mention)

    def retrieve_index_only(self, query, k=None):
        # print('IR Index Based Retrieval')
        relevant_docs = []
        relevant_docs_dict = collections.defaultdict(int)

        query_term_ids = list()

        for term in query.split():
            term_id = self.tokens_to_ids[term]
            if term_id in self.index:
                query_term_ids.append(term_id)
                relevant_docs_all = tuple(self.index[term_id])
                for doc in relevant_docs_all:
                    relevant_docs_dict[doc] = relevant_docs_dict[doc] + 1
                    # if doc in self.term_weights_matrix:

        # how many words overlap each match has
        relevant_docs_dict = sorted(relevant_docs_dict.items(), key=operator.itemgetter(1), reverse=True)
        prediction_string_with_scores = [(self.entities_lookup[pred[0]], pred[1]) for pred in relevant_docs_dict]
        prediction_string_with_scores.sort(key=lambda x: len(x[0]))
        prediction_string_with_scores.sort(key=lambda x: x[1], reverse=True)
        #prediction_string_with_scores = prediction_string_with_scores.sort(key=lambda t: t[1], reverse=True)
        # print(prediction_string_with_scores)

        for key, value in relevant_docs_dict:
            relevant_docs.append(key)

        if not relevant_docs:
            return [], [] # no matchin entities found, return empty

        top_prediction = prediction_string_with_scores[0][0]

        if top_prediction:
            if k:
                return top_prediction, prediction_string_with_scores[0: k]
            else:
                return top_prediction, prediction_string_with_scores
        else:
            return top_prediction, prediction_string_with_scores

    def retrieve_idf_only(self, query, k=None):
        candidates_dict = collections.defaultdict(float)
        relevant_entities = []
        query_terms = query.split()

        # COMPUTE IDF WEIGHTING FOR THE ENTITIES
        for term in query_terms:
            term_id = self.tokens_to_ids[term]
            if term_id in self.index:
                idf = self.idf_matrix[term_id]
                relevant_entities_all = tuple(self.index[term_id])
                for entity in relevant_entities_all:
                    candidates_dict[entity] = candidates_dict[entity] + idf

        # NORMALIZE THE ENTITIES WEIGHTS
        for candidate_id, current_idf in candidates_dict.items():
            entity_norm = self.entities_norms[candidate_id]
            candidates_dict[candidate_id] = current_idf / entity_norm
            relevant_entities.append(candidate_id)

        candidates_dict = sorted(candidates_dict.items(), key=operator.itemgetter(1), reverse=True)

        if not relevant_entities:
            return [], []

        prediction_string_with_scores = [(self.entities_lookup[pred[0]], pred[1]) for pred in candidates_dict]
        top_prediction = prediction_string_with_scores[0][0]

        if top_prediction:
            if k:
                return top_prediction, prediction_string_with_scores[0: k]
            else:
                return top_prediction, prediction_string_with_scores
        else:
            return top_prediction, prediction_string_with_scores

    def retrieve_compact_tfidf_alternative(self, query, k=None):

        ##For every term in the query, retrieve the docID of only those docs that contain the query term
        candidate_entitites_ids = set()
        query_term_ids = list()

        for term in query.split():
            term_id = self.tokens_to_ids[term]
            if term_id in self.index:
                query_term_ids.append(term_id)
                relevant_entities_all = tuple(self.index[term_id])
                for doc in relevant_entities_all:
                    # if doc in self.term_weights_matrix:
                    candidate_entitites_ids.add(doc)


        candidates_dict = collections.defaultdict(float)

        for candidate_id in candidate_entitites_ids:
            # First generate the vector for the candidate
            for query_term_id in query_term_ids:
                query_tuple = (candidate_id, query_term_id)
                # print(query_tuple)
                try:
                    tfidf_score = self.term_weights_matrix[query_tuple]
                    candidates_dict[candidate_id] = candidates_dict[candidate_id] + tfidf_score
                except Exception as e:
                    #traceback.print_exc()
                    continue

        # NORMALIZE THE ENTITIES WEIGHTS
        for candidate_id, current_idf in candidates_dict.items():
            entity_norm = self.entities_norms_tfidf[candidate_id]
            candidates_dict[candidate_id] = current_idf / entity_norm

        candidates_dict = sorted(candidates_dict.items(), key=operator.itemgetter(1), reverse=True)

        # further sort if the mention is of single word
        prediction_string_with_scores = [(self.entities_lookup[pred[0]], pred[1]) for pred in candidates_dict]


        if prediction_string_with_scores:
            top_prediction = prediction_string_with_scores[0][0]
            if k:
                return top_prediction, prediction_string_with_scores[0: k]
            else:
                return top_prediction, prediction_string_with_scores
        else:
            return [], []

    def retrieve_compact_tf_idf(self, query, k=None):

        ##For every term in the query, retrieve the docID of only those docs that contain the query term
        candidate_entitites_ids = set()
        query_term_ids = list()

        for term in query.split():
            term_id = self.tokens_to_ids[term]
            if term_id in self.index:
                query_term_ids.append(term_id)
                relevant_entities_all = tuple(self.index[term_id])
                for doc in relevant_entities_all:
                    # if doc in self.term_weights_matrix:
                    candidate_entitites_ids.add(doc)

        ##Dictionary to store the distance between the query and each document that contains the query terms
        ##key: docID; value: distance

        # If there is a single candidate, no need to compute similarities, return directly
        if len(candidate_entitites_ids) == 1:
            top_prediction = self.get_entities_strings(candidate_entitites_ids)[0]
            return top_prediction, [(top_prediction,1)]

        similarities_dict = collections.defaultdict(float)
        query_vec = np.ones(len(query_term_ids)).tolist()

        for candidate_id in candidate_entitites_ids:
            # First generate the vector for the candidate
            candidate_vec = list()
            for query_term_id in query_term_ids:
                query_tuple = (candidate_id, query_term_id)
                # print(query_tuple)
                try:
                    score = self.term_weights_matrix[query_tuple]
                    # print('tfidf score ', score)
                    if score:
                        candidate_vec.append(score)
                    else:
                        candidate_vec.append(0)
                except Exception as e:
                    # print(e)
                    candidate_vec.append(0)
            # Then compute the similarity to the query
            similarity = tfidf.compute_cosine_similarity_with_norm(query_vec, candidate_vec)
            similarities_dict[candidate_id] = similarity

        ##Sort the documents by the distance, smallest distance to top
        sorted_similarities_dict = sorted(similarities_dict.items(), key=operator.itemgetter(1), reverse=True)
        prediction_string_with_scores = [(self.entities_lookup[pred[0]], pred[1]) for pred in sorted_similarities_dict]

        relevant_docs = []

        for key, value in sorted_similarities_dict:
            relevant_docs.append(key)

        if not relevant_docs:
            return [],[]

        top_prediction = prediction_string_with_scores[0][0]

        if top_prediction:
            if k:
                return top_prediction, prediction_string_with_scores[0: k]
            else:
                return top_prediction, prediction_string_with_scores
        else:
            return top_prediction, prediction_string_with_scores

    def get_entities_strings(self, entities_ids, sort=False):
        entities_strings = []
        for ent_id, score in entities_ids:
            entities_strings.append(self.entities_lookup[ent_id])
        if sort:
            # If the mention is a single word, we want the smallest candidate length
            entities_strings.sort(key=len)
        return entities_strings

    def print_predictions(self, predictions_ids):
        if (len(predictions_ids) > 0):
            for top_pred in predictions_ids:
                print(self.entities_lookup[top_pred])
        else:
            print("Entities not found.")

if __name__ == '__main__':
    # Compact TFIDF ALTERNATIVE
    search = IRModel()

    predictions, relevant_docs_dict = search.get_entity('china')
    print(relevant_docs_dict)
    print('Top prediction ', predictions)

    print()
    predictions, relevant_docs_dict = search.retrieve_idf_only('masayuki okano')
    print(relevant_docs_dict)
    print('Top prediction ', predictions)

    # INDEX ONLY ALTERNATIVE
    # search = IRModel('ir_data/test_index.p', 'ir_data/id_to_entities_index.p',
    #                  'ir_data/entities_to_id_index.p', 'ir_data/test_tokens_to_ids.p', None)
    # predictions, relevant_docs_dict = search.get_entity('law')
    # print(relevant_docs_dict)
    # print(predictions)
