import operator
from collections import defaultdict
import json
import pickle

class ProbabilisticModel(object):
    def __init__(self):
        with open('config.json') as json_data_file:
            config_data = json.load(json_data_file)

        model = config_data['prob_model']
        self.entity_prior = pickle.load(open(model['entity_priors'], "rb"))
        self.name_probability = pickle.load(open(model['mention_candidates_prob'], "rb"))
        self.compute_candidate_scores()

    def compute_candidate_scores(self):
        prior_name_probabilities_combined = self.name_probability.copy()
        for mention, candidates in prior_name_probabilities_combined.items():
            for candidate, probability in candidates.items():
                candidates[candidate] = round(candidates[candidate] * self.entity_prior[candidate], 3)
        self.prior_name_probabilities_combined = prior_name_probabilities_combined
        # return prior_name_probabilities_combined

    def get_entity(self, mention):
        if mention in self.prior_name_probabilities_combined:
            candidates = self.prior_name_probabilities_combined[mention]
            top_candidate = max(candidates.items(), key=operator.itemgetter(1))[0]
        else:
            return [], []
        return top_candidate, candidates

def build_probabilistic_data(docs_train): # from the file 'AIDA-YAGO2-dataset-with-NER.tsv' in Doc format
    kb_entities_frq_dict = defaultdict(int)
    mention_entity_name_probabilities_dict = defaultdict(dict)
    for doc in docs_train:
        for mention, kb_entry in doc.mentions_targets:
            # print(mention, kb_entry)
            kb_entities_frq_dict[kb_entry] += 1
            mention_entity_name_probabilities_dict = update_mention_entity_names_dictionary(mention, kb_entry,
                                                                                            mention_entity_name_probabilities_dict)
    entity_prior = {k: round(v / len(docs_train), 3) for k, v in kb_entities_frq_dict.items()}
    # entity_prior = sorted(entity_prior.items(), key=operator.itemgetter(1), reverse=True)
    name_probabilities = {k: compute_probabilities(v) for k, v in mention_entity_name_probabilities_dict.items()}
    # name_probabilities = {k: sorted(v.items(), key=operator.itemgetter(1), reverse=True) for k, v in name_probabilities.items()}

    return entity_prior, name_probabilities

def update_mention_entity_names_dictionary(mention, kb_entry, mention_entity_name_probabilities_dict):
    if mention in mention_entity_name_probabilities_dict:
        if kb_entry in mention_entity_name_probabilities_dict[mention]:
             mention_entity_name_probabilities_dict[mention][kb_entry] += 1
        else:
            mention_entity_name_probabilities_dict[mention].update({kb_entry:1})
    else:
        mention_entity_name_probabilities_dict.update({mention: {kb_entry:1}})
    return mention_entity_name_probabilities_dict

def compute_probabilities(mention_dict):
    total_frq = sum(mention_dict.values())
    # Need for logarithm???
    return {k: round(v / total_frq, 3) for k, v in mention_dict.items()}