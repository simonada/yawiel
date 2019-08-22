from models.ir import IRModel
from models.embeddings import EmbeddingsModel
import operator
from collections import defaultdict
from itertools import chain

class CombinedModels(object):
    def __init__(self, ir_model, embed_model):
        self.ir_model = ir_model
        self.embed_model = embed_model

    def get_entity(self, mention):
        top, pred_ir = self.ir_model.get_entity(mention)
        ir_length = len(pred_ir)
        top, pred_embed = self.embed_model.get_entity(mention, k = ir_length) # restrict the returned candidates by the number from the IR model

        if not pred_embed and not pred_ir:
            print('--NME--')
            return [], []

        if (len(pred_ir) == 0) | (len(pred_embed) == 0):
            if (len(pred_ir) == 0):
                print('Using embeddings model only.')
                return pred_embed[0][0], pred_embed
            else:
                print('Using IR model only.')
                return pred_ir[0][0], pred_ir

        if (len(pred_ir) != len(pred_embed)):
            pred_ir, pred_embed = self.align_predictions(pred_ir, pred_embed)

        with open('logging_combo_model.txt', 'a') as the_file:

            the_file.write('IR '+ str(pred_ir) + '\n')
            the_file.write('Embed' + str(pred_embed) + '\n')

        # Make the ranking in absolute terms, i.e. based on the entities position
        ir_dict = self.rank_positions(dict(pred_ir))
        embed_dict = self.rank_positions(dict(pred_embed))

        # print('IR ', ir_dict)
        # print('Embed ', embed_dict)

        # Search for exact matches from the two models_data in terms of keys, i.e. entities
        results = {k: (ir_dict[k] + embed_dict[k])/2 for k in ir_dict.keys() & embed_dict.keys()}
        # note that lower score means higher rank !!!
        sorted_entities = sorted(results.items(), key=operator.itemgetter(1), reverse=False)

        if sorted_entities:
            with open('logging_combo_model.txt', 'a') as the_file:
                the_file.write('Alternative sorting: '+str(sorted_entities)+'\n')
            return sorted_entities[0][0], sorted_entities
        else:
            # No intersections, do union
            alternative = defaultdict(list)

            if pred_embed[0][1] > 0.8:
                for k, v in chain(embed_dict.items(), ir_dict.items()): # the order determines which model will be preferred!!! TODO: better sorting in the non-match cases
                    alternative[k].append(v)
            else:
                for k, v in chain( ir_dict.items(), embed_dict.items()): # the order determines which model will be preferred!!! TODO: better sorting in the non-match cases
                    alternative[k].append(v)

            sorted_entities = sorted(alternative.items(), key=operator.itemgetter(1), reverse=False)

            if sorted_entities:
                with open('logging_combo_model.txt', 'a') as the_file:
                    the_file.write('Alternative sorting: ' + str(sorted_entities) + '\n')
                return sorted_entities[0][0], sorted_entities
            else:
                # print('Entity not found by any KB.')
                return [], []


    def rank_positions(self, dictionary):
        i = 1
        for k, v in dictionary.items():
            dictionary[k] = i
            i += 1
        return dictionary

    def align_predictions(self, pred_ir, pred_embed):

        if(len(pred_embed)< len(pred_ir)):
            pred_ir = pred_ir[0:len(pred_embed)]
        else:
            pred_embed = pred_embed[0:len(pred_ir)]

        return pred_ir, pred_embed

if __name__ == '__main__':
    mention = 'ac milan'

    # Compact TFIDF ALTERNATIVE
    ir_mod = IRModel()
    print('IR Model')
    top, pred_ir = ir_mod.get_entity(mention)
    print(pred_ir)
    print(top)
    print()

    # Embeddings
    print('Embeddings Model')
    embed_mod = EmbeddingsModel()
    top, pred_embed = embed_mod.get_entity(mention)
    print(pred_embed)
    print(top)

    # Combine
    print()
    print('Joint')

    comb = CombinedModels(ir_mod, embed_mod)
    top, all = comb.get_entity(mention)
    print(all)
    print(top)
