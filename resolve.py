import pickle

from models.combined import CombinedModels
from models.embeddings import EmbeddingsModel
from models.ir import IRModel
import spacy

class Doc(object):
    def __init__(self, text, tagged_mentions=None, doc_id=None, mentions_targets=None, context_vector=None):
        self.text = text
        self.doc_id = doc_id
        self.mentions = tagged_mentions
        self.mentions_targets = mentions_targets  # list of tuples (mention, target)
        self.context_vec = context_vector

class Mention(object):
    def __init__(self, mention_text, begin, end, target_entity=None):
        self.mention_text = mention_text
        self.begin_idx = begin
        self.end_idx = end
        self.size = end - begin
        self.target_entity = target_entity

def resolve_document_mentions(doc, model):
    doc_id = doc.doc_id.split('/')
    print('Performing entity linking for: ', doc_id[1])
    output_file = 'out/'+doc_id[1] +'_linking_output'
    for mention in doc.mentions:
        mention_text = mention.mention_text
        top_prediction, predictions_list_weighted = model.get_entity(mention_text)
        if top_prediction:
            mention.target_entity = top_prediction
            confidence = predictions_list_weighted[0][1]
            with open(output_file, 'a') as file_to_store:
                file_to_store.write(mention_text + ' , '+ str(mention.begin_idx) + ' - ' + str(mention.end_idx-1) + ' , ' + str(top_prediction) + ' , ' + str(confidence)+'\n')


def resolve(documents_file, model_id, output_dir):
    ir_id = 'idf'

    with open(documents_file, "rb") as fp:  # Entities ids gathered during preprocessing.
        docs = pickle.load(fp)
    if 'embed' in model_id:
        model = EmbeddingsModel(output_dir)
    elif "ir" in model_id:
        model = IRModel(ir_id, output_dir)
    else:
        ir_mod = IRModel(ir_id, output_dir)
        embed_mod = EmbeddingsModel(output_dir)
        model = CombinedModels(ir_mod, embed_mod)

    for doc in docs:
        resolve_document_mentions(doc, model)

    print('Entity linking finished.')

