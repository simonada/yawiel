import pickle
import spacy
import io
from tqdm import tqdm
import glob

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

class SpacyTagger(object):
    def __init__(self, spacy_model = None):
        self.nlp = spacy.load('en_core_web_md')

    def tag(self, doc):
        doc = self.nlp(doc)
        mentions = []
        for chunk in doc.noun_chunks:
            # print(chunk.text)
            mentions.append(Mention(chunk.text, chunk.start, chunk.end))
        return mentions

    def annotate_docs(self, input_dir):
        docs = []
        input_dir = input_dir + '/*'
        for file_nr, extracted_wiki_file in enumerate(tqdm(glob.glob(input_dir))):
            print('Extracting potential entitiy mentions for: '+extracted_wiki_file)
            doc_full_text = []
            with io.open(extracted_wiki_file) as f:
                for doc_text in f.readlines():
                    if doc_text.strip():
                        doc_full_text.append(doc_text)
            doc_full_text = ' '.join(doc_full_text)
            all_mentions = self.tag(doc_full_text.lower())
            docs.append(Doc(doc_full_text, all_mentions, extracted_wiki_file))
        return docs

def chunck_and_store_docs(input_dir, output_dir):
    tagger = SpacyTagger()
    docs = tagger.annotate_docs(input_dir)
    output_dir_docs = output_dir + '/tagged_docs.p'
    with open(output_dir_docs, "wb") as fp:  # Pickling
        pickle.dump(docs, fp)

if __name__ == '__main__':
    list_dir_string = './preprocessing/source/wiki/wiki_*'  # LOCAL
    output_dir = './preprocessing/preprocessed'
    tagger = SpacyTagger()
    docs = tagger.annotate_docs()
    output_dir_docs = output_dir + '/tagged_docs.p'
    with open(output_dir_docs, "wb") as fp:  # Pickling
        pickle.dump(docs, fp)