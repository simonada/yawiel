from models.setup.embeddings import EmbeddingsLearner
from models.setup.ir import InfoRetrievalSetup
import pickle
import collections
import io
from tqdm import tqdm
import glob
import nltk
import os

def merge_mentions(mentions_to_merge, entity):
    root = True
    if len(mentions_to_merge) == 1:
        current = mentions_to_merge.pop()
        curr_key = list(current.keys())[0]
        current[curr_key] = {1: entity}
        return [current]
    while len(mentions_to_merge) > 1:
        current = mentions_to_merge.pop()
        # the final one is also the root
        curr_key = list(current.keys())[0]
        if root:
            current[curr_key] = {1: entity}
            root = False
        # hierarchical merge of the tokens that belong to the same entity/ nesting
        previous = mentions_to_merge.pop()
        prev_key = list(previous.keys())[0]
        previous[prev_key].update(current)
        mentions_to_merge.append(previous)
    return mentions_to_merge


def build_trie_lookup_structure(entitites_mentions_file, out_dir):
    print('Start building tree lookup data structure...')
    entities_lookup_trie = collections.defaultdict(dict)
    entities_list = []

    with io.open(entitites_mentions_file) as f:
        for entities_mentions in f.readlines():
            if '|' in entities_mentions:
                entity, mentions = entities_mentions.split('|')
                mentions = mentions.split('\t')
            else:
                # if no mentions are provided, but a list of entities only, treat the entity as the single mention to be replaced in the text
                entities_mentions = entities_mentions.replace('\n', '')
                entity = entities_mentions
                mentions = [entities_mentions]
            entity = entity.replace(' ', '_')
            entity = 'kb_' + entity
            # store a list of the entities only
            entities_list.append(entity)

            for mention in mentions:
                if mention != '\n':
                    mention_parts = mention.split(' ')
                    first_part = mention_parts[0]
                    # if the first token not in the dict, then build a whole new branch
                    if first_part not in entities_lookup_trie:
                        mentions_to_merge = []
                        for part in mention_parts:
                            mentions_to_merge.append({part: {0: None}}) # indicate that this subpart is not a valid entity
                        mentions_branch = merge_mentions(mentions_to_merge, entity)
                        entities_lookup_trie.update(mentions_branch[0])
                    else:
                        current_branch = entities_lookup_trie[first_part]
                        # remove part for which we have already found match
                        mention_parts.pop(0)
                        if mention_parts:
                            next_mention = mention_parts.pop(0)
                        else:
                            # case when there is a single mention already existing at the root, make sure it's set as allowed entity
                            next_mention = first_part
                            if list(current_branch.keys())[0] != 1:
                                if list(current_branch.keys())[0] == 0:
                                    del current_branch[0]
                                temp = {1: entity}
                                temp.update(current_branch)
                                entities_lookup_trie[next_mention] = temp
                            continue
                        prev_mention = ''
                        while next_mention in current_branch:
                            current_branch = current_branch[next_mention]
                            prev_mention = next_mention
                            if not mention_parts:
                                  continue
                            next_mention = mention_parts.pop(0)
                        if next_mention == prev_mention:  # there is one to one corresp in the dict already
                            if list(current_branch.keys())[0] == 0:
                                current_branch.update({1:entity})
                        else:
                            if mention_parts:
                                mentions_to_merge = []
                                mention_parts.insert(0, next_mention)
                                for part in mention_parts:
                                    mentions_to_merge.append({part: {0: None}})
                                new_node = merge_mentions(mentions_to_merge, entity)
                            else:
                                new_node = {next_mention:{1:entity}}
                            if isinstance(new_node, (list,)):
                                new_node = new_node[0]
                            current_branch.update(new_node)

    print('Finishing...')
    print('Entities Lookup Trie Size', len(entities_lookup_trie))
    with open(out_dir + '/mentions_tree.p', "wb") as fp:  # Pickling
        pickle.dump(entities_lookup_trie, fp)
    entities_list = list(set(entities_list))
    with open(out_dir + '/entities_list.p', "wb") as fp:  # Pickling
         pickle.dump(entities_list, fp)

def replace_mentions_with_entities(input_dir, out_dir):
    mentions_dir =  out_dir + '/mentions_tree.p'
    mentions_tree = pickle.load(open(mentions_dir, "rb"))
    files_to_process = input_dir + '/*'

    for file_nr, extracted_wiki_file in enumerate(tqdm(glob.glob(files_to_process))):
        print(extracted_wiki_file)
        file_name = extracted_wiki_file.split('/')[1]
        with io.open(extracted_wiki_file) as f:
            store_file_dir = out_dir + '/preprocessed/'
            store_file_name = store_file_dir + file_name + '_preprocessed'
            if not os.path.exists(store_file_dir):
                os.mkdir(store_file_dir)
            for doc_text in f.readlines():
                if doc_text.strip():
                    tokenized_sentences = nltk.word_tokenize(doc_text.lower())
                    updatet_text = []

                    for k in range(len(tokenized_sentences)):
                        if not tokenized_sentences:
                            continue
                        else:
                            original_sentence = tokenized_sentences.copy()
                            token = tokenized_sentences.pop(0)
                            entity = []
                            if token in mentions_tree:
                                current_branch = mentions_tree[token]
                                if list(current_branch.keys())[0] == 1:
                                    entity = list(current_branch.values())[0]
                                    original_sentence.pop(0)
                                if tokenized_sentences:
                                    next_mention = tokenized_sentences.pop(0)
                                else:
                                    next_mention = next_mention
                                while next_mention in current_branch:
                                    current_branch = current_branch[next_mention]
                                    # update if by adding the next match, the result is a valid entity
                                    if list(current_branch.keys())[0] == 1:
                                        entity = list(current_branch.values())[0]
                                        if original_sentence:
                                            original_sentence.pop(0)
                                    if tokenized_sentences:
                                        next_mention = tokenized_sentences.pop(0)
                                if entity:
                                    updatet_text.append(entity)
                                else:
                                    updatet_text.append(token)
                                    if original_sentence:
                                        original_sentence.pop(0)
                                tokenized_sentences = original_sentence # what is left after merging the words corresponding to an entitiy mention
                            else:
                                updatet_text.append(token)
                    with open(store_file_name, 'a') as file_to_store:
                        file_to_store.write(' '.join(updatet_text))

    print('Preprocessing finished.')

def train_store_embeddings(kb_entities_file, preprocessed_docs_folder, output_dir ):
    with open(kb_entities_file, "rb") as fp:  # Entities ids gathered during preprocessing.
        kb_entities = pickle.load(fp)
    output_dir = output_dir + '/embeddings'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    embed_model_data_path = output_dir + '/word2vec.model'
    kb_vecs_data_path = output_dir + '/kb_vectors.kv'
    vocab_vecs_data_path = output_dir + '/full_vocab_vectors.kv'

    embed = EmbeddingsLearner(preprocessed_docs_folder)
    embed.train(10)
    embed.split_entities_from_vocab_vectors(kb_entities)
    embed.save(embed_model_data_path, kb_vecs_data_path, vocab_vecs_data_path)
    print('Embeddings Model trained and saved.')

def build_store_ir_datatstructures(kb_entities_file, output_dir):
    with open(kb_entities_file, "rb") as fp:  # Entities ids gathered during preprocessing.
        kb_entities = pickle.load(fp)
    output_dir = output_dir + '/ir'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ir = InfoRetrievalSetup()
    token_entities_index, ids_to_entities, entities_to_ids, ids_to_tokens, tokens_to_ids = ir.build_index(kb_entities)

    idf_map = ir.build_idf_mapping()
    ir.compute_entities_norms_idf()
    most_frq_terms = ir.build_most_frq_term_entity(entities_to_ids)
    tfidf_scores = ir.compute_term_weights_matrix_compact_store(most_frq_terms, idf_map, ids_to_tokens, ids_to_entities,
                                                      token_entities_index, tokens_to_ids, output_dir +'/doc_term_tfidf_scores')
    ir.compute_entities_norms_tfidf(kb_entities, entities_to_ids, tokens_to_ids, tfidf_scores)
    ir.store_ir_datastructures(output_dir)
    print('IR Model data structures build and saved.')