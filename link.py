import sys
import preprocess
import chunk
import resolve

args = sys.argv
print(str(args))

if len(args) != 5:
    print('Missing argument. Expected: input directory, entities file, out directory, model to use for linking.')
#
data_dir = args[1]
entities_file = args[2]
output_dir = args[3]
model = args[4]
output_dir_ir = output_dir + '/ir'
output_dir_preprocessed_files = output_dir + '/preprocessed'
output_dir_embed = output_dir + '/embed'
#
# data_dir = 'data'
# entities_file = 'entities_mentions_sample.txt'
# output_dir = 'out'
# output_dir_ir = 'out/ir'
# output_dir_preprocessed_files = 'out/preprocessed'
# output_dir_embed = 'out/embed'
# model = 'embed'

if __name__ == "__main__":

    ### PREPARE DATA ###
    print()
    preprocess.build_trie_lookup_structure(entities_file, output_dir)

    #1. Embeddings
    if 'embed' in model:
        preprocess.replace_mentions_with_entities(data_dir, output_dir)

        preprocess.train_store_embeddings(output_dir + '/entities_list.p', output_dir_preprocessed_files, output_dir)

    #2. IR
    if 'ir' in model:
        preprocess.build_store_ir_datatstructures(output_dir + '/entities_list.p', output_dir)

    #3. Tag chunks for mention detection
    chunk.chunck_and_store_docs(data_dir, output_dir)

    ##4. Resolve the mentions and output the linked texts.
    resolve.resolve('out/tagged_docs.p', model, output_dir)
