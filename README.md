# YAWIEL

Simple entity linking system that allows to annotate a set of documents for a list of entities. It supports disambiguation based on classical IR approach (idf), a disambiguation based on word embeddings, as well as combination of those two models.

## Getting Started

### Prerequisites

1. Python v.3.7.
2. Packages
- gensim, see https://radimrehurek.com/gensim/install.html
- spaCy v.2.1.6., en_core_web_md, see https://spacy.io/usage

### How to run
1. Place the text documents for annotation in a taget folder, e.g. 'data'.

2. Place the file with the entities in the highest directory level. Expected format for that file is:
```
consumer organization|consumer unions	consumer groups	
```
where the first entry is the canonical entity, separated with a | sign from the possible entity mentions. The entity mentions are tab-separated.

Alternatively only a list of the target entities can be provided.
```
consumer organization
```
In this case the entity will be treated as its only mention.

3. Run link.py providing the path to the folder with the documents, the list of entities,  and the target folder path, as well as the desired model. Possible models are 'embed' for Embeddings based retrieval, 'ir' for Information Retrieval based approach. In any other cases the two models will be combined.

```
python link.py data entities_mentions_sample.txt out ir
```
Depending on the target model to use, the appropriate preprocessing will be performed and the corresponding data structures/ models stored in the 'out' directory.

4. Read the outputs
The final output will be files of lists of disambiguated entites for each original input file. Each line has the following format:

```
computer programs , 8546 - 8547 , kb_computer_programs , 0.9999998807907104
```
Where the first entry is the mention, the second is the offset. This is followed by the entity ID and the confidence of the model for this disambiguation. Note that in the case of IR Models used this will be the corresponding IDF score.

## Authors

* **Simona E. Doneva**

## Acknowledgments
Thanks to Prof.Dr. Goran Glavas and Samuel Broscheit for the advice and discussions.

## License

This project is licensed under the MIT License.




