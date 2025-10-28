

# Implement a dependency parser (using, for instance, spaCY, NLTK or Stanza) and parse the three sentences above

corpus = [
    "Flying planes can be dangerous",
    "The parents of the bride and the groom were flying",
    "The groom loves dangerous planes more than the bride",
]

import spacy

nlp = spacy.load("en_core_web_sm")

def print_tree(token, level=0):
    print("  " * level + f"{token.text} ({token.dep_})")
    for child in token.children:
        print_tree(child, level + 1)

for sentence in corpus:
    print(f"\nSentence: {sentence}")
    doc = nlp(sentence)
    # Find the root of the sentence
    for token in doc:
        if token.head == token:
            root = token
            break
    print_tree(root)