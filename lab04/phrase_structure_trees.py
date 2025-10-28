import nltk


corpus = [
    "Flying planes can be dangerous",
    "The parents of the bride and the groom were flying",
    "The groom loves dangerous planes more than the bride",
]

grammar = nltk.CFG.fromstring("""
    S -> GerundP VP | NP VP | NP VP NP | NP VP Comp 'than' NP

    GerundP -> V N
    NP -> N | Enum | Ofthe | NP Comp 'than' NP
    VP -> Modal V Adj | Vc | V NP

    Ofthe -> NP 'of' NP
    Enum -> NP 'and' NP
    N -> Adj N | Article N
    Vc -> V V

    V -> 'flying' | 'be' | 'were' | 'loves'
    N -> 'planes' | 'parents' | 'bride' | 'groom'
    Article -> 'the'
    Modal -> 'can'
    Comp -> 'more'
    Adj -> 'dangerous' | 'flying'
""")

parser = nltk.ChartParser(grammar)

for sentence in corpus:
    print(f"sentence: {sentence}")
    tokens = sentence.split(" ")
    tokens = list(map(lambda x: x.lower(), tokens))
    print(f"tokens: {tokens}")
    trees = parser.parse(tokens)

    print("trees: ")
    for tree in trees:
        print(tree) 