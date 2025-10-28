### Describe an application which needs syntactic and/or dependency parsing. Explain briefly why. (1/2 pag.)

Application: QA system for customer support

Syntactic Parsing:

Syntactic parsing is crucial in question answering systems because it provides the grammatical structure of user input. For instance, syntactic parsing helps the system identify the main clause and distinguish between subjects, objects, and modifiers. This allows the question answering system to interpret whether the user is asking about a person, place, or thing, and to formulate a suitable search or extraction pattern within texts. For example, understanding the structure of a question like "What does product X do?" enables the system to look specifically for noun phrases following verbs of discovery in the corpus.

Dependency Parsing:

Dependency parsing is particularly important for mapping semantic roles and relationships between words in a question or candidate answer sentence. By establishing direct links, such as which noun serves as the subject or object of a verb, the system can more accurately locate relevant information. For example, with the question "What department does product X belong to?", dependency parsing reveals that the answer sought is the department associated with the product, enabling precise extraction of “Sales” from sentences like “Product X belongs to the Sales department.” This deep understanding of relational structure is critical for high-precision matching in QA systems.