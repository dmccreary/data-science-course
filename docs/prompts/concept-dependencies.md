# Concept Dependencies

## Prompt

```linenums="0"
That was perfect.  Thank you very much.
The next step is to create a dependency graph
of all the 150 concepts on the list. 
For each concept, think about other concepts
that this concept depends upon for understanding.

For each of the 150 concepts return a single row in CSV format.

Column 1 is the ConceptID (the integer 1 to 150)
Column 2 is the ConceptLabel
Column 3 is the Dependencies in the form of pipe-delimited list of ConceptIDs

Make sure that Foundation Concepts (prerequisites) like knowledge of Python don't have any dependencies.
Make sure that every concept except Foundation Concepts have at least one dependency.
```