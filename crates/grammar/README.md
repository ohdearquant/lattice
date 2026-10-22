# lattice-grammar

Grammar-constrained decoding for Lattice inference. JSON Schema and GBNF grammars
compile to a byte-level pushdown automaton, and `GrammarEngine` masks the logits of
tokens the grammar cannot accept at each decoding step.
