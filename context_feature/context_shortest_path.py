import collections
import stanza
import networkx as nx
import numpy as np
import itertools
import operator
from itertools import combinations, product
import collections

nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')


def path(edges,opis,ents, op_pol, tags):
	"""
    Searches for shortest path between entity and opinion word
    :param edges: edge list (entity, opinion_word)
    :param opis: list of opinion words
    :param ents: list of entities
    :param op_pol: dict(key=opinion_word, value=polarity_score)
    :param tags: POS tags
    :return: dict(key=entity, value=polarity score of the closer opinion word)
    """ 
	s = []
	entity_opinion = {}
	graph = nx.Graph(edges)
	if len(ents) > 0:
		t = (ents,opis)
		for pair in product(*t):
			if nx.has_path(graph, pair[0], pair[1]):
				s.append((nx.shortest_path_length(graph, source=pair[0], target=pair[1]),pair[1], pair[0]))
		sx = [(a,b,c) for (a,b,c) in s if tags[c] != 'VERB' and tags[b] !='VERB']
		results = []
		for key, group in itertools.groupby(sx, operator.itemgetter(2)):
			mini = min(list(group))
			entity_opinion[key] =  int(op_pol[mini[1]]) #np.round(int(op_pol[mini[1]]) / (mini[0] + 1), 3)
	return entity_opinion


def sentRelations(main_df, lex_path):
	"""
    Searches for shortest path between entity and opinion word in a document
    :param main_df: article's dataframe
    :param lex_path: location of the lexicon file
    :return: dict(key=entity, value=polarity score of the closer opinion word) on document level
    """ 
	pos = main_df.POS_tag.tolist()
	ids = main_df.ent_id.tolist()
	words = main_df.word.tolist()
	tags = dict(zip(words,pos))
	text = " ".join(words)
	word_ids = dict(zip(words,ids))
	total_relations = []
	total_sent_polarities = []
	ems = get_opinion(lex_path)
	doc = nlp(text)
	edges = []
	op_pol = {}
	for i in range(len(doc.sentences)):
		opis = []
		ents = []
		for word in doc.sentences[i].words:
			head = doc.sentences[i].words[word.head-1].text if word.head > 0 else "root"
			edges.append((head, word.text))
			if word.lemma in ems.keys() and ems[word.lemma] != '0':
				op_pol[word.text] = ems[word.lemma] 
				opis.append(word.text)
			if word.text in word_ids.keys() and word_ids[word.text] != '_':
				ents.append(word.text)
		entity_opinion = path(edges,opis,ents, op_pol, tags)
		total_relations.append(entity_opinion)
	dicc = collections.defaultdict(list)
	rel_result = {k:v for d in total_relations for k,v in d.items()}
	# rel_result = {k:np.mean(v) for k,v in dicc.items()}
	return rel_result
