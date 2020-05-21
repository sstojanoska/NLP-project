import pandas as pd
import stanza
import re
import numpy as np
import glob
import csv
import sklearn
import nltk
from nltk.corpus import stopwords
import networkx as nx
import itertools
import operator
from itertools import combinations, product
import collections
from collections import defaultdict
from context_deprel import get_featureSentiment
from context_shortest_path import sentRelations
from context_window import getWindowPol


def documentPolarity(doc_path):
	"""
    Uses sentiNews dataset to get documentPolarity feature
    :param doc_path: the path of the dataset
    :return: dict of document and its polarity
    """
	numerical_pol = {'neutral':0, 'positive':1, 'negative':-1}
	with open(doc_path) as inf:
		reader = csv.reader(inf, delimiter="\t")
		next(reader)
		cols = list(zip(*reader))
	nid_col = cols[0]
	sentiment_col =cols[-1]
	doc_polarity = {nid_col[i]:numerical_pol[sentiment_col[i]] for i in range(len(nid_col))} 
	return doc_polarity

def get_opinion(lex_path):
	d = {}
	lex = lex_path+"lex.txt"
	with open(lex) as f:
		for line in f:
			(word, AFINN, freq, avg_AFINN, sd_AFINN) = line.split('\t')
			d[word] = AFINN
	return d


class Article:

	def __init__(self, data_path, lex_path, doc_path):
		self.data_path = data_path
		self.lex_path = lex_path
		self.doc_path = doc_path

	def lemmatize_dependency(self, word_list):
		"""
        Uses Stanza pipeline to extract lemmas, POS-tags and dependency relations
        :param word_list: list of words in the article
        :return: lists for each feature
        """
		dep = []
		lem = []
		pos = []
		text = " ".join(word_list)
		doc = nlp(text)
		for sentence in doc.sentences:
			for d in sentence.dependencies:
				dep.append(d[1])	
			for w in sentence.words:
				lem.append(w.lemma)
				pos.append(w.upos)
		return dep, lem, pos

	def unnesting(self, df):
		"""
        Fixes double annotated entities
        :param df: main dataframe of the article
        :return: expanded dataframe
        """
		df["ent_id"] = df.ent_id.str.split("|") 
		df["ner"] = df.ner.str.split("|")
		df["polarity"] = df.polarity.str.split("|")
		dfe = df.explode("ent_id")
		dfn = dfe.explode("ner")
		dfp = dfn.explode("polarity").reset_index()
		return dfp


	def removeSpecialChars(self, main_df):
		"""
        Removes special characters from article
        :param df: main dataframe of the article
        :return: dataframe without special chars
        """
		# stop = stopwords.words('slovene')
		# main_df = main_df[~main_df['word'].isin(stop)]
		main_df = main_df[~main_df.word.str.contains(r'\d', na=False)]
		main_df = main_df[~main_df.word.str.contains(r'[a-zA-Z]+\.[a-zA-Z]+\.*[a-zA-Z]*', na=False)] #remove abbreviations
		spec_chars = ['"',"#","%","&","'","(",")","*","+","-","/",":",";","<","=",">","@","[","\\","]","^","_","`","{","|","}","~","–"]
		for c in spec_chars:
			main_df['word'] = main_df['word'].str.replace(c,'')
		main_df['word'].replace('', np.nan, inplace=True)
		main_df.dropna(subset=['word'], inplace=True)
		return main_df


	def read_data(self):
		"""
        This is main method for preprocessing
        :return: dataframe where each row is feature vector of an entity
        """
		df = pd.read_csv(self.data_path, quotechar='"', quoting=csv.QUOTE_NONE,index_col=0, sep='\t',  names=['char_pos', 'word', 'ner', 'polarity', 'ni', 'ent_id','nan'], header=None, skiprows=7)
		docName = str(self.data_path.split("/")[1].split(".")[0])
		doc = documentPolarity(self.doc_path)
		docPol = doc[docName]
		d = self.removeSpecialChars(df)
		ti = ["".join(w.split("'")) for w in d['word'].tolist()] #apostrophe merged to the word: He's ->Hes
		dependency, lemma, pos = self.lemmatize_dependency(ti)
		d['dependency'] = dependency
		d['lemma'] = lemma
		d['POS_tag'] = pos
		rels = get_featureSentiment(d, lex_path) #sentRelations(d,lex_path)
		expanded_df = self.unnesting(d)
		stop = stopwords.words('slovene')
		expanded_df = expanded_df[~expanded_df['word'].isin(stop)]
		expanded_df.reset_index(inplace=True)
		article_df = self.createFeatures(expanded_df, rels, docPol)
		return article_df



	def createFeatures(self, main_df, rels, docPol):
		"""
        Creates features for each entity dataframe
        :param main_df: main dataframe of the article
        :param polarity_df: dataframe [entity, its_context_polarity]
        :return: dataframe where each row is feature vector of relsan entity
        """	
		df_empty = pd.DataFrame({'isPerson' : [],'isSubject':[],'isObject':[],'hasDescriptors':[],'contPol':[],'docPol':[],'polarity':[]})
		for i, g in main_df.groupby('ent_id'):
			if i != '_':
				df2 = self.processDF(main_df,g, rels, docPol)
				df_empty = df_empty.append(df2, ignore_index=True)
		return df_empty
	

	def mainEntity(self, df):
		"""
        This is a method for finding main entity
        :param df: entity's dataframe
        :return: most suitable main entity of all coreferences (TODO:find better way)
        """	
		if not df.loc[df['POS_tag'] == 'PROPN'].empty:
			return df.loc[df['POS_tag'] == 'PROPN'].head(1).lemma.item()
		elif not df.loc[df['POS_tag'] == 'NOUN'].empty:
			return df.loc[df['POS_tag'] == 'NOUN'].head(1).lemma.item()
		elif not df.loc[df['POS_tag'] == 'ADJ'].empty:
			return df.loc[df['POS_tag'] == 'ADJ'].head(1).lemma.item()
		else:
			#if only verb, det, etc is marked as Entity
			return ""

	#takes main_df, entity_df and calculated polarity_score of the context of the entity
	def processDF(self, main_df, entity_df, rels, docPol):

		"""
        Extracts features for each main entity
        :param main_df: main dataframe of the article
        :param entity_df: subset of the main_df containing only one entity's data
        :param context_polarity: precalculated polarity of the entity-context
        :return: feature vector
        """	  
		df = entity_df.drop_duplicates(subset=['lemma'], keep='last')
		df = entity_df[entity_df['POS_tag'] != 'VERB']
		# df.drop(df[df['POS_tag'] == 'VERB'].index, axis=0, inplace=True)
		lex = get_opinion(self.lex_path)
		poli = getWindowPol(main_df, entity_df, rels)
		# create features for each entity
		isPerson = 1 if df["ner"].str.contains('PER\\[*').any() else 0 # true if there is at least one PERSon type in column ner
		isSubject = 1 if df["dependency"].str.contains('nsubj').any() else 0
		isObject = 1 if df["dependency"].str.contains('obj').any() else 0
		main_ent =self.mainEntity(df)
		hasDes = self.hasDescriptors(main_df, entity_df)
		#take its polarity (target variable)ent_df
		pol = df.polarity.iloc[-1].split("-")[0] if len(df.polarity) > 0 else '_'
		#determine main entity
		if pol != '_' : #entities that don't have polarity annotation are discarded
			if main_ent != "": #main entities different than noun, pronomen and adj are also discarded
				df2 = pd.DataFrame({'isPerson':[isPerson],'isSubject':[isSubject],'isObject':[isObject],'hasDescriptors':[hasDes],'contPol':[poli],'docPol':[docPol],'polarity':[pol]})
				return df2

	def hasDescriptors(self, main_df, entity_df):
		"""
		:param main_df: main dataframe of the article
		:param main_ent: main entity
        :return: int flag if the main entity has descriptor words in the neighborhood
		"""
		mentions = entity_df.word.tolist()
		d = {}
		d_pol = {}
		for m in mentions:
			tags = []
			i = main_df.loc[main_df['word']==m].index.values.astype(int)[0]
			if i < 3:
				tags.append(main_df.loc[i+1,'POS_tag'])
				tags.append(main_df.loc[i+2,'POS_tag'])
				tags.append(main_df.loc[i+3,'POS_tag'])
			elif i >= len(main_df)-3:
				tags.append(main_df.loc[i-3,'POS_tag'])
				tags.append(main_df.loc[i-2,'POS_tag'])
				tags.append(main_df.loc[i-1,'POS_tag'])
			else:
				tags.append(main_df.loc[i-3,'POS_tag'])
				tags.append(main_df.loc[i-2,'POS_tag'])
				tags.append(main_df.loc[i-1,'POS_tag'])
				tags.append(main_df.loc[i+1,'POS_tag'])
				tags.append(main_df.loc[i+2,'POS_tag'])
				tags.append(main_df.loc[i+3,'POS_tag'])
		if ('ADJ' in tags) or ('ADV' in tags):
			return 1
		else:
			return 0


if __name__ == "__main__":

	data_path = 'DATA/2267.tsv'#2267.tsv'
	lex_path = 'Lexicon/'
	doc_path = 'Document/SentiNews_document-level.txt'

	nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')
	pd.set_option('display.max_columns', None)
 
	df_total = pd.DataFrame({'isPerson':[],'isSubject':[],'isObject':[],'hasDescriptors':[],'contPol':[],'docPol':[],'polarity':[]})

	# cnt = 0
	# all_files = glob.glob(data_path + "/*.tsv")
	# for filename in all_files:
		# cnt += 1
		# print(cnt)
	# 	# print(filename)
		# print("----------")
	# 	a = Article(filename, lex_path)
	# 	df_article = a.read_data()
	# # 	df_total = df_total.append(df_article, ignore_index=True)
	# # #dataframe from all entity-vectors is saved for classification
	# # df_total.to_csv(r'total_entities.csv', index = False, header=True)

	a = Article(data_path, lex_path, doc_path)
	article_df = a.read_data()
	print(article_df)