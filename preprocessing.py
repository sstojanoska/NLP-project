import pandas as pd
import stanza
import re
import numpy as np
import glob
import csv
import sklearn
import nltk
from nltk.corpus import stopwords

class Article:

	def __init__(self, data_path, lex_path):
		self.data_path = data_path
		self.lex_path = lex_path

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



	def mapContext(self, main_df, uni_ents):
		"""
        Creates  polarity dataframe for each entity-context
        :param main_df: main dataframe of the article
        :param uni_ents: unique entity IDs
        :return: polarity dataframe
        """
		#create DataFrame where each row contains one sentence
		sentence_df = pd.DataFrame(' '.join(main_df.lemma.values).split('.'), columns=['sentence'])
		ls_indexes = [0] + main_df[main_df['lemma']=='.'].index.values.tolist() + [main_df.tail(1).index.item()]
		ranges = [(ls_indexes[n], ls_indexes[n+1]) for n in range(len(ls_indexes)-1)]
		#create list of entities for each sentence
		ents = [list(set(main_df.ent_id.take(range(main_df.index.get_loc(a),main_df.index.get_loc(b))).tolist())) for (a,b) in ranges]
		ent_serie= pd.Series(ents)
		sentence_df['ents_sentence'] = ent_serie
		# find Context - finds sentences from the article that are related with each entity
		polarities = [self.findContext(main_df,sentence_df, ent) for ent in uni_ents ]
		polarity_df = pd.DataFrame(uni_ents, columns=['ent_id'])
		# #created DataFrame that will keep the entities and their contexes
		polarity_df['context_pol'] = pd.Series(polarities)
		return polarity_df


	def findContext(self, main_df, sent_df, ent):
		"""
        Gathers context for each entity
        :param main_df: main dataframe of the article
        :param sent_df: dataframe containing[sentence, list of entity IDs]
        :return: int (1 positive, -1 negative, 0 neutral context )
        """
		sentence_df = sent_df.dropna()
		#sentence that contains only that entity
		mask_mine = sentence_df.ents_sentence.apply(lambda x: ent in x and len(x)==2)
		#sentence that does not have ANY entity
		mask_alls = sentence_df.ents_sentence.apply(lambda x: len(x)==1)
		#sentence that has multiple entitie (TODO: DEAL FURTHER)
		mask_splited = sentence_df.ents_sentence.apply(lambda x: ent in x and len(x) > 2)
		context = list(set(sentence_df.sentence[mask_mine].tolist()))
		context = "".join(context).split(" ")
		entity_context = [c for c in context if c!='']
		ent_polarity = self.getPolarity(main_df, ent, entity_context) if len(entity_context) > 0 else 0 
		return ent_polarity

	def getPolarity(self, main_df, ent, context):
		"""
        Calculates polarity score of an entity-context
        :param main_df: main dataframe of the article
        :param ent: main entity
        :param context: list of words which are in the same sentence as main entity
        :return: int (1 positive, -1 negative, 0 neutral context )
        """
		polarities = []
		polarities = [main_df[main_df['lemma']==c].polar_score.values[0] for c in context]
		nums = [p for p in polarities if not np.isnan(p)]
		s = sum(nums)
		if s > 1:
				return 1
		elif (s < 1 and s > -1):
			return 0
		else:
			return -1
			

	def join_negation(self, main_df):
		"""
        Markes words preceeded by negated 'to be' verb to keep the negation
        :param main_df: main dataframe of the article
        :return: dataframe with plus one column 'isNegated'
        """
		main_df['isNegated'] = 0
		be_negated = ['nisva','nismo','nisem', 'nista', 'niste', 'nisi', 'niso', 'ni' ]
		negation_idxs = main_df.word.apply(lambda x: x in be_negated)
		for i in main_df.word[negation_idxs].index.tolist():
			main_df.loc[i+1,'isNegated'] = 1
		return main_df

	def removeStopwords(self,main_df):
		"""
        Removes stopwords from the main dataframe
        :param main_df: main dataframe of the article
        :return: updated dataframe
        """
		stop = stopwords.words('slovene')
		mask_stop = main_df.word.apply(lambda x: x not in stop)
		fixed_df = main_df[mask_stop]
		return fixed_df

	def removeSpecialChars(self, main_df):
		main_df = main_df[~main_df.word.str.contains(r'\d', na=False)]
		main_df = main_df[~main_df.word.str.contains(r'[a-zA-Z]+\.[a-zA-Z]+\.*[a-zA-Z]*', na=False)] #remove abbreviations
		spec_chars = ["!",'"',"#","%","&","'","(",")","*","+","-","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~","–"]
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
		lex = pd.read_csv(self.lex_path, index_col=0, sep='\t',  names=['lemma','polar_score','freq','avg_AFINN','sd_AFINN'], header=0)
		d = self.removeSpecialChars(df)
		t = d['word'].tolist()
		ti = ["".join(w.split("'")) for w in t] #apostrophe merged to the word: He's ->Hes
		dependency, lemma, pos = self.lemmatize_dependency(ti)
		d['dependency'] = dependency
		d['lemma'] = lemma
		d['POS_tag'] = pos
		expanded_df = self.unnesting(d)
		uni_ents = [e for e in expanded_df.ent_id.unique().tolist() if e != '_']
		negated_df = self.join_negation(expanded_df)
		fixed_df= self.removeStopwords(negated_df)
		#merge them into one DF
		merged = fixed_df.merge(lex, on='lemma', how='left')
		merged.drop(['ni','nan','freq','avg_AFINN','sd_AFINN'], axis=1, inplace=True)
		polarity_df = self.mapContext(merged, uni_ents)
		# make feature vector for each DataFrame
		article_df = self.createFeatures(merged, polarity_df)
		return article_df

	def createFeatures(self, main_df, polarity_df):
		"""
        Creates features for each entity dataframe
        :param main_df: main dataframe of the article
        :param polarity_df: dataframe [entity, its_context_polarity]
        :return: dataframe where each row is feature vector of an entity
        """	
		df_empty = pd.DataFrame({'isPerson' : [],'isSubject':[],'isObject':[],'isNegated':[],'hasClues':[],'hasDescriptors':[],'contPol':[],'polarity':[]})
		for i, g in main_df.groupby('ent_id'):
			if i != '_':
				context_polarity = polarity_df[polarity_df['ent_id']==i].context_pol.item()
				df2 = self.processDF(main_df,g, context_polarity)
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
	def processDF(self, main_df, entity_df, context_polarity):
		"""
        Extracts features for each main entity
        :param main_df: main dataframe of the article
        :param entity_df: subset of the main_df containing only one entity's data
        :param context_polarity: precalculated polarity of the entity-context
        :return: feature vector
        """			
		df = entity_df.drop_duplicates(subset=['lemma'], keep='last')
		#create features for each entity
		isPerson = 1 if df["ner"].str.contains('PER\\[*').any() else 0 # true if there is at least one PERSon type in column ner
		hasClues = 0 if np.isnan(df["polar_score"]).all() else 1 #true if all polarity scores are Nan, else False
		isSubject = 1 if df["dependency"].str.contains('nsubj').any() else 0
		isObject = 1 if df["dependency"].str.contains('obj').any() else 0
		isNegated = 1 if 1 in df["isNegated"].tolist() else 0
		main_ent =self.mainEntity(df)
		hasDes = 1 if df["POS_tag"].str.contains('VERB').any() else self.hasDescriptors(main_df, main_ent)
		#take its polarity (target variable)
		pol = df.polarity.iloc[-1].split("-")[0]
		#determine main entity
		if pol != '_' : #entities that don't have polarity annotation are discarded
			if main_ent != "": #main entities different than noun, pronomen and adj are also discarded
				df2 = pd.DataFrame({'isPerson':[isPerson],'isSubject':[isSubject],'isObject':[isObject],'isNegated':[isNegated],'hasClues':[hasClues],'hasDescriptors':[hasDes],'contPol':[context_polarity],'polarity':[pol]})
				return df2

	def hasDescriptors(self, main_df, main_ent):
		"""
		:param main_df: main dataframe of the article
		:param main_ent: main entity
        :return: int flag if the main entity has descriptor words in the neighborhood
		"""
		tags = []
		indices = main_df[main_df['lemma']==main_ent].index.values
		for i in indices:
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

	data_path = 'DATA'
	lex_path = 'Lexicon/lex.txt'

	nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')
	pd.set_option('display.max_columns', None)
 
	df_total = pd.DataFrame({'isPerson':[],'isSubject':[],'isObject':[],'isNegated':[],'hasClues':[],'hasDescriptors':[],'contPol':[],'polarity':[]})

	all_files = glob.glob(data_path + "/*.tsv")
	for filename in all_files:
		a = Article(filename, lex_path)
		df_article = a.read_data()
		df_total = df_total.append(df_article, ignore_index=True)
	#dataframe from all entity-vectors is saved for classification
	df_total.to_csv(r'total_entities.csv', index = False, header=True)
	