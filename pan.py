import pandas as pd
import stanza
import re
import numpy as np
import glob
import csv

class Article:

	def __init__(self, data_path, lex_path):
		self.data_path = data_path
		self.lex_path = lex_path

	# STANZA pipeline used to: extract lemmas, word dependencies in a sentence, part-of-speech tags
	def lemmatize_dependency(self, word_list):
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

	#some words have multiple marked entities ex: 'Pivovarna Lasko' -> ORG,LOC
	# Pandas explode provides splitting
	def unnesting(self, df):
		df["ent_id"] = df.ent_id.str.split("|") 
		df["ner"] = df.ner.str.split("|")
		df["polarity"] = df.polarity.str.split("|")
		dfe = df.explode("ent_id") #first explode by entities
		dfn = dfe.explode("ner") #next by NER 
		dfp = dfn.explode("polarity").reset_index() # last by polarity
		return dfp

	#CONTEXT extraction
	def mapContext(self, main_df, uni_ents):
		#create DataFrame where each row contains one sentence
		sentence_df = pd.DataFrame(' '.join(main_df.word.values).split('.'), columns=['sentence'])
		ls_indexes = [0] + main_df[main_df['word']=='.'].index.values.tolist() + [main_df.tail(1).index.item()]
		ranges = [(ls_indexes[n], ls_indexes[n+1]) for n in range(len(ls_indexes)-1)]
		#create list of entities for each sentence
		ents = [list(set(main_df.ent_id.take(range(main_df.index.get_loc(a),main_df.index.get_loc(b))).tolist())) for (a,b) in ranges]
		ent_serie= pd.Series(ents)
		#DF : [[sentence, list of entities in that sentence]]
		sentence_df['ents_sentence'] = ent_serie
		# findContext - finds sentences from the article that are related with each entity
		polarities = [self.findContext(main_df,sentence_df, ent) for ent in uni_ents ]
		polarity_df = pd.DataFrame(uni_ents, columns=['ent_id'])
		#created DataFrame that will keep the entities and their contexes for TODO:context_polarity
		polarity_df['context_polarity'] = pd.Series(polarities)
		return polarity_df

	#for each entity find its context
	def findContext(self, main_df, sentence_df, ent):
		#sentence that contains only that entity
		mask_mine = sentence_df.ents_sentence.apply(lambda x: ent in x and len(x)==2)
		#sentence that does not have ANY entity
		mask_alls = sentence_df.ents_sentence.apply(lambda x: len(x)==1)
		#sentence that has multiple entitie (TODO: DEAL FURTHER)
		mask_splited = sentence_df.ents_sentence.apply(lambda x: ent in x and len(x) > 2)
		context = sentence_df.sentence[mask_mine].tolist() + sentence_df.sentence[mask_alls].tolist()
		context = "".join(context).split(" ")
		entity_context = [c for c in context if c!='']
		ent_polarity = self.getPolarity(main_df, ent, entity_context)
		return ent_polarity

	def getPolarity(self, main_df, ent, context):
		polarities = []
		for c in context:
			if not np.isnan(main_df[main_df['word']==c].polar_score.values[0]):
				polarities.append(main_df[main_df['word']==c].polar_score.values[0])
		avg_pol = np.round(np.mean(polarities),4)
		return avg_pol

	# 1. main function
	def read_data(self):
		# read article and lexicon as DataFrame-s
		d = pd.read_csv(self.data_path, quotechar='"', quoting=csv.QUOTE_NONE,index_col=0, sep='\t',  names=['char_pos', 'word', 'ner', 'polarity', 'ni', 'ent_id','nan'], header=None, skiprows=7)
		lex = pd.read_csv(self.lex_path, index_col=0, sep='\t',  names=['word','polar_score','freq','avg_AFINN','sd_AFINN'], header=0)
		#merge them into one DF
		merged = d.merge(lex, on='word', how='left')
		#remove Columns that are not needed
		merged.drop(['ni','nan','freq','avg_AFINN','sd_AFINN'], axis=1, inplace=True)
		#replace QUOTEs to double (since there are articles that contain diff types of quotes and Pandas reads WRONG the data)
		merged['word'] = np.where(merged['word'] == "'", '"', merged['word']) #convert single quotes to double
		#removes numerical data
		merged = merged[~merged.word.str.contains(r'\d', na=False)]
		#removes Abbreviations (TODO:change it to take abbreviations just remove dots inside it)
		merged = merged[~merged.word.str.contains(r'[a-zA-Z]+\.[a-zA-Z]+\.*[a-zA-Z]*', na=False)] #remove abbreviations
		#remove special characters
		spec_chars = ["!",'"',"#","%","&","'","(",")","*","+",",","-","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~","–"]
		for c in spec_chars:
			merged['word'] = merged['word'].str.replace(c,'')
		merged['word'].replace('', np.nan, inplace=True)
		merged.dropna(subset=['word'], inplace=True)
		t = merged['word'].tolist()
		ti = ["".join(w.split("'")) for w in t] #apostrophe merged to the word: He's ->Hes
		#get dependencies, lemmas and POS tags
		dependency, lemma, pos = self.lemmatize_dependency(ti)
		#add them to the main DataFrame
		merged['dependency'] = dependency
		merged['lemma'] = lemma
		merged['POS_tag'] = pos
		#EXPLODE the entities
		exp = self.unnesting(merged)
		#get unique entities
		uni_ents = [e for e in exp.ent_id.unique().tolist() if e != '_']
		#get context for each entity
		polarity_df = self.mapContext(exp, uni_ents)
		# make feature vector for each DataFrame
		self.createFeatures(exp, polarity_df)

	def createFeatures(self, main_df, polarity_df):
		#this DataFrame will have feature vector as row
		df_empty = pd.DataFrame({'isPerson' : [],'hasClues':[],'isSubject':[],'isObject':[], 'isAmod':[], 'pol_con':[],'polarity':[]})
		for i, g in main_df.groupby('ent_id'):
			if i != '_':#TODO: groupby takes '_' as a group, REMOVE IT!!!
				polar_score = polarity_df[polarity_df['ent_id']==i].context_polarity.item()
				# process each entity_dataframe
				df2 = self.processDF(main_df,g, polar_score)
				#returned feature vector append it to the df_empty DF
				df_empty = df_empty.append(df2, ignore_index=True)
		print(df_empty)
	
	#this is method for finding Main Entity
	#each article has several entities and their occurencies
	#DISCLAIMER: Maybe there is better way to determine it!
	def mainEntity(self, df):
		# first check if in the DF there is a word which is Proper noun
		if not df.loc[df['POS_tag'] == 'PROPN'].empty:
			return df.loc[df['POS_tag'] == 'PROPN'].head(1).lemma.item()
		#next check if the DF contains Noun
		elif not df.loc[df['POS_tag'] == 'NOUN'].empty:
			return df.loc[df['POS_tag'] == 'NOUN'].head(1).lemma.item()
		#last take ADJ from the DF if none of the above exists
		elif not df.loc[df['POS_tag'] == 'ADJ'].empty:
			return df.loc[df['POS_tag'] == 'ADJ'].head(1).lemma.item()
		else:
			#if only verb, det, etc is marked as Entity
			return ""

	#takes main_df, entity_df and calculated polarity_score of the context of the entity
	def processDF(self, main_df, entity_df, polar_score):
		df = entity_df.drop_duplicates(subset=['lemma'], keep='last')
		#create features for each entity
		isPerson = 1 if df["ner"].str.contains('PER\\[*').any() else 0 # true if there is at least one PERSon type in column ner
		hasClues = 0 if np.isnan(df["polar_score"]).all() else 1#true if all polarity scores are Nan, else False
		isSubject = 1 if df["dependency"].str.contains('nsubj').any() else 0
		isObject = 1 if df["dependency"].str.contains('obj').any() else 0
		isAmod = 1 if df["dependency"].str.contains('amod').any() else 0
		#take its polarity (target variable)
		pol = df.polarity.iloc[-1].split("-")[0]
		#determine main entity
		main_ent =self.mainEntity(df)
		if pol != '_' : #entities that don't have polarity annotation are discarded
			if main_ent != "": #entities different than noun, pronomen and adj are also discarded
				df2 = pd.DataFrame({'isPerson':[isPerson], 'hasClues':[hasClues], 'isSubject':[isSubject],'isObject':[isObject], 'isAmod':[isAmod], 'pol_con':[polar_score],'polarity':[pol]})
				return df2

if __name__ == "__main__":

	data_path = 'DATA/1.tsv'
	lex_path = 'Lexicon/lex.txt'
	print(data_path)

	nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')
	pd.set_option('display.max_columns', None)
	
	#RUN through all files
	# all_files = glob.glob(data_path + "/*.tsv")
	# for filename in all_files:
	# 	a = Article(filename, lex_path)
	# 	a.read_data()

	#take only one one article
	a = Article(data_path, lex_path)
	a.read_data()