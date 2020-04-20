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

	def unnesting(self, df):
		df["ent_id"] = df.ent_id.str.split("|")
		df["ner"] = df.ner.str.split("|")
		df["polarity"] = df.polarity.str.split("|")
		dfe = df.explode("ent_id")
		dfn = dfe.explode("ner")
		dfp = dfn.explode("polarity").reset_index()
		return dfp

	def read_data(self):
		d = pd.read_csv(self.data_path, quotechar='"', quoting=csv.QUOTE_NONE,index_col=0, sep='\t',  names=['char_pos', 'word', 'ner', 'polarity', 'ni', 'ent_id','nan'], header=None, skiprows=7)
		lex = pd.read_csv(self.lex_path, index_col=0, sep='\t',  names=['word','polar_score','freq','avg_AFINN','sd_AFINN'], header=0)
		merged = d.merge(lex, on='word', how='left')
		merged.drop(['ni','nan','freq','avg_AFINN','sd_AFINN'], axis=1, inplace=True)
		merged['word'] = np.where(merged['word'] == "'", '"', merged['word']) #convert single quotes to double
		merged = merged[~merged.word.str.contains(r'\d', na=False)]
		merged = merged[~merged.word.str.contains(r'[a-zA-Z]+\.[a-zA-Z]+\.*[a-zA-Z]*', na=False)] #remove abbreviations
		spec_chars = ["!",'"',"#","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~","–"]
		for c in spec_chars:
			merged['word'] = merged['word'].str.replace(c,'')
		merged['word'].replace('', np.nan, inplace=True)
		merged.dropna(subset=['word'], inplace=True)
		t = merged['word'].tolist()
		ti = ["".join(w.split("'")) for w in t] #apostrophe merged to the word: He's ->Hes
		dependency, lemma, pos = self.lemmatize_dependency(ti)
		merged['dependency'] = dependency
		merged['lemma'] = lemma
		merged['POS_tag'] = pos
		exp = self.unnesting(merged)		
		self.createFeatures(exp)

	def createFeatures(self, merged):
		df_empty = pd.DataFrame({'isPerson' : [],'hasClues':[],'isSubject':[],'isObject':[], 'isAmod':[], 'pol_con':[],'polarity':[]})
		for i, g in merged.groupby('ent_id'):
			if i != '_':#TODO: groupby takes '_' as a group, REMOVE IT!!!
				# print(i)
				df2 = self.processDF(merged,g)
				df_empty = df_empty.append(df2, ignore_index=True)
		print(df_empty)
	
	def mainEntity(self, df):
		if not df.loc[df['POS_tag'] == 'PROPN'].empty:
			return df.loc[df['POS_tag'] == 'PROPN'].head(1).lemma.item()
		elif not df.loc[df['POS_tag'] == 'NOUN'].empty:
			return df.loc[df['POS_tag'] == 'NOUN'].head(1).lemma.item()
		elif not df.loc[df['POS_tag'] == 'ADJ'].empty:
			return df.loc[df['POS_tag'] == 'ADJ'].head(1).lemma.item()
		else:
			#verbs++ annotated as entities
			return ""

	def processDF(self, merged, d):
		df = d.drop_duplicates(subset=['lemma'], keep='last')
		isPerson = 1 if df["ner"].str.contains('PER\\[*').any() else 0 # true if there is at least one PERSon type in column ner
		hasClues = 0 if np.isnan(df["polar_score"]).all() else 1#true if all polarity scores are Nan, else False
		isSubject = 1 if df["dependency"].str.contains('nsubj').any() else 0
		isObject = 1 if df["dependency"].str.contains('obj').any() else 0
		isAmod = 1 if df["dependency"].str.contains('amod').any() else 0
		pol = df.polarity.iloc[-1].split("-")[0]
		main_ent =self.mainEntity(df)
		if pol != '_' : #entities that don't have polarity annotation are discarded
			if main_ent != "": #entities different than noun, pronomen and adj are also discarded
				pcontext = self.context_polarity(merged,main_ent)
				df2 = pd.DataFrame({'isPerson':[isPerson], 'hasClues':[hasClues], 'isSubject':[isSubject],'isObject':[isObject], 'isAmod':[isAmod], 'pol_con':[pcontext],'polarity':[pol]})
				return df2

	def context_polarity(self, main_df, main_ent):
		n = 3 #window size
		ls_indexes = main_df[main_df['lemma']==main_ent].index.values
		polarities = []
		for i in ls_indexes:
			if i < 3:
				polarities.append(main_df.loc[i+1,'polar_score'])
				polarities.append(main_df.loc[i+2,'polar_score'])
				polarities.append(main_df.loc[i+3,'polar_score'])
			elif i >= len(main_df)-3:
				polarities.append(main_df.loc[i-3,'polar_score'])
				polarities.append(main_df.loc[i-2,'polar_score'])
				polarities.append(main_df.loc[i-1,'polar_score'])
			else:
				polarities.append(main_df.loc[i-3,'polar_score'])
				polarities.append(main_df.loc[i-2,'polar_score'])
				polarities.append(main_df.loc[i-1,'polar_score'])
				polarities.append(main_df.loc[i+1,'polar_score'])
				polarities.append(main_df.loc[i+2,'polar_score'])
				polarities.append(main_df.loc[i+3,'polar_score'])
		nums = [p for p in polarities if not np.isnan(p)]
		p = np.mean(nums) if len(nums) > 0 else 100 # 100 means no polarity found
		return p

if __name__ == "__main__":

	data_path = 'DATA'
	lex_path = 'Lexicon/lex.txt'
	print(data_path)

	nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')
	pd.set_option('display.max_columns', None)
	
	all_files = glob.glob(data_path + "/*.tsv")
	for filename in all_files:
		a = Article(filename, lex_path)
		a.read_data()