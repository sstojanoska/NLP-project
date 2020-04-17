import pandas as pd
import stanza
import re
import numpy as np
import glob

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
		explode_cols = ['ner','polarity','ent_id']
		idx = df.index.repeat(df[explode_cols[0]].str.len())
		df1 = pd.concat([
			pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode_cols], axis=1)
		df1.index = idx
		return df1.join(df.drop(explode_cols, 1), how='left').reset_index()


	def read_data(self):
		d = pd.read_csv(self.data_path, index_col=0, sep='\t',  names=['char_pos', 'word', 'ner', 'polarity', 'ni', 'ent_id','nan'], header=None, skiprows=7)
		lex = pd.read_csv(self.lex_path, index_col=0, sep='\t',  names=['word','polar_score','freq','avg_AFINN','sd_AFINN'], header=0)
		merged = d.merge(lex, on='word', how='left')
		merged.drop(['ni','nan','freq','avg_AFINN','sd_AFINN'], axis=1, inplace=True)
		t = merged['word'].tolist()
		dependency, lemma, pos = self.lemmatize_dependency(t)
		merged['dependency'] = dependency
		merged['lemma'] = lemma
		merged['POS_tag'] = pos
		exp = self.unnesting(merged)
		self.createFeatures(exp)


	def createFeatures(self, merged):
		df_empty = pd.DataFrame({'isPerson' : [],'hasClues':[],'isSubject':[],'isObject':[], 'isAmod':[], 'pol_con':[],'pscore':[]})
		for i, g in merged.groupby('ent_id'):
			if i != '_':#TODO: groupby takes '_' as a group, REMOVE IT!!!
				df2 = self.processDF(merged,g)
				df_empty = df_empty.append(df2, ignore_index=True)
		print(df_empty)

	def processDF(self, merged, d):
		df = d.drop_duplicates(subset=['lemma'], keep='last')
		isPerson = 1 if df["ner"].str.contains('PER\\[*').any() else 0 # true if there is at least one PERSon type in column ner
		hasClues = 0 if np.isnan(df["polar_score"]).all() else 1#true if all polarity scores are Nan, else False
		isSubject = 1 if df["dependency"].str.contains('nsubj').any() else 0
		isObject = 1 if df["dependency"].str.contains('obj').any() else 0
		isAmod = 1 if df["dependency"].str.contains('amod').any() else 0
		polarity = int(df["polarity"].iloc[-1].split("-")[0]) #for TRAIN set only
		pscore = 1 if polarity != 3 else 0
		main_ent = df[((df['ner'] != '_') & (df['POS_tag'] == 'NOUN')) | ((df['ner'] != '_') & (df['POS_tag'] == 'PROPN')) ].iloc[0].lemma
		pcontext = self.context_polarity(merged,main_ent)
		df2 = pd.DataFrame({'isPerson':[isPerson], 'hasClues':[hasClues], 'isSubject':[isSubject],'isObject':[isObject], 'isAmod':[isAmod], 'pol_con':[pcontext], 'pscore':[pscore]})
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
			elif i > len(main_df)-3:
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
		p = np.mean(nums) if len(nums) > 0 else -1 # -1 means no polarity found
		return p

if __name__ == "__main__":

	#TODO: implement reading multiple files at the same time 
	data_path = 'DATA/10227.tsv'
	lex_path = 'Lexicon/lex.txt'

	nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')
	pd.set_option('display.max_columns', None)

	a = Article(data_path, lex_path)
	a.read_data()


	
