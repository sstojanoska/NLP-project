import collections



def getWindowPol(main_df, entity_df, rels):
	"""Finds polarity for each entity
	:param main_df: article's dataFrame
	:entity_df: entity'S dataFrame
	:rels: dict(word,sentiment) calculated in conext_deprel.py 
	:return: polarity score of an entity [-1,0,1]
	"""
	mentions = entity_df.word.tolist()
	d = {}
	d_pol = {}
	for m in mentions:
		tags = []
		i = main_df.loc[main_df['word']==m].index.values.astype(int)[0]
		if i < 3:
			tags.append(main_df.loc[i+1,'word'])
			tags.append(main_df.loc[i+2,'word'])
			tags.append(main_df.loc[i+3,'word'])
		elif i >= len(main_df)-3:
			tags.append(main_df.loc[i-3,'word'])
			tags.append(main_df.loc[i-2,'word'])
			tags.append(main_df.loc[i-1,'word'])
		else:
			tags.append(main_df.loc[i-3,'word'])
			tags.append(main_df.loc[i-2,'word'])
			tags.append(main_df.loc[i-1,'word'])
			tags.append(main_df.loc[i+1,'word'])
			tags.append(main_df.loc[i+2,'word'])
			tags.append(main_df.loc[i+3,'word'])
		d[m]=tags
	for k,v in d.items():
		for vx in v:
			if vx in rels.keys():
				d_pol[k] = rels[vx]
	if d_pol != {}:
		fl = collections.Counter([ e for ls in d_pol.values() for e in ls]).most_common(1)[0][0]
		return fl
	else:
		return 0