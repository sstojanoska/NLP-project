import pandas as pd
import numpy as np
import csv
import nltk
import collections
import Levenshtein as lev



def get_concordance_polarity(main_df, uni_ents, lex_path):
    """
		Preparing list of entities for finding concordances and their polarity
        :param main_df: the article dataFrame
        :param uni_ents: entites id from articles
        :param lex_path: location of lexicon files
        :return: dataframe with 2 columns ent_id and conc_pol
	"""
    text = nltk.Text(main_df.lemma.tolist())
    c=nltk.ConcordanceIndex(text, key = lambda s: s.lower())
    entities = []
    for ent in uni_ents:
        #extract the words that correspond to the entity id
        word = list(set(main_df.loc[main_df['ent_id'] == ent, 'lemma']))
        temp = []
        for w in word:
            #only continue with words that are PROPN, ADJ or NOUN without duplicating
            if (main_df.loc[main_df['lemma'] == w, 'POS_tag'].head(1).item() in ('PROPN', 'ADJ', 'NOUN')):
                if w.lower() not in temp:
                    temp.append(w.lower())
        entities.append(temp)
    dict_entities = dict(zip(uni_ents, entities))

    polarity_conc = pd.DataFrame(uni_ents, columns=['ent_id'])
    polarities = [polarity_concordances(ent_v, c, text, lex_path) for ent_k, ent_v in dict_entities.items()]
    polarity_conc['conc_pol'] = pd.Series(polarities)

    return polarity_conc

def polarity_concordances(main_ent, c, text, lex_path):
    """
        Finding concordances of main entity and determining their overall polarity
        :param main_ent: the main entity
        :param c: concordance indexes of article
        :param text: list of article lemmas
        :param lex_path: location of lexicon file
        :return: polarity of main entity
    """
    polarities = []
    left_win = 7
    right_win = 7
    lex = pd.read_csv(lex_path + 'lex.txt', sep='\t',  names=['lemma','polar_score','freq','avg_AFINN','sd_AFINN'], header=0)

    concordance_txt = []
    if (len(main_ent)>1):
        for ent in main_ent:
            #map left_win amount of words before the main entity and right_win words after, and all its other appearings in text into one variable 
            conc_text = ([text.tokens[list(map(lambda x: x-left_win if (x-left_win)>0 else 0, [offset]))[0]:offset+right_win] 
            for offset in c.offsets(ent)])
            temp = []
            if(len(conc_text)>0):
                for i in range(1, len(conc_text)):
                    #if Levenshtein ratio is smaller than 0.5 between the two elements in list we add it into final concordances 
                    if (lev.ratio(" ".join(conc_text[i])," ".join(conc_text[i-1])) < 0.5):
                        temp.append(conc_text[i])
            concordance_txt.extend(temp)
    elif(len(main_ent) == 1): 
        concordance_txt = ([text.tokens[list(map(lambda x: x-left_win if (x-left_win)>0 else 0, [offset]))[0]:offset+right_win] 
        for offset in c.offsets(main_ent[0])])

    maksVal = 0
    if (len(concordance_txt) > 0):
        #extraction of polarities directly from lexicon
        polarities = [list(set(lex.loc[lex['lemma'] == c, 'polar_score'])) for conc_sent in concordance_txt for c in conc_sent]
        polarities = [item for subs in polarities for item in subs]
        polarities[:] = [1 if x==2 or x==3 or x==4 or x==5 else -1 if x==-2 or x==-3 or x==-4 or x==-5 else x for x in polarities]

        dict_pol = collections.Counter(polarities)
        if(len(dict_pol)> 0):
            maksVal = max(dict_pol, key=dict_pol.get)

    if maksVal == 1:
        return 1
    elif maksVal == 0:
        return 0
    else:
        return -1