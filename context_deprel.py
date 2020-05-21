import stanza
import collections
from collections import defaultdict

nlp = stanza.Pipeline(lang='sl', processors='tokenize,mwt,pos,lemma,depparse')

def findChildren(sent):
    """
    Finds children for each word in sentence
    :param sent: a sentence
    :return: dict(key=parent_node, value=list(child_nodes))
    """ 
    family = collections.defaultdict(list)
    negation ="Polarity=Neg"
    for word in sent.words:
        if int(sent.words[word.head-1].id) == word.head:
            family[sent.words[word.head-1].text].append(word)
        flist = word.feats.split("|") if word.feats != None else []
        pol = -1 if len(flist) > 0 and negation in flist else 1
    return family


def feature_sentiment(sent, pos, neg):
    """
    Finds aspects and assignes them polarity
    :param sent: a sentence
    :param pos: positive words lexicon
    :param neg: negative words lexicon   
    :return: dict(key=word, value=polarity score)
    """ 
    family = findChildren(sent)
    sent_dict = dict()
    opinion_words = neg + pos
    negation ="Polarity=Neg"
    for word in sent.words:
        if word.lemma in opinion_words:
            sentiment = 1 if word.lemma in pos else -1
            # if target is an adverb modifier
            if (word.deprel == "advmod"):
                continue    
            elif (word.deprel == "amod"):
                sent_dict[sent.words[word.head-1].text] = sentiment
                # for opinion words that are adjectives, adverbs or verbs
            else:
                childs = family[word.text] if word.text in family.keys() else []
                if childs != []:
                    for child in childs:
                        # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                        if ((child.deprel == "amod") or (child.deprel == "advmod")) and (child.lemma in opinion_words):
                            sentiment *= 1.5
                        # check for negation words and flip the sign of sentiment
                        flist = child.feats.split("|") if child.feats != None else []
                        pol = -1 if len(flist) > 0 and negation in flist else 1
                        if pol == -1:
                            sentiment *= -1
                    for child in childs:
                        # if verb, check if there's a direct object
                        if (word.upos == "VERB") & (child.deprel == "obj"):                      
                            sent_dict[child.text] = sentiment
                             # check for conjugates (a AND b), then add both to dictionary
                            subchildren = []
                            conj = 0
                            #TODO check with if:
                            for subchild in family[child.text]:
                                if subchild.text == "in":
                                    conj=1
                                if (conj == 1) and (subchild.text != "in"):
                                    subchildren.append(subchild.text)
                                    conj = 0

                            for subchild in subchildren:
                                sent_dict[subchild] = sentiment
                    # check for negation
                    for child in family[sent.words[word.head-1].text]:
                        noun = ""
                        if ((child.deprel == "amod") or (child.deprel == "advmod")) and (child.lemma in opinion_words):
                            sentiment *= 1.5
                        # check for negation words and flip the sign of sentiment
                        flist = child.feats.split("|") if child.feats != None else []
                        pol = -1 if len(flist) > 0 and negation in flist else 1
                        if pol == -1:
                            sentiment *= -1
                
                     #check for nouns
                    for child in family[sent.words[word.head-1].text]:
                        noun = ""
                        if (child.upos == "NOUN") and (child.text not in sent_dict):
                            noun = child.text
                            sent_dict[noun] = sentiment
    return sent_dict

def get_featureSentiment(main_df, lex_path):
    """
    Creates polarity dict on article level
    :param main_df: the article dataFrame
    :param lex_path: location of lexicon files
    :return: dict(key=word, value=polarity score)
    """ 
    pos = main_df.POS_tag.tolist()
    ids = main_df.ent_id.tolist()
    words = main_df.word.tolist()
    tags = dict(zip(words,pos))
    text = " ".join(words)
    ps = dict(zip(words,ids))

    total_sents = []
    dicc = collections.defaultdict(list)

    with open(lex_path+'positive_words_lemmas.txt') as f:
        pos = f.read().splitlines()
    with open(lex_path+'negative_words_lemmas.txt') as f:
        neg = f.read().splitlines()
    doc = nlp(text)
    for sentence in doc.sentences:
        d = feature_sentiment(sentence, pos, neg)
        total_sents.append(d)
    counter = collections.Counter()
    # print(total_sents)
    for d in total_sents:
        for k,v in d.items():
            dicc[k].append(v)
    return dicc
