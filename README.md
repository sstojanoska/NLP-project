# NLP: Aspect-based sentiment analysis on news articles
Authors: Dina Sarajlić, Sanja Stojanoska, Vanda Antolović 

## Introduction

This is project for NLP course. The goal is to determine entities sentiment on a 3-level sentiment scale.

### Prerequisites

- Dataset: [SentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285)
- Lexicons: [Slovene sentiment lexicon JOB 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1112)
[Slovene sentiment lexicon KSS 1.1](https://www.clarin.si/repository/xmlui/handle/11356/1097)
- Install and download model from: [Stanza - Python NLP Library](https://stanfordnlp.github.io/stanza/)
```
pip install stanza
stanza.download('sl')

```
- Install and download model from: [NLTK](https://www.nltk.org/)
```
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
```
- Install [NetworkX](https://networkx.github.io/)
```
pip install networkx
```
- Install [Levenshtein](https://pypi.org/project/python-Levenshtein/)


### To run preprocessing:

```
python3 preprocessing.py
```

### To run classification:
Classification notebook: [classification](https://colab.research.google.com/drive/1EF5_KOYqLbFAYKPcAWs5f-7aSklYYdqQ?usp=sharing)








