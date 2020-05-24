1.csv : This model has the following features 
('isPerson', 'isSubject','isObject','isNegated','hasClues','hasDescriptors','contPol','polarity')

isPerson:  checks whether an entity subset has a word with 'PER' named-entity tag

isSubject: checks if a word in the subset has a relation 'nsubj' meaning that this word occurs to be subject 		   within some sentence

isNegated: marks whether the word was preceded by a negation verb

hasClues:  checks if the entity subset contains a polarized word

hasDescriptors: descriptor words are verbs, adjectives and adverbs, and this feature has a value of 1 if a 			word in the subset has a POS tag 'VERB'. If this is not the case than it is checked if in the 			window of 3 words, the main entity contains some descriptor words. This is done considering 			the fact that descriptor words are more likely to express some emotion. 

contPol: 	contains useful information for the polarity of an entity. To extract context for each 	      			entity, the main article is transformed with respect to the sentences of the article. Once 			the sentences are extracted and their entities are known each entity gets a sentence to be in 			his context if it is the only entity there. It is assumed that entities which will not have 			context to be neutral because in most cases it happens to be locations or organizations.
'polarity':	target variable
______________________________________
RESULTS: NB:
 precision    recall  f1-score   support

           0       0.80      0.91      0.85      2117
           1       0.40      0.22      0.29       617

    accuracy                           0.75      2734
   macro avg       0.60      0.56      0.57      2734
weighted avg       0.71      0.75      0.72      2734

[[1917  200]
 [ 481  136]]
_____________________________
RFC:
     precision    recall  f1-score   support

           0       0.83      0.67      0.74      2117
           1       0.31      0.52      0.39       617

    accuracy                           0.63      2734
   macro avg       0.57      0.59      0.56      2734
weighted avg       0.71      0.63      0.66      2734

[[1408  709]
 [ 298  319]]
________________________________________________________
SVM:
 precision    recall  f1-score   support

           0       0.83      0.66      0.74      2117
           1       0.31      0.53      0.39       617

    accuracy                           0.63      2734
   macro avg       0.57      0.59      0.56      2734
weighted avg       0.71      0.63      0.66      2734

[[1402  715]
 [ 292  325]]
_________________________________
LR:
precision    recall  f1-score   support

           0       0.81      0.69      0.74      2117
           1       0.29      0.44      0.35       617

    accuracy                           0.63      2734
   macro avg       0.55      0.57      0.55      2734
weighted avg       0.69      0.63      0.66      2734

[[1457  660]
 [ 343  274]]
__________________________________
GB:
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.1887776622963044: 
       0     1    Error    Rate
-----  ----  ---  -------  --------------
0      930   604  0.3937   (604.0/1534.0)
1      188   298  0.3868   (188.0/486.0)
Total  1118  902  0.3921   (792.0/2020.0)

f1: [[0.22573346486753415, 0.42709189544632586]]

_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
2.csv: This model has the same features as 1.csv, the difference is that the verb mentions are discarded (as it was a feedback on midterm defense)

____________________

RESULTS: NB
 precision    recall  f1-score   support

           0       0.81      0.87      0.84      2002
           1       0.40      0.29      0.34       588

    accuracy                           0.74      2590
   macro avg       0.61      0.58      0.59      2590
weighted avg       0.72      0.74      0.73      2590

[[1751  251]
 [ 418  170]]
________________
NB: pos-neg
 precision    recall  f1-score   support

           0       0.64      0.31      0.42       297
           1       0.51      0.81      0.63       267

    accuracy                           0.55       564
   macro avg       0.58      0.56      0.52       564
weighted avg       0.58      0.55      0.52       564

[[ 92 205]
 [ 51 216]]

_________________________
RF:
 precision    recall  f1-score   support

           0       0.84      0.61      0.71      2002
           1       0.32      0.61      0.42       588

    accuracy                           0.61      2590
   macro avg       0.58      0.61      0.56      2590
weighted avg       0.72      0.61      0.64      2590

[[1231  771]
 [ 232  356]]
______________
RF: pos-neg
 precision    recall  f1-score   support

           0       0.59      0.40      0.48       297
           1       0.51      0.70      0.59       267

    accuracy                           0.54       564
   macro avg       0.55      0.55      0.53       564
weighted avg       0.56      0.54      0.53       564

[[119 178]
 [ 81 186]]
___________________________
SVM:
  precision    recall  f1-score   support

           0       0.84      0.61      0.71      2002
           1       0.32      0.61      0.42       588

    accuracy                           0.61      2590
   macro avg       0.58      0.61      0.56      2590
weighted avg       0.72      0.61      0.64      2590

[[1222  780]
 [ 229  359]]
____________________________
LR: 
precision    recall  f1-score   support

           0       0.82      0.71      0.76      2002
           1       0.33      0.49      0.39       588

    accuracy                           0.66      2590
   macro avg       0.58      0.60      0.58      2590
weighted avg       0.71      0.66      0.68      2590

[[1417  585]
 [ 302  286]]
____________________
GB:
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.2332569841294657: 
       0     1    Error    Rate
-----  ----  ---  -------  --------------
0      939   572  0.3786   (572.0/1511.0)
1      146   242  0.3763   (146.0/388.0)
Total  1085  814  0.3781   (718.0/1899.0)

[[0.21840076977766854, 0.41596569355104734]]



___________________________________________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
3.csv: This model has the same features as 1.csv + added feature documentPolarity generated using sentiNews dataset and it has value 1 if the document is polar, otherwise 0.

________________________
RESULTS:
NB: precision    recall  f1-score   support

           0       0.81      0.87      0.84      2002
           1       0.39      0.29      0.33       588

    accuracy                           0.74      2590
   macro avg       0.60      0.58      0.59      2590
weighted avg       0.71      0.74      0.72      2590

[[1740  262]
 [ 417  171]]
___________________________
RF:
 precision    recall  f1-score   support

           0       0.83      0.73      0.78      2002
           1       0.36      0.50      0.42       588

    accuracy                           0.68      2590
   macro avg       0.60      0.62      0.60      2590
weighted avg       0.73      0.68      0.70      2590
_____________
RF: pos-neg
 precision    recall  f1-score   support

           0       0.73      0.66      0.69       297
           1       0.66      0.73      0.69       267

    accuracy                           0.69       564
   macro avg       0.70      0.70      0.69       564
weighted avg       0.70      0.69      0.69       564

[[196 101]
 [ 72 195]]
______________________________
SVM:
precision    recall  f1-score   support

           0       0.84      0.71      0.77      2002
           1       0.35      0.52      0.42       588

    accuracy                           0.67      2590
   macro avg       0.59      0.62      0.59      2590
weighted avg       0.73      0.67      0.69      2590

[[1431  571]
 [ 281  307]]
__________________
SVM: pos-neg
precision    recall  f1-score   support

           0       0.78      0.64      0.70       297
           1       0.67      0.80      0.73       267

    accuracy                           0.71       564
   macro avg       0.72      0.72      0.71       564
weighted avg       0.73      0.71      0.71       564

[[190 107]
 [ 54 213]]
___________________
neural:
 precision    recall  f1-score   support

           0       0.72      0.70      0.71       297
           1       0.68      0.70      0.69       267

    accuracy                           0.70       564
   macro avg       0.70      0.70      0.70       564
weighted avg       0.70      0.70      0.70       564

[[207  90]
 [ 80 187]]

___________________________
LR:
   precision    recall  f1-score   support

           0       0.83      0.69      0.75      2002
           1       0.33      0.52      0.40       588

    accuracy                           0.65      2590
   macro avg       0.58      0.60      0.58      2590
weighted avg       0.72      0.65      0.67      2590

[[1382  620]
 [ 285  303]]
_________________
LR: pos-neg
   precision    recall  f1-score   support

           0       0.76      0.65      0.70       297
           1       0.66      0.78      0.72       267

    accuracy                           0.71       564
   macro avg       0.71      0.71      0.71       564
weighted avg       0.72      0.71      0.71       564

[[192 105]
 [ 60 207]]
_______________________________
GB:
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.22584542498430024: 
       0     1    Error    Rate
-----  ----  ---  -------  --------------
0      1078  433  0.2866   (433.0/1511.0)
1      179   209  0.4613   (179.0/388.0)
Total  1257  642  0.3223   (612.0/1899.0)

[[0.2050779760003741, 0.43082251082251083]]

_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
4.csv: This model has the same features as 1.csv + docPol except 'contPol': semantic role labeling is used to determine whether the neighrbourhood of words is related to specific mention or other as 'contPol' feature: (-isNegated, -hasClauses)

Dependency parser builds a grammar tree from each sentence given in a document. Using this ability and having entities marked, we search for a shortest path connecting an entity with an opinion word. An opinion word is a word whoose lemma is contained in the lexicon and its polarity is different than zero. In this way, each entity ocurence has the polarity of its pair opinion word. Overall contextual polarity of an entity is gathered from the polarities of its coreferences. Since one entity has many occurences and the polarity of an opinion word can be from -5 till 5 using  the JOB lexicon, the final contPol(context polarity) of an entity finds minimum and maximum value of all its mention's polarity scores in the following way
x- list
f(x)  = { 1 if min(x) < -1 and max(x) > 1, 0 otherwise }
_________________________________
RESULTS:
NB
   precision    recall  f1-score   support

           0       0.79      0.93      0.86      2002
           1       0.42      0.17      0.25       588

    accuracy                           0.76      2590
   macro avg       0.61      0.55      0.55      2590
weighted avg       0.71      0.76      0.72      2590

[[1864  138]
 [ 486  102]]
________________________________
RF:
 precision    recall  f1-score   support

           0       0.85      0.60      0.71      2002
           1       0.32      0.63      0.42       588

    accuracy                           0.61      2590
   macro avg       0.58      0.62      0.56      2590
weighted avg       0.73      0.61      0.64      2590

[[1210  792]
 [ 218  370]]
___________________
RF: pos-neg
precision    recall  f1-score   support

           0       0.82      0.57      0.67       297
           1       0.64      0.86      0.74       267

    accuracy                           0.71       564
   macro avg       0.73      0.72      0.70       564
weighted avg       0.73      0.71      0.70       564

[[170 127]
 [ 38 229]]

_____________________________________________________
SVM:
     precision    recall  f1-score   support

           0       0.84      0.63      0.72      2002
           1       0.32      0.61      0.42       588

    accuracy                           0.62      2590
   macro avg       0.58      0.62      0.57      2590
weighted avg       0.73      0.62      0.65      2590

[[1258  744]
 [ 231  357]]
____________________
SVM pos-neg
precision    recall  f1-score   support

           0       0.82      0.57      0.67       297
           1       0.64      0.86      0.74       267

    accuracy                           0.71       564
   macro avg       0.73      0.72      0.70       564
weighted avg       0.73      0.71      0.70       564

[[170 127]
 [ 38 229]]
___________________
neural:
  precision    recall  f1-score   support

           0       0.71      0.69      0.70       297
           1       0.67      0.69      0.68       267

    accuracy                           0.69       564
   macro avg       0.69      0.69      0.69       564
weighted avg       0.69      0.69      0.69       564

[[206  91]
 [ 84 183]]

____________________________________________________
LR:
  precision    recall  f1-score   support

           0       0.83      0.69      0.75      2002
           1       0.33      0.52      0.41       588

    accuracy                           0.65      2590
   macro avg       0.58      0.61      0.58      2590
weighted avg       0.72      0.65      0.67      2590

[[1378  624]
 [ 280  308]]
___________________
LR: pos-neg
  precision    recall  f1-score   support

           0       0.82      0.59      0.68       297
           1       0.65      0.85      0.74       267

    accuracy                           0.71       564
   macro avg       0.73      0.72      0.71       564
weighted avg       0.74      0.71      0.71       564

[[174 123]
 [ 39 228]]
_____________________________________________________
GB:
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.1977310427590964: 
       0    1    Error    Rate
-----  ---  ---  -------  --------------
0      854  657  0.4348   (657.0/1511.0)
1      132  256  0.3402   (132.0/388.0)
Total  986  913  0.4155   (789.0/1899.0)

[[0.2078767088089647, 0.41345525800130634]]
_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
5.csv: This model has the same features as 1.csv (-isNegated, hasCLues) except 'contPol': as entity context a window of size +-3 is taken around all entity mentiones and all their polarities are assigned to the entity


RESULTS:
NB:
     precision    recall  f1-score   support

           0       0.80      0.92      0.85      2002
           1       0.42      0.19      0.26       588

    accuracy                           0.76      2590
   macro avg       0.61      0.56      0.56      2590
weighted avg       0.71      0.76      0.72      2590

[[1850  152]
 [ 476  112]]
__________
nb: pos-neg
 precision    recall  f1-score   support

           0       0.82      0.60      0.69       297
           1       0.66      0.85      0.74       267

    accuracy                           0.72       564
   macro avg       0.74      0.72      0.72       564
weighted avg       0.74      0.72      0.71       564

_____________________________________________________
RF:
  precision    recall  f1-score   support

           0       0.83      0.71      0.77      2002
           1       0.34      0.50      0.40       588

    accuracy                           0.67      2590
   macro avg       0.58      0.61      0.59      2590
weighted avg       0.72      0.67      0.68      2590

[[1428  574]
 [ 293  295]]
_____________________________________
RF: pos-neg
precision    recall  f1-score   support

           0       0.81      0.60      0.69       297
           1       0.66      0.85      0.74       267

    accuracy                           0.72       564
   macro avg       0.74      0.72      0.72       564
weighted avg       0.74      0.72      0.71       564

[[179 118]
 [ 41 226]]
____________________________________________________
SVM: 
precision    recall  f1-score   support

           0       0.84      0.68      0.75      2002
           1       0.34      0.55      0.42       588

    accuracy                           0.65      2590
   macro avg       0.59      0.62      0.58      2590
weighted avg       0.72      0.65      0.68      2590

[[1363  639]
 [ 265  323]]
____________________________________
svm: pos-neg
precision    recall  f1-score   support

           0       0.82      0.60      0.69       297
           1       0.66      0.85      0.74       267

    accuracy                           0.72       564
   macro avg       0.74      0.73      0.72       564
weighted avg       0.74      0.72      0.72       564
__________________________________
neural:
precision    recall  f1-score   support

           0       0.73      0.71      0.72       297
           1       0.69      0.70      0.70       267

    accuracy                           0.71       564
   macro avg       0.71      0.71      0.71       564
weighted avg       0.71      0.71      0.71       564

___________________________________________________
LR:
   precision    recall  f1-score   support

           0       0.83      0.69      0.75      2002
           1       0.33      0.52      0.41       588

    accuracy                           0.65      2590
   macro avg       0.58      0.61      0.58      2590
weighted avg       0.72      0.65      0.67      2590

[[1380  622]
 [ 280  308]]
_______________________________________
LR: pos-neg
 precision    recall  f1-score   support

           0       0.82      0.60      0.69       297
           1       0.65      0.85      0.74       267

    accuracy                           0.72       564
   macro avg       0.73      0.72      0.71       564
weighted avg       0.74      0.72      0.71       564

[[177 120]
 [ 40 227]]
___________________________________________________

GB:
   0     1    Error    Rate
-----  ----  ---  -------  --------------
0      1045  466  0.3084   (466.0/1511.0)
1      177   211  0.4562   (177.0/388.0)
Total  1222  677  0.3386   (643.0/1899.0)

[[0.197656185308871, 0.4147188881869867]]





_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
6.csv  This model has the same features as 1.csv (-isNegated) except 'contPol': to calculate this feature a special walk in the dependency tree is designed which checks the type of relation that connects opinion word. The relations such as 'amod', 'advmod' and 'obj' are considered as strong relationships which contain a polarized word. Moreover this approach checks for negation words and inverts the sentiment if that is the case. This model uses the lexicon KSS which has positive and negative words only without expressed sentiment score.
(to see more: context_deprel.py) 


__________________
RESULTS:
NB pol
 precision    recall  f1-score   support

           0       0.79      0.93      0.86      2002
           1       0.42      0.18      0.25       588

    accuracy                           0.76      2590
   macro avg       0.61      0.55      0.55      2590
weighted avg       0.71      0.76      0.72      2590

[[1862  140]
 [ 485  103]]
________________
NB pos-neg
   precision    recall  f1-score   support

           0       0.79      0.73      0.76       297
           1       0.72      0.78      0.75       267

    accuracy                           0.76       564
   macro avg       0.76      0.76      0.76       564
weighted avg       0.76      0.76      0.76       564

[[218  79]
 [ 59 208]]

____________________________
RF:
 precision    recall  f1-score   support

           0       0.83      0.70      0.76      2002
           1       0.34      0.52      0.41       588

    accuracy                           0.66      2590
   macro avg       0.59      0.61      0.59      2590
weighted avg       0.72      0.66      0.68      2590

[[1403  599]
 [ 281  307]]

---------------------------
RF: pos-neg
 precision    recall  f1-score   support

           0       0.76      0.70      0.73       297
           1       0.70      0.75      0.72       267

    accuracy                           0.73       564
   macro avg       0.73      0.73      0.73       564
weighted avg       0.73      0.73      0.73       564

[[209  88]
 [ 66 201]]

______________________________
SVM: POL
 precision    recall  f1-score   support

           0       0.84      0.72      0.78      2002
           1       0.35      0.52      0.42       588

    accuracy                           0.68      2590
   macro avg       0.60      0.62      0.60      2590
weighted avg       0.73      0.68      0.70      2590

[[1446  556]
 [ 282  306]]
_____________________________
SVM pos-neg:
  precision    recall  f1-score   support

           0       0.78      0.71      0.74       297
           1       0.71      0.78      0.74       267

    accuracy                           0.74       564
   macro avg       0.74      0.74      0.74       564
weighted avg       0.74      0.74      0.74       564

[[210  87]
 [ 59 208]]
_______________________
neural:
  precision    recall  f1-score   support

           0       0.74      0.73      0.74       297
           1       0.70      0.72      0.71       267

    accuracy                           0.73       564
   macro avg       0.72      0.73      0.72       564
weighted avg       0.73      0.73      0.73       564

[[216  81]
 [ 74 193]]
____________________________

LR:
   precision    recall  f1-score   support

           0       0.83      0.69      0.75      2002
           1       0.33      0.52      0.41       588

    accuracy                           0.65      2590
   macro avg       0.58      0.61      0.58      2590
weighted avg       0.72      0.65      0.67      2590

[[1378  624]
 [ 280  308]]
_____________________
LR: pos-neg

 precision    recall  f1-score   support

           0       0.80      0.67      0.73       297
           1       0.69      0.81      0.74       267

    accuracy                           0.74       564
   macro avg       0.74      0.74      0.74       564
weighted avg       0.75      0.74      0.74       564


_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
TODO:7.csv 'contPol' as a concordance

_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
polarTest2NB.csv - This dataset is consisted of rows from 2.csv which are both 'ground-truth' polar (classes 1,2,4,5)
and classified also as polar with Naive Bayes. On this dataset is peformed only positive-negative classification.

RESULTS:

NaiveBayes: pos-neg

              precision    recall  f1-score   support

           0       0.70      0.77      0.73        30
           1       0.63      0.55      0.59        22

    accuracy                           0.67        52
   macro avg       0.66      0.66      0.66        52
weighted avg       0.67      0.67      0.67        52

[[23  7]
 [10 12]]
  ---------------------------------------------------------
RandomForestClassifier: pos-neg

              precision    recall  f1-score   support

           0       0.65      0.67      0.66        30
           1       0.52      0.50      0.51        22

    accuracy                           0.60        52
   macro avg       0.58      0.58      0.58        52
weighted avg       0.59      0.60      0.59        52

[[20 10]
 [11 11]]
 ---------------------------------------------------------
LogisticRegression: pos-neg

              precision    recall  f1-score   support

           0       0.74      0.77      0.75        30
           1       0.67      0.64      0.65        22

    accuracy                           0.71        52
   macro avg       0.70      0.70      0.70        52
weighted avg       0.71      0.71      0.71        52

[[23  7]
 [ 8 14]]
  ---------------------------------------------------------
SVC: pos-neg

              precision    recall  f1-score   support

           0       0.70      0.70      0.70        30
           1       0.59      0.59      0.59        22

    accuracy                           0.65        52
   macro avg       0.65      0.65      0.65        52
weighted avg       0.65      0.65      0.65        52

[[21  9]
 [ 9 13]]
  ---------------------------------------------------------
KNeighborsClassifier: pos-neg

              precision    recall  f1-score   support

           0       0.62      0.77      0.69        30
           1       0.53      0.36      0.43        22

    accuracy                           0.60        52
   macro avg       0.58      0.57      0.56        52
weighted avg       0.58      0.60      0.58        52

[[23  7]
 [14  8]]

_______________________________________________________________________________________________________________
_______________________________________________________________________________________________________________
polarTest6RF.csv - This dataset is consisted of rows from 2.csv which are both 'ground-truth' polar (classes 1,2,4,5)
and classified also as polar with Naive Bayes. On this dataset is peformed only positive-negative classification.

NaiveBayes:

              precision    recall  f1-score   support

           0       0.75      0.78      0.77        51
           1       0.72      0.68      0.70        41

    accuracy                           0.74        92
   macro avg       0.74      0.73      0.73        92
weighted avg       0.74      0.74      0.74        92

[[40 11]
 [13 28]]
RandomForestClassifier:

              precision    recall  f1-score   support

           0       0.79      0.75      0.77        51
           1       0.70      0.76      0.73        41

    accuracy                           0.75        92
   macro avg       0.75      0.75      0.75        92
weighted avg       0.75      0.75      0.75        92

[[38 13]
 [10 31]]
LogisticRegression:

              precision    recall  f1-score   support

           0       0.78      0.71      0.74        51
           1       0.67      0.76      0.71        41

    accuracy                           0.73        92
   macro avg       0.73      0.73      0.73        92
weighted avg       0.73      0.73      0.73        92

[[36 15]
 [10 31]]
SVC:

              precision    recall  f1-score   support

           0       0.82      0.71      0.76        51
           1       0.69      0.80      0.74        41

    accuracy                           0.75        92
   macro avg       0.75      0.76      0.75        92
weighted avg       0.76      0.75      0.75        92

[[36 15]
 [ 8 33]]
KNeighborsClassifier:

              precision    recall  f1-score   support

           0       0.76      0.80      0.78        51
           1       0.74      0.68      0.71        41

    accuracy                           0.75        92
   macro avg       0.75      0.74      0.74        92
weighted avg       0.75      0.75      0.75        92

[[41 10]
 [13 28]]