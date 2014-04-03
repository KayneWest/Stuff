import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk import bigrams, trigrams
import math
from collections import Counter
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords



#second type of skills counter
with open("ALLCLEAN6.txt", "r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	for line in openfile:
		frequency = Counter() 
		frequency2 = Counter()
		words1 = line.lower().strip()
		words = re.sub(r'[0-9]','',words1)
		words = pattern.sub('', words)
		words = words.split('\t')
		tokens = [s.lstrip() for s in words]
		tokens1 = [tuple(i.split()) for i in tokens]
		frequency.update(tokens1)
		tokens2 = [token for token in tokens if token not in Stopwords]
		tokens2 = [x for x in tokens2 if not x.startswith('-')]
		frequency2.update(tokens2)
SKILLS=frequency.keys()
SKILLS2=frequency2.keys()


#second type of skills counter version 2
with open("ALLCLEAN6.txt", "r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	for line in openfile:
		frequency = Counter() 
		frequency2 = Counter()
		words1 = line.lower().strip()
		words = re.sub(r'[0-9]','',words1)
		words = pattern.sub('', words)
		words = words.split('\t')
		tokens = [s.lstrip() for s in words]
		tokens1 = [tuple(i.split()) for i in tokens]
		frequency.update(tokens1)
SKILLS=frequency.keys()


#TechSkillsOnly
with open("techskills2.txt", "r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	for line in openfile:
		frequency = Counter() 
		frequency2 = Counter()
		words1 = line.lower().strip()
		words = re.sub(r'[0-9]','',words1)
		words = pattern.sub('', words)
		words = words.split('\r')
		tokens = [s.lstrip() for s in words]
		tokens1 = [tuple(i.split()) for i in tokens]
		frequency.update(tokens1)
TechSKILLS=frequency.keys()




sa = Site('stackoverflow')
for p in count(1):
	tags = sa.tags.page(p)
	for t in tags:
		print t
	if not tags['has_more']:
		break








#Undirected Graph
#Boolean CoOccurrenceCounter 
with open("datascience545.txt","r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	WordGramWordGram= Counter()
	WordGramBiGram = Counter()
	WordGramTriGram = Counter()
	BiGramBiGram = Counter()
	BiGramTriGram = Counter()
	TriGramTriGram = Counter()
	tokenizer = WhitespaceTokenizer()
	for line in openfile:
		words = line.lower().strip().replace('(',',').replace(')',',')
		words=re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\.|\>|\?|\/|\;|\:|\"|\'', '',words)
		words = pattern.sub('', words)
		words=words.split('\r')
		words = [s.lstrip() for s in words]
		ReservoirALL={}
		for word in words:
			CountWordGrams = Counter()
			CountBiGrams = Counter()
			CountTriGrams = Counter()
			
			wordsplit= tokenizer.tokenize(word)
			wordsplit = [s.lstrip() for s in wordsplit]
			NoDupes = list(set(wordsplit))
			TuplesNoDupes=[tuple(i.split()) for i in NoDupes]
			skillsonly=[x for x in TuplesNoDupes if x in SKILLS]
			skillsclean = [token for token in skillsonly if token not in Stopwords]
			
			BiGrams=bigrams(wordsplit)
			NoDupesBiGrams = list(set(BiGrams))
			BiGrams=[x for x in NoDupesBiGrams if x in SKILLS]
			TriGrams=trigrams(wordsplit)
			NoDupesTriGrams = list(set(TriGrams))
			TriGrams=[x for x in NoDupesTriGrams if x in SKILLS]
		
			CountWordGrams.update(skillsclean)
			CountBiGrams.update(BiGrams)
			CountTriGrams.update(TriGrams)

			for key1 in CountWordGrams.keys():
				for key2 in CountWordGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountWordGrams[key2]
			WordGramWordGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			WordGramBiGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			WordGramTriGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			BiGramBiGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			BiGramTriGram.update(ReservoirALL)
			for key1 in CountTriGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			TriGramTriGram.update(ReservoirALL)
COOCCURRENCE = dict(WordGramWordGram.items() + WordGramBiGram.items() + WordGramTriGram.items() + BiGramBiGram.items() + BiGramTriGram.items() + TriGramTriGram.items())

sorted(COOCCURRENCE.items(),key=lambda x:x[1])

FG=nx.Graph()
for key,value in COOCCURRENCE.items():
	FG.add_edges_from([(key[0],key[1])],weight=value)
DG=nx.DiGraph()
for key,value in DIGRAPHCOOCCURRENCE.items():
	DG.add_edges_from([(key[0],key[1])],weight=value)




#DIRECTED Graph Creator
#Boolean CoOccurrenceCounter 
with open("datascience_6.txt","r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	WordGramWordGram= Counter()
	WordGramBiGram = Counter()
	WordGramTriGram = Counter()
	BiGramWordGram = Counter()
	BiGramBiGram = Counter()
	BiGramTriGram = Counter()
	TriGramWordGram = Counter()
	TriGramBiGram = Counter()
	TriGramTriGram = Counter()
	tokenizer = WhitespaceTokenizer()
	for line in openfile:
		words = line.lower().strip().replace('(',',').replace(')',',')
		words=re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\.|\>|\?|\/|\;|\:|\"|\'', '',words)
		words = pattern.sub('', words)
		words=words.split('\r')
		words = [s.lstrip() for s in words]
		ReservoirALL={}
		for word in words:
			CountWordGrams = Counter()
			CountBiGrams = Counter()
			CountTriGrams = Counter()
			
			wordsplit= tokenizer.tokenize(word)
			wordsplit = [s.lstrip() for s in wordsplit]
			NoDupes = list(set(wordsplit))
			skillsonly=[x for x in NoDupes if x in SKILLS2]
			skillsclean = [token for token in skillsonly if token not in Stopwords]
			
			BiGrams=bigrams(wordsplit)
			NoDupesBiGrams = list(set(BiGrams))
			BiGrams=[x for x in NoDupesBiGrams if x in SKILLS]
			TriGrams=trigrams(wordsplit)
			NoDupesTriGrams = list(set(TriGrams))
			TriGrams=[x for x in NoDupesTriGrams if x in SKILLS]
		
			CountWordGrams.update(skillsclean)
			CountBiGrams.update(BiGrams)
			CountTriGrams.update(TriGrams)

			for key1 in CountWordGrams.keys():
				for key2 in CountWordGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountWordGrams[key2]
			WordGramWordGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			WordGramBiGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			WordGramTriGram.update(ReservoirALL)
			
			for key1 in CountBiGrams.keys():
				for key2 in CountWordGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountWordGrams[key2]
			BiGramWordGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			BiGramBiGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			BiGramTriGram.update(ReservoirALL)

			for key1 in CountTriGrams.keys():
				for key2 in CountWordGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountWordGrams[key2]
			TriGramWordGram.update(ReservoirALL)
			for key1 in CountTriGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			TriGramBiGram.update(ReservoirALL)			
			for key1 in CountTriGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			TriGramTriGram.update(ReservoirALL)
DIGRAPHCOOCCURRENCE = dict(WordGramWordGram.items() + WordGramBiGram.items() + WordGramTriGram.items() + BiGramWordGram.items() + BiGramBiGram.items() + BiGramTriGram.items() + TriGramWordGram.items() + TriGramBiGram.items() + TriGramTriGram.items())




























#Co Occurrence Creator

with open("datascience5.txt","r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	WordGramWordGram= Counter()
	WordGramBiGram = Counter()
	WordGramTriGram = Counter()
	BiGramBiGram = Counter()
	BiGramTriGram = Counter()
	TriGramTriGram = Counter()
	tokenizer = WhitespaceTokenizer()
	for line in openfile:
		words = line.lower().strip().replace('(',',').replace(')',',')
		words=re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\.|\>|\?|\/|\;|\:|\"|\'', '',words)
		words = pattern.sub('', words)
		words=words.split('\r')
		words = [s.lstrip() for s in words]
		ReservoirALL={}
		StopwordCreator = Counter()
		for word in words:
			CountWordGrams = Counter()
			CountBiGrams = Counter()
			CountTriGrams = Counter()
			
			wordsplit= tokenizer.tokenize(word)
			wordsplit = [s.lstrip() for s in wordsplit]
			skillsonly=[x for x in wordsplit if x in SKILLS2]
			skillsclean = [token for token in skillsonly if token not in Stopwords]
			
			BiGrams=bigrams(wordsplit)
			BiGrams=[x for x in BiGrams if x in SKILLS]
			TriGrams=trigrams(wordsplit)
			TriGrams=[x for x in TriGrams if x in SKILLS]
		
			StopwordCreator.update(skillsclean)
			CountWordGrams.update(skillsclean)
			CountBiGrams.update(BiGrams)
			CountTriGrams.update(TriGrams)

			for key1 in CountWordGrams.keys():
				for key2 in CountWordGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountWordGrams[key2]
			WordGramWordGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			WordGramBiGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			WordGramTriGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			BiGramBiGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			BiGramTriGram.update(ReservoirALL)
			for key1 in CountTriGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			TriGramTriGram.update(ReservoirALL)	
CoOccurrenceWords = dict(WordGramWordGram.items() + WordGramBiGram.items() + WordGramTriGram.items() + BiGramBiGram.items() + BiGramTriGram.items() + TriGramTriGram.items())

	

FG=nx.Graph()
for key,value in CoOccurrenceWords.items():
	FG.add_edges_from([(key[0],key[1])],weight=value)

A=nx.betweenness_centrality(FG)
sorted(A.items(),key=lambda x:x[1])

shortestpath = nx.dijkstra_path(FG,('research'),('sales'))
shortestpath2=nx.shortest_path(FG,'research','sales')
Gsp3 = nx.subgraph(FG,shortestpath)
Gsp4 = nx.subgraph(FG,shortestpath2)


pos = nx.spring_layout(FG,k=0.5,iterations=20)
nx.draw_networkx_nodes(FG,pos,node_size=100,linewidths=0)
nx.draw_networkx_edges(Gsp3,pos,width=5.0,style='solid')
nx.draw_networkx_edges(Gsp4,pos,width=5.0,style='dotted')
nx.draw_networkx_labels(Gsp3,pos,font_size=15)
nx.draw_networkx_labels(Gsp4,pos,font_size=15)


pos = nx.spring_layout(g,k=0.5,iterations=20)
nx.draw_networkx_nodes(g,pos,node_size=100)
nx.draw_networkx_edges(Gsp3,pos,width=5.0,style='solid')
nx.draw_networkx_labels(Gsp3,pos,font_size=15)


source_list = ['research', 'sql', 'c', ('data', 'analysis'), ('research', 'development')]
target_list= ['sales', 'selling', 'sell', 'marketing', 'communications', ('lead', 'generation'), 'support']
Paths=[]
for i in source_list:
	for n in target_list:
		x=nx.dijkstra_path(FG, source=i,target=n)
		Paths.append(x)
gg = Counter()
for i in Paths:
	gg.update(i)
PathList = gg.keys()
Gsp5 = nx.subgraph(FG,PathList)
pos = nx.spring_layout(FG,k=0.5,iterations=20)
nx.draw_networkx_nodes(FG,pos,node_size=100,linewidths=0)
nx.draw_networkx_edges(Gsp5,pos,width=1,edge_color='b')
nx.draw_networkx_labels(Gsp5,pos,font_size=15)
plt.show()


g1=Counter()
for i in Paths:
	g1.update(i)
aa=g1.keys()

res = {}
for i in Paths:
	gg = Counter()
	gg.update(i)
	for key1 in gg.keys():
				for key2 in gg.keys():
					if key1 != key2:
						res[(key1, key2)] = gg[key2]
G=nx.Graph()
for key,value in res.items():
	G.add_edges_from([(key[0],key[1])],weight=value)

nx.draw(G)
plt.show()

aa=source_list+target_list


#Nodes in Path from Target to Source
X=Counter()
for i in Paths:
	bb = [x for x in i if x not in aa]
	X.update(bb)
leftovers=X.keys()
leftovers.sort()
leftovers














source_list = ['research', 'sql', 'c', ('data', 'analysis'), ('research', 'development')]
target_list= ['sales', 'selling', 'sell', 'marketing', 'communications', ('lead', 'generation'), 'support']
FPaths=[]
for i in source_list:
	for n in target_list:
		if nx.has_path(DG,source=i,target=n)==True:
			x=nx.dijkstra_path(DG, source=i,target=n)
			FPaths.append(x)
		else:
			pass
GG = Counter()
for i in FPaths:
	GG.update(i)
FPathList = GG.keys()
Gsp = nx.subgraph(DG,FPathList)
pos = nx.spring_layout(DG,k=0.5,iterations=20)
nx.draw_networkx_nodes(DG,pos,node_size=100,linewidths=0)
nx.draw_networkx_edges(Gsp,pos,width=1,edge_color='b')
nx.draw_networkx_labels(Gsp,pos,font_size=15)
plt.show()


G1=Counter()
for i in FPaths:
	G1.update(i)
AA=G1.keys()

RES = {}
for i in FPaths:
	GG = Counter()
	GG.update(i)
	for key1 in GG.keys():
				for key2 in GG.keys():
					if key1 != key2:
						RES[(key1, key2)] = GG[key2]
GGG=nx.DiGraph()
for key,value in res.items():
	GGG.add_edges_from([(key[0],key[1])],weight=value)

nx.draw(GGG)
plt.show()

AAA=source_list+target_list


#Nodes in Path from Target to Source
XX=Counter()
for i in FPaths:
	bbb = [x for x in i if x not in AAA]
	XX.update(bbb)
leftovers1=XX.keys()
leftovers1.sort()
leftovers1

























import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk import bigrams, trigrams
import math
from collections import Counter
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.tag.brill import SymmetricProximateTokensTemplate, ProximateTokensTemplate
from nltk.tag.brill import ProximateTagsRule, ProximateWordsRule, FastBrillTaggerTrainer
tokenizer = WhitespaceTokenizer()
templates = [
	SymmetricProximateTokensTemplate(ProximateTagsRule, (1,1)),
	SymmetricProximateTokensTemplate(ProximateTagsRule, (2,2)),
	SymmetricProximateTokensTemplate(ProximateTagsRule, (1,2)),
	SymmetricProximateTokensTemplate(ProximateTagsRule, (1,3)),
	SymmetricProximateTokensTemplate(ProximateWordsRule, (1,1)),
	SymmetricProximateTokensTemplate(ProximateWordsRule, (2,2)),
	SymmetricProximateTokensTemplate(ProximateWordsRule, (1,2)),
	SymmetricProximateTokensTemplate(ProximateWordsRule, (1,3)),
	ProximateTokensTemplate(ProximateTagsRule, (-1, -1), (1,1)),
	ProximateTokensTemplate(ProximateWordsRule, (-1, -1), (1,1)),	]

tagged_sentences=[]
tokenizer =WhitespaceTokenizer()
with open("datascience_6.txt","r") as openfile:
	for line in openfile:
		words = line.lower().strip()
		words=re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\>|\?|\/|\;|\:|\"|\'', '',words)
		words=words.split('\r')
		jobposts = [s.lstrip() for s in words]
		for jobpost in jobposts:
			sentences=jobpost.split('.')
			for sentence in sentences:
				tokenized_sentence=tokenizer.tokenize(sentence)
				initial_tagged_sentence=nltk.pos_tag(tokenized_sentence)
				tagged_sentences.append(initial_tagged_sentence)
tagged_no_empties = []
a =[]
for i in tagged_sentences:
	if a==i:
		pass	
	else:
		tagged_no_empties.append(i)
unigram_tagger=nltk.UnigramTagger(tagged_no_empties)
trainer = FastBrillTaggerTrainer(initial_tagger=unigram_tagger,
templates=templates, trace=3,deterministic=True)
brill_tagger = trainer.train(tagged_sentences, max_rules=10)


test_sentences=[]
with open("testset2.txt","r") as openfile:
	for line in openfile:
		words = line.lower().strip()
		words=re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\>|\?|\/|\;|\:|\"|\'', '',words)
		words=words.split('\r')
		jobposts = [s.lstrip() for s in words]
		for jobpost in jobposts:
			sentences=jobpost.split('.')
			for sentence in sentences:
				tokenized_sentence=tokenizer.tokenize(sentence)
				test_initial_tagged_sentence=nltk.pos_tag(tokenized_sentence)
				test_sentences.append(test_initial_tagged_sentence)
test_tagged_no_empties = []
a =[]
for i in test_sentences:
	if a==i:
		pass	
	else:
		test_tagged_no_empties.append(i)

brill_tagger.evaluate(test_tagged_no_empties) 

				
	















tagged_sentences=[]
with open("trainingset.txt","r") as openfile:
	for line in openfile:
		words = line.lower().strip().replace('(',',').replace(')',',')
		words=words.split('\r')
		jobposts = [s.lstrip() for s in words]
		for jobpost in jobposts:
			sentences=jobpost.split('.')
			for sentence in sentences:
				tokenized_sentence=tokenizer.tokenize(sentence)
				initial_tagged_sentence=nltk.pos_tag(tokenized_sentence)
				tagged_sentences.append(initial_tagged_sentence)
tagged_no_empties = []
a =[]
for i in tagged_sentences:
	if a==i:
		pass	
	else:
		tagged_no_empties.append(i)
unigram_tagger=nltk.UnigramTagger(tagged_no_empties)
trainer = FastBrillTaggerTrainer(initial_tagger=unigram_tagger,
templates=templates, trace=3,deterministic=True)
brill_tagger = trainer.train(tagged_sentences, max_rules=10)
















#TechSkillsOnly
with open("techskills2.txt", "r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	for line in openfile:
		frequency = Counter() 
		frequency2 = Counter()
		words1 = line.lower().strip()
		words = re.sub(r'[0-9]','',words1)
		words = pattern.sub('', words)
		words = words.split('\r')
		tokens = [s.lstrip() for s in words]
		tokens1 = [tuple(i.split()) for i in tokens]
		frequency.update(tokens1)
TechSKILLS=frequency.keys()

#Undirected Graph
#Boolean CoOccurrenceCounter 
with open("datascience_6.txt","r") as openfile:
	Stopwords = nltk.corpus.stopwords.words('english')
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	WordGramWordGram= Counter()
	WordGramBiGram = Counter()
	WordGramTriGram = Counter()
	BiGramBiGram = Counter()
	BiGramTriGram = Counter()
	TriGramTriGram = Counter()
	tokenizer = WhitespaceTokenizer()
	for line in openfile:
		words = line.lower().strip().replace('(',',').replace(')',',')
		words=re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\.|\>|\?|\/|\;|\:|\"|\'', '',words)
		words = pattern.sub('', words)
		words=words.split('\r')
		words = [s.lstrip() for s in words]
		ReservoirALL={}
		for word in words:
			CountWordGrams = Counter()
			CountBiGrams = Counter()
			CountTriGrams = Counter()
			
			wordsplit= tokenizer.tokenize(word)
			wordsplit = [s.lstrip() for s in wordsplit]
			NoDupes = list(set(wordsplit))
			TuplesNoDupes=[tuple(i.split()) for i in NoDupes]
			skillsonly=[x for x in TuplesNoDupes if x in TechSKILLS]
			skillsclean = [token for token in skillsonly if token not in Stopwords]
			
			BiGrams=bigrams(wordsplit)
			NoDupesBiGrams = list(set(BiGrams))
			BiGrams=[x for x in NoDupesBiGrams if x in TechSKILLS]
			TriGrams=trigrams(wordsplit)
			NoDupesTriGrams = list(set(TriGrams))
			TriGrams=[x for x in NoDupesTriGrams if x in TechSKILLS]
		
			CountWordGrams.update(skillsclean)
			CountBiGrams.update(BiGrams)
			CountTriGrams.update(TriGrams)

			for key1 in CountWordGrams.keys():
				for key2 in CountWordGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountWordGrams[key2]
			WordGramWordGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			WordGramBiGram.update(ReservoirALL)
			for key1 in CountWordGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			WordGramTriGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountBiGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountBiGrams[key2]
			BiGramBiGram.update(ReservoirALL)
			for key1 in CountBiGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			BiGramTriGram.update(ReservoirALL)
			for key1 in CountTriGrams.keys():
				for key2 in CountTriGrams.keys():
					if key1 != key2:
						ReservoirALL[(key1, key2)] = CountTriGrams[key2]
			TriGramTriGram.update(ReservoirALL)
COOCCURRENCE = dict(WordGramWordGram.items() + WordGramBiGram.items() + WordGramTriGram.items() + BiGramBiGram.items() + BiGramTriGram.items() + TriGramTriGram.items())
















json_data=open('jsondes.json')
data=json.load(json_data)

CourseDescriptions={}
for i in data:
	key1=i['title']
	value1=i['description']
	CourseDescriptions[key1]=[value1]
Descriptions=[]
for i in data:
    Descriptions.append(i['description'])

Boolean class counter
CountWordGrams=Counter()
CountBiGrams=Counter()
CountTriGrams=Counter()
for i in Descriptions:
	wordsplit= tokenizer.tokenize(i)
	wordsplit = [s.lstrip() for s in wordsplit]
	NoDupes = list(set(wordsplit))
	TuplesNoDupes=[tuple(i.split()) for i in NoDupes]
	skillsonly=[x for x in TuplesNoDupes if x in SKILLS]
	skillsclean = [token for token in skillsonly if token not in Stopwords]
	BiGrams=bigrams(wordsplit)
	NoDupesBiGrams = list(set(BiGrams))
	BiGrams=[x for x in NoDupesBiGrams if x in SKILLS]
	TriGrams=trigrams(wordsplit)
	NoDupesTriGrams = list(set(TriGrams))
	TriGrams=[x for x in NoDupesTriGrams if x in SKILLS] 					
	CountWordGrams.update(skillsclean)
	CountBiGrams.update(BiGrams)
	CountTriGrams.update(TriGrams)

Combined = dict(CountWordGrams.items()+CountBiGrams.items()+CountTriGrams.items())



