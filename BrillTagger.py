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


