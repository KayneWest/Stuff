from py2neo import neo4j, cypher, node, rel
import math
import re
import nltk
import datetime
import itertools
from itertools import combinations,chain
from multiprocessing import Pool,cpu_count
from nltk.tokenize import WhitespaceTokenizer
from nltk import bigrams, trigrams
from nltk import PunktSentenceTokenizer
import copy_reg
import types
import signal
import time
from collections import defaultdict
## makes the default pickle method work for just about everything
def reduce_method(m):
	return (getattr, (m.__self__, m.__func__.__name__))
pickle = copy_reg.pickle(types.MethodType, reduce_method)
def log(msg):
	print("{} {}".format(str(datetime.datetime.now()), msg))
## so if something takes awhile, we can raise an error
class Timeout():
	"""Timeout class using ALARM signal."""
	class Timeout(Exception):
		pass

	def __init__(self, sec):
		self.sec = sec

	def __enter__(self):
		signal.signal(signal.SIGALRM, self.raise_timeout)
		signal.alarm(self.sec)

	def __exit__(self, *args):
		signal.alarm(0)    # disable alarm

	def raise_timeout(self, *args):
		raise Timeout.Timeout()


class SkillMatcher():
		argmax = -math.log((1) / (2.0 * 10.0))
		argmin = -math.log((10) / (2.0 * 10.0))
		def __init__(self):
			self.Stopwords = nltk.corpus.stopwords.words('english')
			additions = ['mostly', 'done', 'back', 'ive', 've', 'vs', 'much', 'throw', 'kids', 'kid', 'smart','megathread','serious', 'got', 'get', 'getting', 'use', 'people', 'show', 'peopl', 'youll', 'you ll', ]
			self.Stopwords = self.Stopwords + additions
			self.tokenizer = WhitespaceTokenizer()
			self.tknr = PunktSentenceTokenizer()
			self.graph_db = neo4j.GraphDatabaseService("http://54.85.127.28:8080/db/data")
			self.reall='.*'
			self.doub='\\ '
			self.money='$'
			## commenting out so the cores only initialize as needed, up for debate, but tis what i think is best
			#self.pool = Pool(cpu_count())
			self.pats=[]
			for i in self.Stopwords:
				i1='^'+i+'\ .*'
				i2=self.reall+self.doub+i+self.money
				self.pats.append(i1)
				self.pats.append(i2)
			self.stoppatterns = re.compile('|'.join(self.pats))

		def parallelQuery(self,x):
			forcypher = 'MATCH (n:Skill {name_lower:"'+x+'"}) RETURN n.name'
			return  list(cypher.execute(self.graph_db,forcypher))[0]

		##is generic parallel map, boots up, runs on max cpu, waits til results collect,shutsdown cpu
		def mapz(self, fn, lst):
			pool = Pool(cpu_count())
			try:
				with Timeout(15):
					ans = pool.map(fn,lst)
					return ans
			except Timeout.Timeout:
				print 'timeout'
			finally:
				pool.close()
				pool.join()


		def find_matches(self, text):
			greedyS=re.compile('.*s$')
			# tknr = PunktSentenceTokenizer()  #this tokenizer breaks apart text by separating sentences into lists.
			# tokenizer = WhitespaceTokenizer()  #this tokenizer breaks apart text via finding spaces between words
			text = text.lower()
			#       log('tokenizing text')
			wordsplit = self.tknr.tokenize(text)
			wordsplit2=[i.strip().split(',') for i in wordsplit]
			wordsplit3=[[self.tokenizer.tokenize(x) for x in wordsplit2[i]] for i in range(len(wordsplit2))]

			cleanedwords=[[[re.sub(r'\~|\`|\@|\$|\%|\^|\&|\*|\(|\)|\_|\=|\{|\[|\}|\]|\\|\<|\,|\<|\.|\>|\?|\/|\;|\:|\"|\'', '',q).strip()
			for q in wordsplit3[i][x]] for x in range(len(wordsplit3[i]))] for i in range(len(wordsplit3))]


			#the text has been split into sentences, then tokenized via the whitespace
			BiGrams=[[bigrams(cleanedwords[i][x]) for x in range(len(cleanedwords[i]))] for i in range(len(cleanedwords))]
			TriGrams=[[trigrams(cleanedwords[i][x]) for x in range(len(cleanedwords[i]))] for i in range(len(cleanedwords))]
			clean = []
			for i in range(len(cleanedwords)):
				for x in range(len(cleanedwords[i])):
					for q in cleanedwords[i][x]:
						#elminates whitespaces from text
						if q not in self.Stopwords:
							if q!='':
								q=q.lower().capitalize()
								#finds the greedy s pattern within a word
								#this appends 's' to words without an 's' and
								#suspends 's' from words with an 's' at the end
								#this also keeps words original form to ensure all
								#possibilities are searched for. Same for BiGram TriGram
								if re.findall(greedyS,q)==[q]:
									clean.append(q)
									clean.append(re.sub(r's$','',q))

								else:
									clean.append(q)
									clean.append(q+'s')
			for i in range(len(BiGrams)):
				for x in range(len(BiGrams[i])):
					for q in BiGrams[i][x]:
						if q!='':
							bigram=q[0]+' '+q[1]
							if re.match(self.stoppatterns,bigram)==None:
								if re.findall(greedyS,bigram)==[bigram]:
									bigram=bigram.lower().capitalize()
									clean.append(bigram)
									clean.append(re.sub(r's$','',bigram))
								else:
									bigram=bigram.lower().capitalize()
									clean.append(bigram)
									clean.append(bigram+'s')
			for i in range(len(TriGrams)):
				for x in range(len(TriGrams[i])):
					for q in TriGrams[i][x]:
						if q!='':
							trigram=q[0]+' '+q[1]+' '+q[2]
							if re.match(self.stoppatterns,trigram)==None:
								if re.findall(greedyS,trigram)==[trigram]:
									trigram=trigram.lower().capitalize()
									clean.append(trigram)
									clean.append(re.sub(r's$','',trigram))
								else:
									trigram=trigram.lower().capitalize()
									clean.append(trigram)
									clean.append(trigram+'s')
			clean = list(set(clean))
			clean = [i.lower() for i in clean]
			log('starting cypher queries')
			result = filter(None,self.mapz(self.parallelQuery, clean))
			search = list(set(chain(*list(chain(*result)))))
			log('finished cypher queries')
			search_results = [x.encode('utf-8') for x in search]
			new_set=[]
			for i in range(len(sorted(search_results))):
				try:
					if sorted(search_results)[i]+'s'==sorted(search_results)[i+1]:
						new_set.append(sorted(search_results)[i+1])
					else:
						new_set.append(sorted(search_results)[i])
				except:
					pass
			return list(set(new_set))


		def experimental_leacock_chodrow_sim2(self, word):
			#word1 is the first skill found in a course/job, eg "Markov Models", word2 follows that same format
			#topic is the mother node, for now as it pertains to comparing two "machine learning" courses, the topic would be ML
			word1 = word[0].lower()
			word2 = word[1].lower()

			query_string2 = "MATCH (a:Skill {name_lower:'" + word1 + "'}),(n:Skill {name_lower:'" + word2 + "'}),p=shortestPath((a)-[:has*]-(n)) return length(p) limit 1"
			try:
				shortest_path = list(cypher.execute(self.graph_db, query_string2))
				shortest_path = shortest_path[0][0][0]
				return (word1,-math.log((shortest_path) / (2.0 * 10.0)))
			except:
				pass

		def finding_the_max(self, set1diff, set2diff):
			"""
			I CHANGED THE MATH SLIGHTLY, meaning that it will take the shorter list and do comparisons,
			so that the final score won't be affected by two courses with different sized descriptions and this
			will get rid of all "None"s that exist in the final result

			:type set1diff: set
			:type set2diff: set
                """
			s1, s2 = (set1diff, set2diff) if len(set1diff) > len(set2diff) else (set2diff, set1diff)
			combos=[]
			for element1 in s1:
				for element2 in s2:
					combos.append((element1,element2))
			combos=list(set(combos))
			combos.sort()
			search = filter(None,self.mapz(self.experimental_leacock_chodrow_sim2, combos))
			d= defaultdict( list )
			for k, v in search:
				d[k].append(v)
			d=[{k:d[k]} for k in sorted(d)]
			maximum=[]
			for item in d:
				for v in item.values():
					m=max(v)
					if m == None:
						maximum.append(SkillMatcher.argmin)
					else:
						maximum.append(max(v))
			return maximum








		def CompareSkills(self, desc1, desc2):

			desc1Clean = self.find_matches(desc1)

			desc2Clean = self.find_matches(desc2)

			log("Starting intersection call")

			set1 = set(desc1Clean)
			set2 = set(desc2Clean)

			InList = list(set1.intersection(set2))

			#intersection members
			#       InList = list(set(desc1Clean).intersection(desc2Clean))

			#this is non-interesection members
			set1diff = set1.difference(set2)
			set2diff = set2.difference(set1)
			#       XX = differences(desc1Clean, desc2Clean)


			#this is for the event where set1/set2 has the same skills as the other,
			#but one simply has more. This will happen when the difference of one is 0, but the
			#other has more skills.
			if set1diff==set():
				uni=list(set1.union(set2))
				set1diff.add(uni[0])
			if set2diff==set():
				uni=list(set2.union(set1))
				set2diff.add(uni[0])


			log("Max similarity")
			#max similarity

			result = self.finding_the_max(set1diff, set2diff)


			#figure out a way to find the argMax of the node structure
			#argmax in this case would be -math.log((2 #shortest possible path) / (2.0 * 10#max distance of the root structure))

			InList2 = []
			for i in InList:
				InList2.append(self.argmax)  #argmax for Machine Learning is this

			score = result + InList2
			log("Calculating average similarity")
			#average similarity of their differences
			Final = sum(score) / len(score)

			return (set1diff, set2diff, InList, Final)


import csv
import random
samples=[]
with open('has_math.txt','rU') as openfile:
	reader = csv.reader(openfile, delimiter="\t")
	for line in reader:
		for row in line:
			samples.append(row)
samples = list(combinations(samples, 2))

random_samples=random.sample(set(samples), 600)


if __name__ == "__main__":
	sigmoids=[]
	for i in random_samples:
		try:
			sm = SkillMatcher()
			set1diff, set2diff, common, similarityScore = sm.CompareSkills(i[0], i[1])
			log("Final: {}".format(similarityScore))
			#You put a zero in there again
			#perc = ((0 - sm.argmin)/(sm.argmax - sm.argmin))
			perc = ((similarityScore - sm.argmin)/(sm.argmax - sm.argmin))
			log("perc: {}".format(perc))
			sigmoids.append(perc)
		except:
			log('there was an error')
