from py2neo import neo4j,cypher,node,rel
import math
import py2neo
import re

graph_db = neo4j.ServiceRoot("http://54.85.127.28:8080").graph_db


session = cypher.Session("http://54.85.127.28:8080")
tx = session.create_transaction()


tx.append("start center=node(1) MATCH p=(center)-[:has*]->last where  not(last-->()) RETURN length(p) order by length(p) Desc limit 1;")



def leacock_chodrow_sim(word1,word2,topic):
	#word1 is the first skill found in a course/job, eg "Markov Models", word2 follows that same format
	#topic is the mother node, for now as it pertains to comparing two "machine learning" courses, the topic would be ML  
        if word1==word2:
		print "same word, mothafucka"
	query_string1 = "MATCH p=(center)-[:has*]->last where  not(last-->()) and center.name='" + topic + "' RETURN length(p) order by length(p) Desc limit 1"
	max_depth=list(cypher.execute(graph_db,query_string1))
	max_depth=max_depth[0][0][0]
	query_string2 =  "MATCH (a),(b), p = shortestPath((a)-[*]-(b)) where a.name='" + word1 + "' and b.name='" + word2 + "' RETURN length(p)"
	shortest_path=list(cypher.execute(graph_db, query_string2))
	shortest_path = shortest_path[0][0][0]
	if shortest_path is None or shortest_path < 0 or max_depth == 0:
		return None
        return -math.log((shortest_path) / (2.0 * max_depth))


result = neo4j.CypherQuery(graph_db, query_string).execute()



def wu_palmer_sim(word1,word2,topic):
	#Either need to get indexes, then proceed
	query_string1="MATCH p =(a)<-[:has*]-common_ancestor-[:has*]->(b) where a.name='Category:"+word1+"' and b.name='Category:"+word2+"' RETURN common_ancestor.name order by length(p)limit 1"
	common_ancestor=list(cypher.execute(graph_db,query_string1))
	common_ancestor=common_ancestor[0][0][0].encode('ascii','ignore')
	query_string2="MATCH path=(c)<-[:has*]-(t) WHERE t.name='"+topic+"' AND c.name='"+common_ancestor+"' RETURN length(path)"
	depth_of_common=list(cypher.execute(graph_db,query_string2))
	depth_of_common=depth_of_common[0][0][0]
	query_string3="MATCH path=(c)<-[:has*]-(t) WHERE c.name='Category:"+word1+"' AND t.name='"+topic+"' RETURN length(path)"
	depth_of_word1=list(cypher.execute(graph_db,query_string3))
	depth_of_word1=depth_of_word1[0][0][0]
	query_string4="MATCH path=(c)<-[:has*]-(t) WHERE c.name='Category:"+word2+"' AND t.name='"+topic+"' RETURN length(path)"
	depth_of_word2=list(cypher.execute(graph_db,query_string4))
	depth_of_word2= depth_of_word2[0][0][0]
	return depth_of_common/(depth_of_word1+depth_of_word2)

start a=node(353),a=node(355) MATCH p =(a)<-[:has*]-common_ancestor-[:has*]->(b) RETURN p order by length(p)limit 1;

start a=node(353) MATCH p =(a)<-[:has*]-common_ancestor-[:has*]->(b) where b.name='Casey Scheuerell' RETURN p order by length(p)limit 1;





#need the text version of a Course.


nodeslist=list(cypher.execute(graph_db,"START center=node:Category(topic) MATCH p=center-[:has]->last WHERE NOT (last-->()) RETURN nodes(p)")) 
#result in this format [[[],[],[],<tag>]
nodelist2=[]
for subitem in range(len(nodelist[0])):
	for skill in nodelist[0][subitem]:
		skill=skill.encode('ascii', 'ignore').strip()
		nodelist2.append(skill)
keywords=list(set(nodelist2))
escaped=[re.escape(i) for i in keywords]
pattern = re.compile(r'\b(?:%s)\b' % '|'.join(escaped))
patternss = [re.compile(re.escape(keyword),re.IGNORECASE) for keyword in keywords]
patternsss = [re.compile((keyword),re.IGNORECASE) for keyword in keywords]
Course1=[]
for singlepattern in patternss:
	for match in re.findall(singlepattern, Stanford):
		match=match.lower()
		Course1.append(match)
Course1Clean=list(set(StanfordList))
Course2=[]
for singlepattern in patternss:
	for match in re.findall(singlepattern, Washington):
		match=match.lower()
		Course2.append(match)
Course2Clean=list(set(WashingtonList))
InList=[]
for i in Course2Clean:
	if i in Course1Clean:
		InList.append(i)
Course2Out=[]
for i in Course2Clean:
	if i not in Course1Clean:
		Course2Out.append(i)
Course1Out=[]
for i in Course1Clean:
	if i not in Course2Clean:
		Course1Out.append(i)
InList2=[]
for i in Course2Clean:
	if i in Course1Clean:
		InList2.append(1)


avg=[]
for i in Course2Out:
	for x in Course1Out:
		answer=leacock_chodrow_sim(i,x,topic)
		avg.append(answer)

score=avg+InList2
Final=sum(score)/len(score)
