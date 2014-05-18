
from bs4 import BeautifulSoup
from urllib2 import urlopen

url="http://www.uic.edu/ucat/courses/"
soup = BeautifulSoup(urlopen(url).read())
links = soup.findAll("a")
pattern=re.compile('".*"')
actual_links=[]
for link in links:
	link=re.findall(pattern,str(link))
	link=url+link[0].replace('"','')
	actual_links.append(link)
actual_links=actual_links[2:]
import time
pattern2=re.compile('<b>.*\n.*\n\n')

course_full={}
for i in actual_links:
	url=i
	log('sleepytime')
	time.sleep(10)
	log('now going into '+url)
	soup = BeautifulSoup(urlopen(url).read())
	courses=re.findall(pattern2,str(soup))
	for course in courses:
		course=re.sub(r'</b><br /><b>.*.</b>|<i>Prerequisite.*','',course).replace('<b>','').replace('</b>','').split('\n')
		course_full[course[0]]=course[1]
for k,v in dicitonary
	
	
	time.sleep(2)


while True:
    print "This prints once a minute."
    time.sleep(60) 
	
	
pattern3="
