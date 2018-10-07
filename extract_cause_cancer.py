import os
import jellyfish
import re
import stop_words
import wordsegment
from nltk.tokenize import TweetTokenizer
import emoji
import difflib
from fuzzywuzzy import fuzz
import spacy
from spacy.symbols import *
from nltk import Tree
from nltk import *
import wordsegment
ps_stemmer= stem.porter.PorterStemmer()

nlp= spacy.load('en')

in_dir='/home/ritam/Desktop/SUMMER PROJECTS/Cancer Detection/DATA/OUT_TEXT/'

try:
	# Wide UCS-4 build
	myre = re.compile(u'['
		u'\U0001F300-\U0001F64F'
		u'\U0001F680-\U0001F6FF'
		u'\u2600-\u26FF\u2700-\u27BF]+', 
		re.UNICODE)
except re.error:
	# Narrow UCS-2 build
	myre = re.compile(u'('
		u'\ud83c[\udf00-\udfff]|'
		u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
		u'[\u2600-\u26FF\u2700-\u27BF])+', 
		re.UNICODE)		

remove_puncts="[\{\};,.[!@#$%^&*()_+=?/\'\"\]]"
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
remoji = re.compile('|'.join(re.escape(p) for p in emojis_list))
web_url="http[s]?:[a-zA-Z._0-9/]+[a-zA-Z0-9]"
replacables="RT\s|-\s|\s-|#|@|[|}|]|{|(|)"
prop_name="([A-Z][a-z]+)"
num="([0-9]+)"
name="([A-Za-z]+)"
and_rate="([&][a][m][p][;])"
ellipses="([A-Za-z0-9]+[…])"
hashtags_2="([#][a-zA-z0-9]+[\s\n])"
letters_only= "[^a-z0-9 ]"

tweet_texts=[]
infile=open(in_dir+'tweet_unique.csv')

tknzr=TweetTokenizer(strip_handles=True,reduce_len=True)

def preprocess_tweet(text):
	text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '_URL_', text)
	text = re.sub('http://', '_URL_', text)
	text = re.sub('https://', '_URL_', text)
	#print urls
	text = re.sub('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',text)
	text = re.sub('@', '',text)
	text=re.sub(and_rate,'and',text)
	text=re.sub(replacables,'',text)
	text= re.sub(remoji,'',text)
	try:
		hashtag_list=[i for i in text.split() if i.startswith("#")]
	except:
		hashtag_list=[]	

	hashtag_list=list(set(hashtag_list))
	for elem in hashtag_list:
		segmented_val=wordsegment.segment(elem[1:])
		text.replace(elem,segmented_val)

	return text


def tok_format(tok):
	#return "_".join([tok.orth_, tok.dep_,tok.ent_type_,tok.pos_])
	return "|".join([tok.orth_, tok.dep_])

def to_nltk_tree(node):
	if node.n_lefts + node.n_rights > 0:
		return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
	else:
		return tok_format(node)    

cause_list=['cause']
cause_list=[ps_stemmer.stem(i) for i in cause_list]
allowable_postags=['NOUN']        

def x_causes_cancer(doc):
	for sent in doc.sents:
		nsubj_list=[]
		for word in sent:
			cancer_flag=False
			
			if ps_stemmer.stem(word.orth_.lower()) in cause_list and word.pos_=='VERB':
				for child in word.children:
					if child.orth_.lower()=='cancer':
						cancer_flag=True
				
				if cancer_flag== True:
					for child in word.children:                        
						if child.dep_.lower()=='nsubj'and child.pos_ in allowable_postags:
							nsubj_list.append(str(child.orth_.lower())+'_'+str(child.tag_))
					
		for elem in nsubj_list:
			if elem not in final_nsubj_causes:
				final_nsubj_causes[elem]=0
			final_nsubj_causes[elem]+=1
		
		
cause_file=open(in_dir+'tweet_unique.csv')
print('Loading Done')
c=0
final_nsubj_causes={}

for line in cause_file:
	line=line.strip().split('\t')
	if 'cause' in line[1]:
		print(line[0], line[1])
	continue
	c+=1
	if c%1000==0:
		print(c)
	tweet_text=line.strip()
	doc= nlp(preprocess_tweet(tweet_text))
	x_causes_cancer(doc)
	