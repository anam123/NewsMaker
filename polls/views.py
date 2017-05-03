# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import re
from django.shortcuts import get_object_or_404, render

# Create your views here.
from django.http import HttpResponse
from .models import Question
import nltk.data
from nltk.corpus import stopwords
from py_ms_cognitive import PyMsCognitiveImageSearch
from nltk.tag import pos_tag
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import unicodedata
import random
import operator
import string

categories = ['comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.mac.hardware',
'comp.windows.x',
'rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey', 
'sci.electronics',
'sci.med',
'sci.space',
'misc.forsale',
'talk.politics.misc',
'talk.politics.guns',
'talk.politics.mideast',
]

class SummaryTool(object):

    # Naive method for splitting a text into sentences
    def split_cont_to_sentences(self, cont):
        cont = cont.replace("\n", ". ")
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        data = cont
        return tokenizer.tokenize(data)

    # Naive method for splitting a text into paragraphs
    def split_cont_to_paragraphs(self, cont):
        return cont.split("\n")

    # Caculate the intersection between 2 sentences
    def sentences_intersection(self, sent1, sent2):

        # split the sentence into words/tokens
        s1 = set(sent1.split(" "))
        s2 = set(sent2.split(" "))

        # If there is not intersection, just return 0
        if (len(s1) + len(s2)) == 0:
            return 0

        # We normalize the result by the average number of words
        return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2)

    # Format a sentence - remove all non-alphbetic chars from the sentence
    # We'll use the formatted sentence as a key in our sentences dictionary
    def format_sentence(self, sentence):
        sentence = re.sub(r'\W+', '', sentence)
        return sentence

    # Convert the cont into a dictionary <K, V>
    # k = The formatted sentence
    # V = The rank of the sentence
    def get_senteces_ranks(self, cont):

        # Split the cont into sentences
        sentences = self.split_cont_to_sentences(cont)

        # Calculate the intersection of every two sentences
        n = len(sentences)
        values = [[0 for x in xrange(n)] for x in xrange(n)]
        for i in range(0, n):
            for j in range(0, n):
                values[i][j] = self.sentences_intersection(sentences[i], sentences[j])

        # Build the sentences dictionary
        # The score of a sentences is the sum of all its intersection
        sentences_dic = {}
        for i in range(0, n):
            score = 0
            for j in range(0, n):
                if i == j:
                    continue
                score += values[i][j]
            sentences_dic[self.format_sentence(sentences[i])] = score
        return sentences_dic

    # Return the best sentence in a paragraph
    def get_best_sentence(self, paragraph, sentences_dic):

        # Split the paragraph into sentences
        sentences = self.split_cont_to_sentences(paragraph)

        # Ignore short paragraphs
       

        # Get the best sentence according to the sentences dictionary
        best_sentence = ""
        max_value = 0
        for s in sentences:
            strip_s = self.format_sentence(s)
            if strip_s:
                if sentences_dic[strip_s] > max_value:
                    max_value = sentences_dic[strip_s]
                    best_sentence = s

        return best_sentence

    # Build the summary
    def get_summary(self, cont, sentences_dic):

        # Split the cont into paragraphs
        paragraphs = self.split_cont_to_paragraphs(cont)

        # Add the title
        summary = []
        
       # print len(paragraphs)
        # Add the best sentence from each paragraph
        for p in paragraphs:
            #print p+"1"
            sentence = self.get_best_sentence(p, sentences_dic).strip()
            if sentence:
                summary.append(sentence)

        return ("\n").join(summary)

def summarize(ques):
    st = SummaryTool()
    title=ques.news_Article_Heading
    text=  ques.content 
    #print text
    sentences_dic = st.get_senteces_ranks(text)
    summary = st.get_summary( text, sentences_dic)
    #print summary
    return summary

def keyword_finder(ques):
    text=  ques.content
    stop = set(stopwords.words('english'))
    list=[i for i in text.split() if i not in stop]
    str=' '.join(list)
    txt=str

    str = str.replace(".","")
    str = str.replace(",","")
    str = str.replace(";","")
    str = str.replace(":","")
    str = str.replace("!","")
    str = str.replace("?","")
    str = str.replace("'s","")
    str = str.replace("\u2019s","")
    str= unicodedata.normalize('NFKD', str).encode('ascii','ignore')
    tagged_sent = pos_tag(str.split())
    propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
    #print propernouns
    proper_dict={}
    for i in propernouns:
        if i in proper_dict:
            proper_dict[i]=proper_dict[i]+1
            
        else:
            proper_dict[i]=1
    print proper_dict
    
    sorted_x = sorted(proper_dict.items(), key=operator.itemgetter(1))
  
    sorted_x.reverse()
    print sorted_x
    count=0
    search=""
    for i in sorted_x:
        count=count+1
        if count>5:
            break
        else:
            if i[1]>=2:
                search=search+" "+ i[0]
    print search
    search_service = PyMsCognitiveImageSearch('a99a009ec03c47b49389194014f65663', search)
    first_fifty_result = search_service.search(limit=20, format='json')

    url=first_fifty_result[0].__dict__['content_url']
    return url

def categorize(ques):
    docs_new=[]
    docs_new.append(ques.content)
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    X_train_counts.shape
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))
    return twenty_train.target_names[category]

def index(request):
    latest_question_list = Question.objects.order_by('-created')[:50]

    context = {'latest_question_list': latest_question_list}
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    ab=summarize(question)
    ab1=keyword_finder(question)
    ab2=categorize(question)
    print ab1
    print ab2
    return render(request, 'polls/detail.html', {'question': question, 'ab':ab,'url':ab1,'category':ab2 })


def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)

def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)

