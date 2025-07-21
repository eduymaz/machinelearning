import nltk
import numpy as np
import re
import heapq
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

text = """
Mysterious Blue Spheres Appear in Rural England

Residents of a small village in Norfolk woke up yesterday to find dozens of glowing blue spheres scattered across their fields. The spheres, each about the size of a football, emitted a soft humming noise and pulsed with light. Locals initially thought it was an art installation or an elaborate prank. However, no one has come forward to claim responsibility.

The objects were first discovered by farmer Gerald Miles during his morning rounds. “I nearly tripped over one of them,” he said. “It was warm to the touch, but not hot.” Scientists from a nearby university were called in to investigate. Preliminary tests suggest the spheres are made of an unknown composite material.

Dr. Helen Porter, a physicist, stated that the energy readings were unlike anything she had ever seen. By midday, the fields were cordoned off by authorities. The village quickly became the center of media attention. Tourists and conspiracy theorists began arriving in large numbers.

Some villagers believe the objects are of extraterrestrial origin. Others are convinced it’s part of a government experiment gone wrong. Meanwhile, the spheres remain stationary and continue to glow. Wildlife seems unaffected by their presence. One fox was even seen napping next to a cluster of them.

Local schools were closed for the day as a precaution. The mayor urged residents to remain calm and avoid touching the spheres. “We’re working closely with experts to ensure everyone’s safety,” she said. As night fell, the spheres glowed even brighter, casting an eerie blue light over the countryside.
"""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    return text

cleaned_text = clean_text(text)

from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(cleaned_text)

vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(sentences)

sentence_scores = {}

for i in range(len(sentences)):
    score = X[i].toarray().sum()
    sentence_scores[sentences[i]] = score

#en yüksek skora sahip üç cümleyi seç
summary_sentences = heapq.nlargest(3,sentence_scores, key = sentence_scores.get)

summary = ' '.join(summary_sentences)

print("SUMMARY of your text")
print(summary)