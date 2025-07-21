import nltk
import numpy as np
import re
import heapq
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

text = """
Gizemli Mavi Küreler İngiltere'nin Kırsalında Ortaya Çıktı
Norfolk'taki küçük bir köyde yaşayanlar, dün sabah uyandıklarında tarlalarına dağılmış onlarca parlayan mavi küreyle karşılaştı. Futbol topu büyüklüğündeki her bir küre, hafif bir vızıltı sesi çıkarıyor ve ışıkla nabız atar gibi titreşiyordu. Köylüler başlangıçta bunun bir sanat enstalasyonu ya da detaylı bir şaka olduğunu düşündü. Ancak, kimse sorumluluğu üstlenmedi.

Cisimler ilk olarak çiftçi Gerald Miles tarafından sabahki kontrol turu sırasında fark edildi. “Neredeyse birinin üstüne basıyordum,” dedi. “Dokunduğumda sıcaktı ama yakıcı değildi.” Yakındaki bir üniversiteden bilim insanları araştırma yapmak üzere çağrıldı. İlk testler, kürelerin bilinmeyen bir bileşik malzemeden yapıldığını öne sürüyor.

Fizikçi Dr. Helen Porter, enerji ölçümlerinin daha önce gördüğü hiçbir şeye benzemediğini belirtti. Öğle saatlerine doğru, yetkililer tarafından tarlalar kordona alındı. Köy kısa sürede medyanın ilgi odağı haline geldi. Turistler ve komplo teorisyenleri bölgeye akın etmeye başladı.

Bazı köylüler cisimlerin dünya dışı kökenli olduğuna inanıyor. Diğerleri ise bunun kontrolden çıkmış bir hükümet deneyi olduğunu düşünüyor. Bu sırada küreler yerlerinden kıpırdamadan parlamaya devam ediyor. Yaban hayatı onların varlığından etkilenmiş gibi görünmüyor. Hatta bir tilkinin, kürelerin hemen yanında uyuduğu görüldü.

Yerel okullar tedbir amacıyla bir günlüğüne kapatıldı. Belediye başkanı halkı sakin olmaya ve kürelere dokunmamaya çağırdı. “Herkesin güvenliği için uzmanlarla yakın çalışıyoruz,” dedi. Akşam karanlığı çökünce, küreler daha da parlak bir şekilde ışıldamaya başladı ve kırsal alanı gizemli bir mavi ışığa boğdu.
"""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9.,!? ]', '', text)
    return text

cleaned_text = clean_text(text)

from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(cleaned_text,language="turkish")

vectorizer = TfidfVectorizer(stop_words=stopwords.words("turkish"))
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

#fast api haline getir, paremetrik yap. parametreler 1 : text , 2:sentence number (bizde 3 olan). default 3 olsun
#uygulamanın başarısını ölç : ROUGE analizi