import string
from collections import Counter

sentence = input("Lütfen bir cümle girin: ")

words = [
    word.strip(string.punctuation).lower()
    for word in sentence.split()
]


###########

import string
from collections import Counter

sentence = input("Lütfen bir cümle girin: ")
words = [
    word.strip(string.punctuation).lower()
    for word in sentence.split()
]

word_counts = Counter(words)
 
sorted_words = sorted(
    word_counts.items(),
    key=lambda x: (-x[1], x[0])
)
print("\nKelimeler ve Tekrar Sayıları:")
for word, count in sorted_words:
    print(f"[count]{word}: {count}")

###########

def count(elements):
	if elements in dictionary:
		dictionary[elements] += 1
	else:
		dictionary.update({elements: 1})

Sentence = "bu bir test test bu bir test"
dictionary = {}
lst = Sentence.split()
for elements in lst:
	count(elements)

for allKeys in dictionary:
	print ("Frekns", allKeys, end = " ")
	print (":", end = " ")
	print (dictionary[allKeys], end = " ")
	print()

###########


from collections import Counter

def kelime_frekansi(cumle):
    kelimeler = cumle.lower().split()  # Cümleyi küçük harfe çevir ve kelimelere ayır
    frekans = Counter(kelimeler)  # Kelime frekanslarını hesapla
    
    # Frekansa göre azalan sırada sırala
    sirali_frekans = sorted(frekans.items(), key=lambda x: x[1], reverse=True)
    
    for kelime, adet in sirali_frekans:
        print(f"{kelime}: {adet}")

# Kullanıcıdan cümle al
cumle = input("Bir cümle girin: ")
kelime_frekansi(cumle)


###########


from collections import Counter
import re

def kelime_frekansi(cumle):
    cumle = re.sub(r'[^\w\s]', '', cumle)
    kelimeler = cumle.lower().split()  
    frekans = Counter(kelimeler)
    
    sirali_frekans = sorted(frekans.items(), key=lambda x: x[1], reverse=True)
    
    for kelime, adet in sirali_frekans:
        print(f"{kelime}: {adet}")

cumle = input("Bir cümle girin: ")
kelime_frekansi(cumle)

###########


def find_anagrams(words):
    anagram_dict = {} 
    for word in words:
        sorted_word = ''.join(sorted(word))
        if sorted_word in anagram_dict:
            anagram_dict[sorted_word].append(word)
        else:
            anagram_dict[sorted_word] = [word]

    longest_anagram = []
    for group in anagram_dict.values():
        if len(group) > 1 and len(group[0]) > len(longest_anagram[0]) if longest_anagram else True:
            longest_anagram = group
    
    return longest_anagram

words = input("Boşluk ile ayrıdığınız kelimeleri giriniz: ").split()
anagram = find_anagrams(words)

if anagram:
    print("En uzun anagram ->> ")
    print(", ".join(anagram))
else:
    print("Anagram bulunamadı. :( ")


words = input("Boşluk ile ayrıdığınız kelimeleri giriniz: ").split()
anagram = find_anagrams(words)

if anagram:
    print("En uzun anagram ->> ")
    print(", ".join(anagram))
else:
    print("Anagram bulunamadı. :( ")




words = input("Boşluk ile ayrıdığınız kelimeleri giriniz: ").split()
anagram = find_anagrams(words)

if anagram:
    print("En uzun anagram ->> ")
    print(", ".join(anagram))
else:
    print("Anagram bulunamadı. :( ")

