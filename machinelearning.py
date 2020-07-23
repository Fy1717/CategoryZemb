#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 7 22:11:39 2019

@author: nur
"""

import jpype

jpype.startJVM("/home/nur/Masaüstü/jdk-8u231-linux-x64/jdk1.8.0_231/jre/lib/amd64/server/libjvm.so","-Djava.class.path=/home/nur/Masaüstü/dizin/zemberek-tum-2.0.jar","-ea")

Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")

tr = Tr()

Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")

zemberek = Zemberek(tr)

import pandas as pd
import numpy as np
data = pd.read_csv(r"dataset.csv", encoding="UTF-8")
data.Kategori = [1 if each == "Spor" else 2 if each == "Teknoloji" else 3 if each == "Ekonomi" else 0 for each in data.Kategori]

Kategori = data.Kategori
İçerik = data.İçerik
data.İçerik = np.array(data.İçerik)
print(data.İçerik)
#df_docs = pd.DataFrame({'Sinif': Kategori,'Dokuman': İçerik})
#df_docs = df_docs[['Sinif', 'Dokuman']]
#print(df_docs)


import nltk
WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')
stop_word = ['a', 'ancak', 'artık', 'ama', 'asla', 'aynı', 'b','bazı', 'bana', 'bazen', 'bazıları', 'bazısı', 'ben', 
             'beni', 'benim', 'beş', 'bile', 'bin', 'bir', 'birçoğu', 'birçok', 'birçokları', 'biri', 'birisi', 'birkaçı',
             'birşey', 'birşeyi', 'biz', 'bize', 'bizi', 'bizim', 'böyle', 'böylece', 'bu', 'buna', 'bunda', 'bundan', 
             'bunu', 'bunun', 'burada', 'bütün', 'çoğu', 'çoğuna', 'çoğunu', 'd', 'değil', 'demek', 'diğer', 'diğeri',
             'diğerleri', 'diye', 'dolayı', 'elbette', 'fakat', 'falan', 'felan', 'filan', 'gene', 'geri', 'göre', 
             ' hangi', 'hangisi', 'hani', 'hatta', 'henüz', 'hepsine', 'hepsini', 'her biri', 'herkes', 'herkese', 
             'herkesi', 'hiç kimse', 'hiçbiri', 'hiçbirine', 'hiçbirini', 'i', 'ı', 'ilk', 'içinde', 'işte', 'iken', 
             'ila', 'ileri', 'iyi', 'kaç', 'kadar', 'kendi', 'kendine', 'kendini', 'kime', 'kimi', 'kimin', 'kimisi',
             'ler', 'lar', 'madem', 'mi', 'ne kadar', 'ne zaman', 'nedir', 'nereden', 'nesi', 'neyse', 'ö', 'ona', 
             'ondan', 'onlar', 'onlara', 'onlardan', 'onların', 'onu', 'onun', 'orada', 'oysa', 'oysaki', 'öbürü', 
             'ön', 'önce', 'ötürü', 'öyle', 'peki', 'sana', 'sen', 'senden', 'seni', 'senin', 'sizden', 'size', 'sizi',
             'sizin', 'son', 'sonra', 'şayet', 'şimdi', 'şöyle', 'şuna', 'şunda', 'şundan', 'şunlar', 'şunu', 'şunun', 
             'tabi', 'tamam', 'tümü', 'u', 'ü', 'üzere', 'var', 'vb', 'veyahut', 'ya da', 'yerine', 'yine', 'yoksa', 
             'zaten', 'zira' ]
stop_word_list.extend(stop_word)
print(stop_word_list)

import re
import nltk as nlp
nltk.download('wordnet')
İçerik_list = []
for İçerik in data.İçerik:
    İçerik = re.sub("[^abcçdefgğhıiklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ]"," ",İçerik)
    İçerik = İçerik.lower()
    İçerik =nltk.word_tokenize(İçerik)
    İçerik = [kelime for kelime in İçerik if not kelime in set(stop_word_list)]
    lemma = nlp.WordNetLemmatizer()
    İçerik = [lemma.lemmatize(kelime) for kelime in İçerik]
    for kelime in İçerik:
        if kelime.strip()>'':
            yanit = zemberek.kelimeCozumle(kelime)
            if yanit:
                print("{}".format(yanit[0]))
                İçerik_list.append(zemberek.kelimeCozumle(kelime)[0].kok().icerik())
            else:
                print("{} ÇÖZÜMLENEMEDİ".format(kelime))
    İçerik = " ".join(İçerik_list)
print(İçerik)


jpype.shutdownJVM()



from sklearn.feature_extraction.text import CountVectorizer 
max_features = 500

count_vectorizer = CountVectorizer(max_features = max_features)
sparce_matrix = count_vectorizer.fit_transform(İçerik_list).toarray() #x

print("en sık kullanilan {} keilmeler : {} ".format(max_features,count_vectorizer.get_feature_names()))

import numpy as np

y = data.iloc[:,0].values
#x = sparce_matrix[:,0]
x = sparce_matrix

from sklearn.model_selection import train_test_split

#x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)

x_train,x_test = train_test_split(x, test_size = 0.2)
y_train, y_test = train_test_split(y, test_size = 0.2)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

x_pred = nb.predict(y_test)
print("accuracy : ", nb.score(x_pred.reshape(1,-1),y_test))















































