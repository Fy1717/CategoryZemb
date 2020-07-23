import jpype
import pandas as pd
import numpy as np
import nltk
import re
import nltk as nlp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords

nltk.download('punkt')

jpype.startJVM(jpype.getDefaultJVMPath(),"-Djava.class.path=/home/x/Desktop/zemberek-tum-2.0.jar","-ea")
Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
tr = Tr()

Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
zemberek = Zemberek(tr)

data = pd.read_csv(r"dataset.csv", encoding="UTF-8")

data.Kategori = [1 if each == "Spor" else 2 if each == "Teknoloji" else 3 if each == "Ekonomi" else 0 for each in data.Kategori]
Kategori = data.Kategori

İçerik = data.İçerik
data.İçerik = np.array(data.İçerik)

df_docs = pd.DataFrame({'Sinif': Kategori,'Dokuman': İçerik})
df_docs = df_docs[['Sinif', 'Dokuman']]

nltk.download('stopwords')
stopWords = set(stopwords.words('turkish'))

WPT = nltk.WordPunctTokenizer()

stop_word_list = nltk.corpus.stopwords.words('turkish')
stop_word = ['abd', 'ancak', 'artık', 'ama', 'asla', 'aynı', 'b', 'bazı', 'bana', 'bazen', 'bazıları', 'bazısı', 'ben',
             'beni', 'benim', 'beş', 'bile', 'bin', 'bir', 'birçoğu', 'birçok', 'birçokları', 'biri', 'birisi',
             'birkaçı',
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
             'zaten', 'zira']

stop_word_list.extend(stop_word)

Document_list = pd.DataFrame(columns=['Kategori', 'İçerik'], index=range(20))
i=0
for İçerik in data.İçerik:
    İçerik = re.sub("[^abcçdefgğhıiklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ]", " ", İçerik)
    """İçerik = İçerik.lower()"""
    İçerik = nltk.word_tokenize(İçerik)
    İçerik = [kelime for kelime in İçerik if not kelime in set(stop_word_list)]
    lemma = nlp.WordNetLemmatizer()
    İçerik = [lemma.lemmatize(kelime) for kelime in İçerik]
    
    bos = []
    
    for kelime in İçerik:

        if kelime.strip() > '':
            yanit = zemberek.kelimeCozumle(kelime)
            
            if yanit:
                bos.append(zemberek.kelimeCozumle(kelime)[0].kok().icerik())            
    Document_list.Kategori[i] = data.Kategori[i]
    Document_list.İçerik[i] = bos
    """Document_list.İçerik[i] = " ".join(Document_list.İçerik[i])"""

jpype.shutdownJVM()

max_features = 1000
count_vectorizer = CountVectorizer(max_features)
matrix = count_vectorizer.fit_transform(Document_list.İçerik).toarray()

x = matrix
y = data.iloc[:, 0:1].values

x_train, x_test = train_test_split(x, test_size=0.2)
y_train, y_test = train_test_split(y, test_size=0.2)

nb = GaussianNB()


nb.fit(x_train, y_train)

x_pred = nb.predict(y_test)

def tahminOutput(index):    
    if index == 0:
        kategori = 'SPOR'
    elif index == 1:
        kategori = 'TEKNOLOJİ'
    elif index == 2:
        kategori = 'EKONOMİ'
    else:
        kategori = 'EĞİTİM'

    return kategori



tahmin0 = tahminOutput(x_pred[0])
tahmin1 = tahminOutput(x_pred[1])
tahmin2 = tahminOutput(x_pred[2])
tahmin3 = tahminOutput(x_pred[3])


print('\n\n\n\n*************************************')
print('*************************************')


print('1. YAZI TAHMİNİ : ', tahmin0)
print('--------------------------------------')

print('2. YAZI TAHMİNİ : ', tahmin1)
print('--------------------------------------')

print('3. YAZI TAHMİNİ : ', tahmin2)
print('--------------------------------------')

print('4. YAZI TAHMİNİ : ', tahmin3)

print('*************************************')
print('*************************************')

