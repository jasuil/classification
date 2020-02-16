from gensim import corpora
from gensim import models

documents = ['나는 나다', '오늘은 눈이 많이 왔다.', '내일은 월요일이다.',
             '월요일에는 일찍 일어나야지', '나는 지금 뭘 하고 있나?']

stoplist = ('.!?')
texts = [[word for word in document.split() if word not in stoplist]
         for document in documents]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
print('corpus : {}'.format(corpus))

#모든 내용에 대하여 2개의 주제로 분류한다.
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                               num_topics=2, random_state=1)


for t in range(lda.num_topics):

  #빈도수가 높은 상위 2개의 단어만
  print(lda.show_topic(t, 2))



