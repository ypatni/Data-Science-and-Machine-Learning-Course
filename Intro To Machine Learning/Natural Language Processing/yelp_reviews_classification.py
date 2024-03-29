import nltk 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

yelp = pd.read_csv('yelp.csv')
print(yelp.head())
#print(yelp.describe())
yelp['text length'] = yelp['text'].apply(len)
print(yelp.head())

graph = sns.FacetGrid(yelp, col = 'stars')
graph.map(plt.hist, 'text length')
plt.show()
sns.boxplot(x = 'stars', y = 'text length', data = yelp, palette = 'plasma')
plt.show()
sns.countplot(x = 'stars',data=yelp, palette='rainbow')
plt.show()
stars_mean = yelp.groupby('stars').mean()
print(stars_mean)

sns.heatmap(stars_mean.corr(), cmap= 'coolwarm', annot= True)
plt.show()

#NLP Classification
yelp_class = yelp[(yelp['stars']==1) | (yelp['stars'] ==5)]
print(yelp_class)

x = yelp_class['text']
y = yelp_class['stars']
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(x_train, y_train)
predictions = nb.predict(x_test)
print(classification_report(y_test, predictions)) 
print(confusion_matrix(y_test, predictions)) 




#including tfidf and using pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfidf', TfidfTransformer()), #integers to weighted tfidf scores 
    ('model', MultinomialNB())
])

x = yelp_class['text']
y = yelp_class['stars']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
print(classification_report(y_test, predictions)) 
print(confusion_matrix(y_test, predictions)) 
#tfidf made it worse!