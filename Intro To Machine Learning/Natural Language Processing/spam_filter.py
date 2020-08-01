import nltk 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

#nltk.download_shell()

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))
print(messages[588])
#looking at how the messages are listed
for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')

#now we need to figure out messages that are spam or ham 
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names = ['label', 'message'])
# print(messages.head()) #now we have a table 
print(messages.groupby('label').describe())
print('\n')
messages['length'] = messages['message'].apply(len)
print(messages.head())
messages['length'].plot.hist(bins=90)
plt.show()#theres a really really long message that I want to find 
#print(messages[messages['length'] ==910]['message'].iloc[0])#very creepy love letter don't look at it 
messages.hist(column = 'length', by = 'label', bins = 60, figsize=(12,4))
plt.show() #ham texts seem to be centered around 50 words while spam usually has around 150 words

#TEXT PREPROCESSING
#we need numercial feature vectors to use classification tasks

#function that splits message into individual words and returns a list and remove common words like the, a, if

#mess = "Sample Message. It has: punctuations!"
#nopunct = [c for c in mess if c not in string.punctuation]
#print(nopunct)
#print(stopwords.words('English'))  - produces all the common words that probably don't affect the spam or ham detection
#nopunct = ''.join(nopunct) #joining elements in the list together
#print(nopunct) #now it has no punctuations 
#clean_mess = [word for word in nopunct.split() if word.lower() not in stopwords.words('English')] #now we remove the stopwords 
#print(clean_mess)

def text_process(mess):
    #remove punctuation
    # remove stopwords
    # return list of clean words
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('English')] 

#tokenization - converting normal text string into clean version 
messages['message'].head(5).apply(text_process)

bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
mess4 =messages['message'][3]
bow4 = bow_transformer.transform([mess4])
print(bow4)