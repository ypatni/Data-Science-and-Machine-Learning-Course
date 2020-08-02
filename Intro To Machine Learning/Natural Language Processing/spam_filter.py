import nltk 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
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
#print(messages[messages['length'] ==910]['message'].iloc[0]) very creepy love letter don't look at it 
messages.hist(column = 'length', by = 'label', bins = 60, figsize=(12,4))
plt.show() #ham texts seem to be centered around 50 words while spam usually has around 150 words
