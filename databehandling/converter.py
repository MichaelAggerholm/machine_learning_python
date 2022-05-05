import numpy as np
import pandas as pd

# læs data fil
data = pd.read_csv("SMSSpamCollection", delimiter='\t')

# gem data til csv med category og value headers uden index
data.to_csv('SMSSpamCollection.csv', header=["Category", "Value"], index=False)

# indlæs csv til panda dataframe
df = pd.read_csv('SMSSpamCollection.csv')
# print(df)

# print(df.head())
# print(df['Category'].value_counts())

# spredning af data
print(str(round((365 / 2422), 2)) + '%')

# Balancer data
df_spam = df[df['Category'] == 'spam']
df_ham = df[df['Category'] == 'ham']
#print("Ham Dataset Shape:", df_ham.shape)
#print("Spam Dataset Shape:", df_spam.shape)

# Trim længeste dataframe ned til samme længde som korteste dataframe
df_ham_downsampled = df_ham.sample(df_spam.shape[0])
# print(df_ham_downsampled.shape)

# Flet begge dataframes sammen til et balanceret datasæt
df_balanced = pd.concat([df_spam, df_ham_downsampled])
# print(df_balanced['Category'].value_counts())

# print(df_balanced.sample(10))

# "One Hot Encoding"
df_balanced['spam'] = df_balanced['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(df_balanced['spam'].value_counts())
print(df_balanced.sample(10))
