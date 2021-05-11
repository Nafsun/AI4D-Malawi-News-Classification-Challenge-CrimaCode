import pandas as pd
import numpy as np

df = pd.read_csv("Train.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df["Label"])

def removeStopWords(df):
    import re
    counter = 0
    df = df.copy()
    #open the stop word file
    stop_word = ""
    with open("stop_nyanja_words.txt", "r") as f:
        stop_word = f.read()
    #tokenize the stop words
    from nltk import word_tokenize
    stop_word = list(set(word_tokenize(stop_word)))
    for i in df:
        #tokenize words of the article
        word_tokens = word_tokenize(i)
        #remove the stopwords
        filtered_sentence = [w for w in word_tokens if not w in stop_word] 
        #join the sentence
        filtered_sentence = " ".join([x for x in filtered_sentence])
        #remove all numbers
        filtered_sentence = re.sub("\d+", "", filtered_sentence)
        #replace the current version with the filtered one
        df[counter] = filtered_sentence
        counter += 1
    return df

import os
import joblib
import numpy as np

vectorizer = joblib.load("tfidf.bin")

df_test = pd.read_csv("Test.csv")

def articles_preprocessing(df):
    #convert to lowercase
    df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    #remove punctuations
    df['Text'] = df['Text'].str.replace('[^\w\s]', '')
    #remove stopwords
    df['Text'] = removeStopWords(df['Text'])        
    return df['Text']

x_test = articles_preprocessing(df_test)

new_tfidf = vectorizer.transform(x_test)

y_preds = np.array([])
counter = 0

for i in os.listdir():
    if(i.endswith("bin") == True):
        if("tfidf.bin" != i):
            model = joblib.load(i)
            if(counter == 0):
                y_preds = model.predict_proba(new_tfidf)
            else:
                y_preds = y_preds + model.predict_proba(new_tfidf)
            print(i)
            counter += 1

overall = y_preds / counter

save_pos = []
counter = 0

for i in range(len(overall[:, 1])):
    l = list(overall[counter])
    m = max(l)
    save_pos.append(l.index(m))
    counter += 1
counter = 0

prediction = save_pos

pd_df = pd.DataFrame(np.c_[df_test.ID, prediction], columns=["ID", "Label"]).set_index("ID")

mapper = {0: 'ARTS AND CRAFTS', 1: 'CULTURE', 2: 'ECONOMY', 3: 'EDUCATION', 4: 'FARMING', 5: 'FLOODING',
 6: 'HEALTH', 7: 'LAW/ORDER', 8: 'LOCALCHIEFS', 9: 'MUSIC', 10: 'OPINION/ESSAY', 11: 'POLITICS',
 12: 'RELATIONSHIPS', 13: 'RELIGION', 14: 'SOCIAL', 15: 'SOCIAL ISSUES', 16: 'SPORTS', 17: 'TRANSPORT',
 18: 'WILDLIFE/ENVIRONMENT', 19: 'WITCHCRAFT'}

pd_df.Label = pd_df.Label.map(mapper)

pd_df.to_csv("new_pred.csv")
