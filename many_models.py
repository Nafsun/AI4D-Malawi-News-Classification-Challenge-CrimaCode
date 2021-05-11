import pandas as pd
import numpy as np
import argparse

def train_model(num, algorithm, randomstate):
    df = pd.read_csv("Train.csv")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df["Label"])
    
    def new_shuffled_data(train_data, trunc):
        train_data_label = train_data["Label"].copy()
        train_data_text = train_data["Text"].copy()
        from nltk import word_tokenize
        for i in range(len(train_data_text)):
            #tokenize the sentence into words
            shuf = word_tokenize(train_data_text[i])
            #truncate words
            if(trunc < len(shuf)):
                shuf = shuf[:trunc]
                #replace the current text with the truncated text
                train_data_text[i] = " ".join(z for z in shuf)
            else:
                del train_data_text[i]
                del train_data_label[i]
        train_data_text = pd.DataFrame({"Text": train_data_text, "Label": train_data_label}, columns=["Text", "Label"])
        return train_data_text

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

    rand = num

    def articles_preprocessing(df):
        #convert to lowercase
        df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        #remove punctuations
        df['Text'] = df['Text'].str.replace('[^\w\s]', '')
        #remove stopwords
        df['Text'] = removeStopWords(df['Text'])   

        #convert labels into numbers
        df["Label"] = le.transform(df["Label"])
        df = df.sample(frac=1, random_state=randomstate).reset_index(drop=True)
        
        from sklearn import model_selection
        df["kfold"] = -1
        df = df.sample(frac=1, random_state=randomstate).reset_index(drop=True)
        y = df["Label"]
        kf = model_selection.StratifiedKFold(n_splits=5, random_state=randomstate, shuffle=True)
        for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'kfold'] = f

        df_train = df[df.kfold != rand].reset_index(drop=True)
        
        df_train = df_train.drop(["ID", "kfold"], axis=1)

        #add new shuffled data
        df_train_1 = new_shuffled_data(df_train, 100)
        df_train_2 = new_shuffled_data(df_train, 200)
        df_train_3 = new_shuffled_data(df_train, 300)
        df_train_4 = new_shuffled_data(df_train, 400)
        df_train_5 = new_shuffled_data(df_train, 500)
        df_train_6 = new_shuffled_data(df_train, 600)
        df_train_7 = new_shuffled_data(df_train, 700)
        df_train_8 = new_shuffled_data(df_train, 800)
        df_train_9 = new_shuffled_data(df_train, 900)
        df_train_10 = new_shuffled_data(df_train, 1000)
        df_train_11 = new_shuffled_data(df_train, 1100)
        df_train_12 = new_shuffled_data(df_train, 1200)
        df_train_13 = new_shuffled_data(df_train, 1300)
        df_train_14 = new_shuffled_data(df_train, 1400)
        df_train_15 = new_shuffled_data(df_train, 1500)
        df_train_16 = new_shuffled_data(df_train, 1600)
        df_train_17 = new_shuffled_data(df_train, 1700)
        df_train_18 = new_shuffled_data(df_train, 1800)
        df_train_19 = new_shuffled_data(df_train, 1900)
        df_train_20 = new_shuffled_data(df_train, 2000)
        df_train_21 = new_shuffled_data(df_train, 2100)
        df_train_22 = new_shuffled_data(df_train, 2200)
        df_train_23 = new_shuffled_data(df_train, 2300)
        df_train_24 = new_shuffled_data(df_train, 2400)
        df_train_25 = new_shuffled_data(df_train, 2500)
        df_train_26 = new_shuffled_data(df_train, 2600)
        df_train_27 = new_shuffled_data(df_train, 2700)
        df_train_28 = new_shuffled_data(df_train, 2800)
        df_train_29 = new_shuffled_data(df_train, 2900)
        df_train_30 = new_shuffled_data(df_train, 3000)
        df_train_31 = new_shuffled_data(df_train, 3100)
        df_train_32 = new_shuffled_data(df_train, 3200)
        df_train_33 = new_shuffled_data(df_train, 3300)
        df_train_34 = new_shuffled_data(df_train, 3400)
        df_train_35 = new_shuffled_data(df_train, 3500)
        df_train_36 = new_shuffled_data(df_train, 3600)
        df_train_37 = new_shuffled_data(df_train, 3700)
        df_train_38 = new_shuffled_data(df_train, 3800)
        df_train_39 = new_shuffled_data(df_train, 3900)
        df_train_40 = new_shuffled_data(df_train, 4000)
        
        df_train = pd.concat([df_train, df_train_1, df_train_2, df_train_3, df_train_4, df_train_5, df_train_6, df_train_7, 
                              df_train_8, df_train_9, df_train_10, df_train_11, df_train_12, df_train_13, df_train_14, df_train_15, 
                              df_train_16, df_train_17, df_train_18, df_train_19, df_train_20, df_train_21, df_train_22, df_train_23, 
                              df_train_24, df_train_25, df_train_26, df_train_27, df_train_28, df_train_29, df_train_30, df_train_31,
                              df_train_32, df_train_33, df_train_34, df_train_35, df_train_36, df_train_37, df_train_38, df_train_39, 
                              df_train_40], ignore_index=True)
        
        df_train = df_train.sample(frac=1, random_state=randomstate).reset_index(drop=True)
        
        df_valid = df[df.kfold == rand].reset_index(drop=True)

        x_train = df_train.drop(["Label"], axis=1)
        y_train = df_train.Label.values

        x_valid = df_valid.drop(["ID", "Label", "kfold"], axis=1)
        y_valid = df_valid.Label.values
        
        x_train = x_train["Text"]
        x_valid = x_valid["Text"]

        return x_train, y_train, x_valid, y_valid

    x_train, y_train, x_valid, y_valid = articles_preprocessing(df)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    import joblib

    vectorizer = joblib.load("tfidf.bin")

    tfidf_data = vectorizer.transform(x_train)
    
    model = None
    
    if(algorithm == "MultinomialNB"):
        model = MultinomialNB(alpha=0.001).fit(tfidf_data, y_train)  
    elif(algorithm == "XGBClassifier"):
        model = XGBClassifier(random_state=randomstate, max_depth=20, n_estimators=40, use_label_encoder=False, objective="multi:softmax", gamma=0.1, learning_rate=0.3).fit(tfidf_data, y_train, eval_metric="merror")
    elif(algorithm == "LogisticRegression"):
        model = LogisticRegression(random_state=randomstate, tol=5, penalty="l2", C=10, solver="lbfgs", multi_class="multinomial", max_iter=100, class_weight="balanced").fit(tfidf_data, y_train)
        
    new_tfidf = vectorizer.transform(x_valid)

    y_preds = model.predict(new_tfidf)

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_valid, y_preds)

    df_test = pd.read_csv("Test.csv")

    def articles_preprocessing(df, label=None, stratify=None):
        #convert to lowercase
        df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        #remove punctuations
        df['Text'] = df['Text'].str.replace('[^\w\s]', '')
        #remove stopwords
        df['Text'] = removeStopWords(df['Text'])        
        return df['Text']

    x_test = articles_preprocessing(df_test)

    new_tfidf = vectorizer.transform(x_test)

    prediction = model.predict(new_tfidf)

    pd_df = pd.DataFrame(np.c_[df_test.ID, prediction], columns=["ID", "Label"]).set_index("ID")

    mapper = {0: 'ARTS AND CRAFTS', 1: 'CULTURE', 2: 'ECONOMY', 3: 'EDUCATION', 4: 'FARMING', 5: 'FLOODING',
     6: 'HEALTH', 7: 'LAW/ORDER', 8: 'LOCALCHIEFS', 9: 'MUSIC', 10: 'OPINION/ESSAY', 11: 'POLITICS',
     12: 'RELATIONSHIPS', 13: 'RELIGION', 14: 'SOCIAL', 15: 'SOCIAL ISSUES', 16: 'SPORTS', 17: 'TRANSPORT',
     18: 'WILDLIFE/ENVIRONMENT', 19: 'WITCHCRAFT'}

    pd_df.Label = pd_df.Label.map(mapper)

    import joblib

    joblib.dump(model, f"{algorithm}-{rand}-{accuracy.round(3)}-validate-{randomstate}.bin")

    print("Executed", num, algorithm, accuracy, randomstate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--num", #number of kfold to use
        type=int
    )

    parser.add_argument(
        "--algorithm", #select an algorithm to train
        type=str
    )
    
    parser.add_argument(
        "--randomstate", #choose a random state
        type=int
    )
    
    args = parser.parse_args()
    
    train_model(
        num=args.num,
        algorithm=args.algorithm,
        randomstate=args.randomstate
    )
    