import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
import pickle
import scipy
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Læs og omskriv filen til at have headers.
data = pd.read_csv("SMSSpamCollection", delimiter='\t')
data.to_csv('SMSSpamCollection.csv', header=["Kategori", "SMS"], index=False)

# Læs nye fil data til df
df = pd.read_csv("SMSSpamCollection.csv")

# Tæl froskellige værdier der er i columns.
# print(df.shape)

# Print de første 5 rows.
# print(df.head())

# Tæl hvor mange værdier der er af hver kategori værdi.
# print(df['Kategori'].value_counts())

# Lav en plot graph over antal af kategorier fra Kategori column.
# df['Kategori'].value_counts().plot.bar(rot=50, title='Antal af kategori værdier')
# plt.show()

# Statistik over værdier i hvert row fra filen
# print(df['SMS'].str.split().apply(len).describe())
# Gennemsnit af ord pr. row.
# print("Rows/Ord-per-row forhold er:", round(5571 / 15.59))
# Print det row med den korteste værdi, som er ét ord. "Yup"
# print(df[df['SMS'].str.split().apply(len) == 1]['SMS'].values[0])
# Print det row med fleste værdier, som er 171 ord.
# print(df[df['SMS'].str.split().apply(len) == 171]['SMS'].values[0][:1000])

# Ny colonne med kategori værdien pr. row.
# le = LabelEncoder().fit(df["Kategori"])
# df['Kategori_værdi'] = le.transform(df["Kategori"])
# print(df.head())

# Fjern dupliketter og print de to kategorier spam / ham som er i kategori_værdi
# df_categories = df[['Kategori', 'Kategori_værdi']].drop_duplicates().sort_values('Kategori_værdi')
# print(df_categories)

# Split test data til train og test datasæt.
# x_train, x_test, y_train, y_test = train_test_split(
#    df['SMS'], df['Kategori_værdi'], test_size=.2, stratify=df['Kategori'], random_state=42)

# Print første værdi af x_train data, max 1000 karakterer.
# print(x_train[0][:1000])


# tyvstjålet ..
# def tfidf_transform(x_train, x_test):
#    kwargs = {
#        'ngram_range': (1, 1),  # Use 1-grams + 2-grams.
#        'analyzer': 'word',  # Split text into word tokens.
#        'min_df': 1,
#        'stop_words': "english",
#    }
#    vectorizer = TfidfVectorizer(**kwargs)
# Learn vocabulary from training texts and vectorize training texts.
#    x_train_transformed = vectorizer.fit_transform(x_train)
# Vectorize validation texts.
#    x_test_transformed = vectorizer.transform(x_test)
#    return x_train_transformed, x_test_transformed


# tfidf_train, tfidf_test = tfidf_transform(x_train, x_test)
# print(tfidf_train.shape)  # (4456, 7441)

# Print første test data efter tfidf transformationen
# print(tfidf_train[0].data)  # [0.64582673 0.52647801 0.55292743]

# train_tfidf_dense = scipy.sparse.csr_matrix.todense(tfidf_train)
# print(train_tfidf_dense[0])  # [[0. 0. 0. ... 0. 0. 0.]]
# print(len(train_tfidf_dense))  # 4456
# print(train_tfidf_dense[0][train_tfidf_dense[0] != 0][0])  # [[0.52647801 0.64582673 0.55292743]]


# classification models med sickit library

# print(df.shape, df['Kategori'].nunique())
# df.head(2)  # (5571, 3) 2

X_train, X_test, y_train, y_test = train_test_split(
    df['SMS'], df['Kategori'], test_size=.10, stratify=df['Kategori'], random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (4456,) (1115,) (4456,) (1115,)


# Tyvstjålet SVC pipeline..
def svc_pipleline():
    return Pipeline(
        [
            (
                "tfidf_vector_com",
                TfidfVectorizer(
                    input="array",
                    norm="l2",
                    max_features=None,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                SVC(
                    C=10,
                    kernel="rbf",
                    gamma=0.1,
                    probability=True,
                    class_weight=None,
                ),
            ),
        ]
    )


def print_metrics(pred_test, y_test, pred_train, y_train):
    print("test accuracy", str(np.mean(pred_test == y_test)))
    print("train accuracy", str(np.mean(pred_train == y_train)))
    print("\n Metrics and Confusion for SVM \n")
    print(metrics.confusion_matrix(y_test, pred_test))
    print(metrics.classification_report(y_test, pred_test))


# SVM præcision test
print('SVM Precision')
svc_pipe = svc_pipleline()
svc_pipe.fit(X_train, y_train)
pred_test = svc_pipe.predict(X_test)
pred_train = svc_pipe.predict(X_train)
print_metrics(pred_test, y_test, pred_train, y_train)

# Vis prediction af "trained data" i et confusion matrix display
cm = confusion_matrix(y_train, pred_train, labels=svc_pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc_pipe.classes_)
disp.plot()
plt.show()
