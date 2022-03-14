# Pipeline implimentation

1. TOC
{:toc}


## Loading the data


```python
#hide
import pandas as pd
import numpy as np
```


```python
data = pd.read_csv('appliances_cleaned_final.csv')
data.dropna(inplace=True)
```

## Determine feature types


```python
def feature_catogoriser(df, target):
    num_features = []
    text_features = []
    for feature in df.columns:
        if ((df[feature].dtype) == 'O') & (feature != target):
            text_features.append(feature)
        elif feature != target:
            num_features.append(feature)

    return text_features, num_features
```

The features catogoriser runs over all features in the data frame, labeling Object types as
text columns and others as numerical columns.

We then give a list of features to be used (determined by our model development) and use the 
feature catogoriser to sort them.


```python
cols = ['answer_length', 'q_upper', 'a_upper', 'q_model',
       'q_do', 'a_model', 'a_imp_dim', 'q_pos_sentiment', 'a_pos_sentiment',
       'q_adverbs', 'q_determiners',
       'q_nouns', 'q_numerals', 'a_adpositions', 
       'a_pronouns', 'a_re_joined', 'q_punc_count', 'a_punc_ratio',
       'jaccard', 'lev_distance', 'our_label']


text_features, num_features = feature_catogoriser(data[cols], 'our_label')
```

## Fitting and Evaluating Model

The next series of functions run an experiment and output the model and an evaluation of it.

Experimenter takes our data and feature information along with a dictionary of classifiers 
and some other parameters. It performs our train/test split then runs and evaluates the 
model using each classifier in turn. We experimented with taking an ensemble aproach 
at this stage but stripped it out for readability as it was not used in our final model.


```python
from sklearn.model_selection import train_test_split
from sklearn import metrics


def experimenter(df, target, classifiers,threshold=0.5, text_features=[], num_features=[], 
                 stop_words=None, n_gram=(1,2), chi_k=300):
    features = text_features + num_features
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142, stratify=y)
    y_test = y_test.to_frame()
    
    out = pd.DataFrame()
    
    for key, value in classifiers.items():  #for each classifier provided run model and produce scores
        y_test[value], pl = modeling_f(X_train, X_test, y_train, key, features=features, 
                                  chi_k = chi_k, n_gram=n_gram)
        prediction = np.where(y_test[value] > threshold , 1, 0)
        out.loc[value,'accuracy'] = round(metrics.accuracy_score(y_test[target], prediction),3)
        out.loc[value,'precision'] = round(metrics.precision_score(y_test[target], prediction ),3)
        out.loc[value,'recall'] = round(metrics.recall_score(y_test[target], prediction),3)
        out.loc[value,'auc'] = round(metrics.roc_auc_score(y_test[target], y_test[value]),3)
    
    return out, pl
    
```

### The pipeline

We quickly define two functions which sort the columns into numerical and textual for textual features
we combine them into a single string (though in our final model we only use a single textual string).


```python
def select_num_columns(df):
    
    out = df.loc[:, num_features]
    return out
```


```python
def combine_text_columns(df):

    out = df.loc[:, text_features]
    out['text'] =""
    for col in df.columns:
        if col in text_features:
            out.text += " "
            out.text += df[col]
    return out.text
```

We then run the model. In cases where there is only numeric data this is simple - we scale the data and run
the model. If howerver there is textual data the process is more involved:
- We use a feature union to combine two sub pipelines
- The first simply selects and returns our numeric data
- The second selects text data, count vectorises it and then uses SelectKBest to take the most useful


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

def modeling_f(X_train, X_test, y_train, classifier, features, n_gram=(1,2),
               name='experiment', chi_k=300):
    
    # Perform preprocessing
    get_text_data = FunctionTransformer(combine_text_columns, validate=False)
    get_numeric_data = FunctionTransformer(select_num_columns, validate=False)
    
    # Instantiate pipeline: pl
    if text_features == []:
        pl = Pipeline([
                ('scale', MaxAbsScaler()),
                ('clf', classifier)
            ])
    else:
        pl = Pipeline([
                ('union', FeatureUnion(
                    transformer_list = [
                        ('numeric_features', Pipeline([
                            ('selector', get_numeric_data),
                        ])),
                        ('text_features', Pipeline([
                            ('selector', get_text_data),
                            ('vectorizer', CountVectorizer(analyzer = 'word',
                                                           max_features=350,
                                                           ngram_range = n_gram)),
                            ('dim_red', SelectKBest(chi2, k =chi_k))
                        ]))
                     ]
                )),
                ('scale', MaxAbsScaler()),
                ('clf', classifier)
            ])
    
    # Train test split
    X_train = X_train[features]
    X_test = X_test[features]
               
    # Make predictions
    pl.fit(X_train, y_train)
    pred = pl.predict_proba(X_test)[:,1]

    return pred, pl
```

We fit the pipeline on the training data and make our predictions



## Running it

First our Classifier:


```python
from sklearn.linear_model import LogisticRegression

w1, w2 = 100*data['our_label'].value_counts()/data.shape[0]
w = {0:w2, 1:w1}

classifiers = {LogisticRegression(class_weight=w, max_iter=400):'lr'}
```

Now we run our experiment:


```python
out, pl = experimenter(df=data, target='our_label',classifiers=classifiers, 
                       text_features=text_features, num_features=num_features, 
                       chi_k=100, n_gram=(1,3))
print(out)
```

        accuracy  precision  recall    auc
    lr     0.743      0.343   0.739  0.824
    

and finally the result is pickled


```python
from joblib import dump

dump(pl, 'amazon_model.joblib') 
```




    ['amazon_model.joblib']


