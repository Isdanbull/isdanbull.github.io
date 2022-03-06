# Down a rabbit hole in search of mushrooms

Welcome to my very first blog post. I hope you are as excited as I am? If not, I'm sure a little exploration of mushrooms should fix that!

So let's get started! 

## The data

For this project we will be working with the Mushroom Classification dataset from Kaggle ([link](https://www.kaggle.com/uciml/mushroom-classification)). First things first, let's 
perform our initial imports and load the data into a pandas DataFrame.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('mushrooms.csv')
```

Whenever I load a new dataframe the first thing I like to do is run .describe:


```python
data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>stalk-shape</th>
      <th>stalk-root</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>e</td>
      <td>x</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>b</td>
      <td>t</td>
      <td>b</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>w</td>
      <td>v</td>
      <td>d</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>4608</td>
      <td>3776</td>
      <td>5176</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
</div>



From the looks of things we are dealing with 8124 observations, each containing various catagorical data. It dosn't look like we have any missing data but let's check to be sure:


```python
data.shape
```




    (8124, 23)



Yep, looking good. Our next step is to figure out what all of these letters mean - fortunetly kaggle provides a data dictionary. Let's use it to map the letters to their meanings and run the describe again:


```python
#hide
#credit the Awesome Neil Molky
mush_keys = {}
laziness = """class: edible=e, poisonous=p
cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises: yes=t,no=f
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
gill-attachment: attached=a,descending=d,free=f,notched=n
gill-spacing: close=c,crowded=w,distant=d
gill-size: broad=b,narrow=n
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
stalk-shape: enlarging=e,tapering=t
stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
veil-type: partial=p,universal=u
veil-color: brown=n,orange=o,white=w,yellow=y
ring-number: none=n,one=o,two=t
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d"""
laziness = laziness.split("\n")
for l in laziness:
    col_list = l.split(":")[1]
    col_list = col_list.split(",")
    col_dict = {}
    for col in col_list:
        dict_hack = col.split("=")
        col_dict[dict_hack[1]] = dict_hack[0].strip()
    mush_keys[l.split(":")[0]] = col_dict

data.replace(mush_keys, inplace=True)

renamer = {}
for col in data.columns: 
    renamer[col] = col.replace('-','_')
data.rename(renamer, axis='columns', inplace=True)
```


```python
#hide
data.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap_shape</th>
      <th>cap_surface</th>
      <th>cap_color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill_attachment</th>
      <th>gill_spacing</th>
      <th>gill_size</th>
      <th>gill_color</th>
      <th>stalk_shape</th>
      <th>stalk_root</th>
      <th>stalk_surface_above_ring</th>
      <th>stalk_surface_below_ring</th>
      <th>stalk_color_above_ring</th>
      <th>stalk_color_below_ring</th>
      <th>veil_type</th>
      <th>veil_color</th>
      <th>ring_number</th>
      <th>ring_type</th>
      <th>spore_print_color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>edible</td>
      <td>convex</td>
      <td>scaly</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>buff</td>
      <td>tapering</td>
      <td>bulbous</td>
      <td>smooth</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>white</td>
      <td>several</td>
      <td>woods</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>4608</td>
      <td>3776</td>
      <td>5176</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
</div>



Much better! Most of the columns seem pretty self explanitory. The exception is the class column which turns out to have two unique values: 'edible' and 'poisonous'. Well, as budding (or in this case sporing??) data scientists our next step
seems obvious - let's see how well we can classify mushrooms as poisonous or edible!

## EDA

Let's take a look at how our data looks when broken down by class:



```python
sns.set_palette('colorblind')
to_plot = list(data.columns)
to_plot.remove('class')
fig, axs = plt.subplots(5,5, figsize=(20,30))
for i in range(5):
    for n, x in enumerate(to_plot[5*i:5*(i+1)]):
        ax = axs[n,i]
        plt.sca(ax)
        plt.xticks(rotation=30)
        sns.countplot(data=data, x=x, hue='class', ax=ax)
        

```


    
![](/images/output_11_0.png)
    


That's... a little overwhelming but we can certainly make out some interesting points:

- veil_type is a useless feature - all of our mushrooms have partial veils.
- All poisonous mushrooms appear to have white veils.
- All mushrooms with buff coloured gills are poisonous.
- All mushrooms with large rings are poisonous.
- Odor seems to be a very good predictor - any mushroom with a smell can be classified on the alone

We could investigate some of these relationships further but I get the strong feeling that several of these features are going to do a very good job of predicting so; let's take a look!

## If a mushroom falls in a random forest and noones there to see...

To begin with lets try out a random forest. It will be easy to fit and should give us an idea of how challenging our task will be. We will split our data into a training as a test set, fit a very basic Random Forest Classifier (RFC) 
use it to make some predictions and then evaluate them. Our key metric here will be recall i.e. the proportion of poisonous mushrooms we correctly identify (I can live with not eating an edible mushroom. Eating a poisonous one on the
otherhand... well let's not find out!)



```python
from sklearn.model_selection import train_test_split

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

One of the advantages of RFC's is that we can be very laissez faire about what we feed them. Unlike some of the more statistical ML algorithms RFC's just wont use anything problematic so, besides stripping out our target variable we will
give the model our data as is. The one thing we will need to do is encode our data numerically but unlike many other algorithms, RFC's don't need it to be One Hot Encoded so we shall use an Ordinal Encoder, arbitrarily assigning each
unique category to an integer for each feature. In the background I have manually assigned class in order to ensure poisonous is given the value of 1.

It's probably overkill for such a simple opporation but I am going to use an SKlearn pipeline for funsies.


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



rf = RandomForestClassifier()
oe = OrdinalEncoder()

steps = [('encoder', oe), ('random_forest', rf)]
pipe = Pipeline(steps)

pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('encoder', OrdinalEncoder()),
                    ('random_forest', RandomForestClassifier())])



Now that we have fitted our model to the training set we can apply it to the unseen test set and then do some evaluation. Lets look at the Recall and overall Accuracy.


```python
from sklearn.metrics import accuracy_score, recall_score

y_pred = pipe.predict(X_test)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Our first model achieved an accuracy of {accuracy} and a recall of {recall}')
```

    Our first model achieved an accuracy of 1.0 and a recall of 1.0
    

... Well that went better then expected. Let's come up with a new plan!

## Less is more

Now that we know that mushrooms are very predictible (sorry Mycologiests, looks like the machines are here for your jobs) we shall have to find something a little bit more challenging. 

Lets take a look at what our model was using.


```python
feature_names = X_train.columns
importances = pipe['random_forest'].feature_importances_
std = np.std([pipe['random_forest'].feature_importances_ for tree in pipe['random_forest'].estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

```


    
![](/images/output_22_0.png)    


As we suspected during the EDA odor is a very powerful feature. Looking back at spore_print_color it's also not suprising that the model has honed in on it. veil_color on the other hand is
suprisingly irrelivant. 

We can see from the plot that there are definitly some features (veil_type chief among them) that we could probably do without. Lets see what we can get away with losing. We will build a function
that runs a model, removes the least important feature and runs the model again, continuing until it has reached some lower level of efficiancy.


```python
def feature_eliminator(df, target, pipe, min_prob=1, initial_drops=[]):
    drops = initial_drops + [target]
    while True:      
        X = df.drop(drops, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X)
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        if (accuracy < min_prob):
            try:
                print(f'Minimum # of features with at least {min_prob*100}% accuracy: {len(feature_names)}, accuracy:{round(old_accuracy, 3)*100}%, recall:{round(old_recall, 3)*100}%')
                return list(feature_names)
                
            except:
                print(f'no features give {min_prob*100}% accuracy!')
                return
        elif (len(X.columns) < 2):
            print(f'Only one feature needed to achieve {min_prob*100}% accuracy. Accuracy:{round(accuracy, 3)*100}%, recall:{round(recall, 3)*100}%')
            return list(X.columns)
        feature_names = X.columns
        importances = pipe['random_forest'].feature_importances_
        forest_importances = pd.Series(importances, index=feature_names)
        drops += [forest_importances.idxmin()]
        old_accuracy = accuracy.copy()
        old_recall = recall.copy()
```

Our function has three exit clauses:
- if at the very start it can't reach its target it will imediatly tell us.
- Otherwise it will run untill it has no longer hits the target and then reports the last successful feature configuration.
- Finally if it can hit the target with only one feature it will report.


```python
feature_eliminator(data, 'class', pipe, min_prob=0.95)
```

    Only one feature needed to achieve 95.0% accuracy. Accuracy:98.5%, recall:96.89999999999999%
    




    ['odor']



I knew something smelled fishy! let's try again with some deodorized mushrooms:


```python
feature_eliminator(data, 'class', pipe, min_prob=0.95, initial_drops=['odor'])
```

    Minimum # of features with at least 95.0% accuracy: 2, accuracy:97.2%, recall:99.1%
    




    ['gill_size', 'spore_print_color']



Well... one more try with all three removed?


```python
feature_eliminator(data, 'class', pipe, min_prob=0.95, initial_drops=['odor', 'gill_size', 'spore_print_color'])
```

    Minimum # of features with at least 95.0% accuracy: 3, accuracy:95.89999999999999%, recall:93.89999999999999%
    




    ['ring_type', 'population', 'habitat']



That's enough of that I think! There are obviously a multitude of very strong predictors. We are going to have to be more creative to find a challenge in the world of mushrooms.

## Mushrooms for everyone!

Okay, let's set out a project. We shall build a usable field guide that is fully accessable. For this we will need:

- A model that has no risk of poisoning its users; that is 100% recall.
- The model should not use any features that might be inaccessable; in this case that means colors and odor.
- The model should be as simple as possible.

Our model will be a Decision Tree; we shall try to minimise the number of end leaves. We will use a GridsearchCV ensemble to try to find the best parameters and try different random seeds.

Our first step will be to build a function runs our GridSearchCV with increasing Maximum leaf values searching for one that achieves 100% recall. It will then produce the best decision tree it found and 
report its accuracy.


```python
from sklearn.model_selection import GridSearchCV

def leaf_finder(df, target, features, params, recall_limit=1.0, min_leaf=2, max_leaf=20):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    encoder = OrdinalEncoder()
    tree = DecisionTreeClassifier()
    for i in range(min_leaf,max_leaf+1):
        
        params['max_leaf_nodes'] = [i]
        grid = GridSearchCV(tree, params, scoring='recall')
        steps = [('encoding', encoder), ('modeling', grid)]
        pipeline = Pipeline(steps)
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        if recall >= recall_limit:
            print(f'Best estimator achieved accuracy of {round(accuracy,3)*100}%, using {i} leaves')
            return pipeline['modeling'].best_estimator_
    raise NameError(f'The function was not able to achieve 100% recall witin {max_leaf} leaves')
```

Eagle eyed readers may have noticed something a bit naughty about our function: We are predicting upon the entire dataset rather then just the training set. There is a reason for this; As we are looking for 100%
recall we are protected from overfitting - if the algorithm is achieving 100% on the whole data set it must necessarily be achieving it on the training set as well. Why doing it this way is desirable is because it is
possible for an algorithm to achieve high but not 100% recall on the training data but then, by fluke, get 100% on the test data. This protects us from that case.

Now, lets impliment it.


```python
features = list(data.columns)
for i in ['odor', 'spore_print_color', 'veil_color', 'stalk_color_below_ring', 'stalk_color_above_ring', 'gill_color', 'cap_color', 'class']:#
    features.remove(i)

params = {'criterion':['entropy', 'gini'],
              'random_state':range(10),
              'max_features':[7,9,11,13], 
              'max_depth':[3,5,7,9,11]}

best = leaf_finder(data, 'class', features, params, min_leaf=5, max_leaf=15)
```

    Best estimator achieved accuracy of 96.5%, using 8 leaves
    

8 leaves, not bad! An overall accuracy of 96.5% is completly acceptable as well. Lets take a look at the winning tree:


```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))
plot_tree(best, feature_names=features, class_names=['edible', 'poisonous'], filled=True)
plt.show()
```


![](/images/output_37_0.png)   

    



```python
#hide
data['stalk_root'] = data['stalk_root'].map({'bulbous': 1, 'club': 0, 'equal': 3, 'missing': 2, 'rooted': 4})
```

Can we make it any better? After a bit of experimentation I found two stratagies. Firstly gill_size seems to be a trap. While it is a great initial seporator the two groups it creats seem to be hard
to brake down further. Secondly the stalk root feature is obviously a very useful one, but the arbitraty order it has been assigned seems sub-optimal (as we can see from the fact it has been split on three
seperate times in our tree). I remaped the values to make the splitting more efficient.


```python

features = list(data.columns)
for i in ['odor', 'spore_print_color', 'veil_color', 'stalk_color_below_ring', 'stalk_color_above_ring', 'gill_color', 'cap_color', 'class', 'gill_size']:#
    features.remove(i)

params = {'criterion':['entropy', 'gini'],
              'random_state':range(10),
              'max_features':[7,9,11,13], 
              'max_depth':[3,5,7,9,11]}

best = leaf_finder(data, 'class', features, params, min_leaf=5, max_leaf=15)
```

    Best estimator achieved accuracy of 93.5%, using 6 leaves
    


```python
from sklearn.tree import plot_tree

plt.figure(figsize=(10,10))
plot_tree(best, feature_names=features, class_names=['edible', 'poisonous'], filled=True)
plt.show()
```


    

![](/images/output_41_0.png)    


And there we have it! Five questions that anyone can answer and we have sucessfully sorted out all of the inedible fungus. Trufflers rejoyce!
