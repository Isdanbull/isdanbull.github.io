<div class="cell markdown">

# Down a rabbit hole in search of mushrooms

Welcome to my very first blog post. I hope you are as excited as I am?
If not, I'm sure a little exploration of mushrooms should fix that\!

So let's get started\!

## The data

For this project we will be working with the Mushroom Classification
dataset from Kaggle
([link](https://www.kaggle.com/uciml/mushroom-classification)). First
things first, let's perform our initial imports and load the data into a
pandas DataFrame.

</div>

<div class="cell code" data-execution_count="76">

``` python
#hide
!pip install nbdev
```

<div class="output stream stdout">

    Collecting nbdev
      Downloading nbdev-1.2.2-py3-none-any.whl (48 kB)
         |████████████████████████████████| 48 kB 6.9 MB/s             
    at>=4.4.0
      Downloading nbformat-5.1.3-py3-none-any.whl (178 kB)
         |████████████████████████████████| 178 kB 110.1 MB/s            
    l
      Downloading PyYAML-6.0-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (701 kB)
         |████████████████████████████████| 701 kB 81.3 MB/s            
    anylinux2010_x86_64.whl (427 kB)
         |████████████████████████████████| 427 kB 95.3 MB/s            
    q>=13
      Downloading pyzmq-22.3.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
         |████████████████████████████████| 1.1 MB 99.1 MB/s            
    istune<2,>=0.8.1
      Downloading mistune-0.8.4-py2.py3-none-any.whl (16 kB)
    Collecting nbclient<0.6.0,>=0.5.0
      Downloading nbclient-0.5.12-py3-none-any.whl (71 kB)
         |████████████████████████████████| 71 kB 16.1 MB/s            
    ents
      Downloading jupyterlab_pygments-0.1.2-py2.py3-none-any.whl (4.6 kB)
    Collecting pandocfilters>=1.4.1
      Downloading pandocfilters-1.5.0-py2.py3-none-any.whl (8.7 kB)
    Collecting testpath
      Downloading testpath-0.6.0-py3-none-any.whl (83 kB)
         |████████████████████████████████| 83 kB 4.6 MB/s              
    ents>=2.4.1
      Downloading Pygments-2.11.2-py3-none-any.whl (1.1 MB)
         |████████████████████████████████| 1.1 MB 95.0 MB/s            
    l
      Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Collecting jinja2>=2.4
      Downloading Jinja2-3.0.3-py3-none-any.whl (133 kB)
         |████████████████████████████████| 133 kB 84.6 MB/s            
    a!=2.5.0,>=2.4
      Downloading jsonschema-4.4.0-py3-none-any.whl (72 kB)
         |████████████████████████████████| 72 kB 2.1 MB/s             
    anylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.9 MB)
         |████████████████████████████████| 1.9 MB 87.9 MB/s            
    atplotlib-inline<0.2.0,>=0.1.0
      Downloading matplotlib_inline-0.1.3-py3-none-any.whl (8.2 kB)
    Collecting ipython>=7.23.1
      Downloading ipython-8.1.1-py3-none-any.whl (750 kB)
         |████████████████████████████████| 750 kB 96.1 MB/s            
    pt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0
      Downloading prompt_toolkit-3.0.28-py3-none-any.whl (380 kB)
         |████████████████████████████████| 380 kB 90.8 MB/s            
    anylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
    Collecting pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0
      Downloading pyrsistent-0.18.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (119 kB)
         |████████████████████████████████| 119 kB 91.1 MB/s            
    portlib-resources>=1.4.0
      Downloading importlib_resources-5.4.0-py3-none-any.whl (28 kB)
    Collecting attrs>=17.4.0
      Downloading attrs-21.4.0-py2.py3-none-any.whl (60 kB)
         |████████████████████████████████| 60 kB 14.4 MB/s            
    inado>=0.8.3
      Downloading terminado-0.13.2-py3-none-any.whl (14 kB)
    Collecting argon2-cffi
      Downloading argon2_cffi-21.3.0-py3-none-any.whl (14 kB)
    Collecting prometheus-client
      Downloading prometheus_client-0.13.1-py3-none-any.whl (57 kB)
         |████████████████████████████████| 57 kB 9.4 MB/s             
    anylinux_2_17_x86_64.manylinux2014_x86_64.whl (86 kB)
         |████████████████████████████████| 86 kB 8.7 MB/s             
    anylinux_2_12_x86_64.manylinux2010_x86_64.whl (446 kB)
         |████████████████████████████████| 446 kB 64.2 MB/s            
    portlib-resources, attrs, wcwidth, tornado, pyzmq, python-dateutil, pyparsing, pycparser, pure-eval, ptyprocess, parso, nest-asyncio, jupyter-core, jsonschema, ipython-genutils, executing, entrypoints, asttokens, webencodings, stack-data, setuptools, pygments, prompt-toolkit, pickleshare, pexpect, packaging, nbformat, matplotlib-inline, MarkupSafe, jupyter-client, jedi, decorator, cffi, backcall, testpath, pandocfilters, nbclient, mistune, jupyterlab-pygments, jinja2, ipython, defusedxml, debugpy, bleach, argon2-cffi-bindings, terminado, Send2Trash, prometheus-client, nbconvert, ipykernel, argon2-cffi, pip, notebook, widgetsnbextension, qtpy, jupyterlab-widgets, fastcore, qtconsole, pyyaml, jupyter-console, ipywidgets, ghapi, jupyter, fastrelease, nbdev
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    dc-workspace 1.0.0 requires jupytext, which is not installed.
    tensorflow 2.4.1 requires numpy~=1.19.2, but you have numpy 1.22.2 which is incompatible.
    tensorflow 2.4.1 requires six~=1.15.0, but you have six 1.16.0 which is incompatible.
    networkx 2.5.1 requires decorator<5,>=4.3, but you have decorator 5.1.1 which is incompatible.
    jsonrpcserver 5.0.5 requires jsonschema<4, but you have jsonschema 4.4.0 which is incompatible.
    awscli 1.19.53 requires PyYAML<5.5,>=3.10, but you have pyyaml 6.0 which is incompatible.
    pythonwhat 2.23.1 requires asttokens~=1.1.10, but you have asttokens 2.0.5 which is incompatible.
    pythonwhat 2.23.1 requires dill~=0.2.7.1, but you have dill 0.3.3 which is incompatible.
    pythonwhat 2.23.1 requires jinja2~=2.10, but you have jinja2 3.0.3 which is incompatible.
    protowhat 2.0.1 requires jinja2~=2.10, but you have jinja2 3.0.3 which is incompatible.
    Successfully installed MarkupSafe-2.1.0 Send2Trash-1.8.0 argon2-cffi-21.3.0 argon2-cffi-bindings-21.2.0 asttokens-2.0.5 attrs-21.4.0 backcall-0.2.0 bleach-4.1.0 cffi-1.15.0 debugpy-1.5.1 decorator-5.1.1 defusedxml-0.7.1 entrypoints-0.4 executing-0.8.3 fastcore-1.3.29 fastrelease-0.1.12 ghapi-0.1.19 importlib-resources-5.4.0 ipykernel-6.9.1 ipython-8.1.1 ipython-genutils-0.2.0 ipywidgets-7.6.5 jedi-0.18.1 jinja2-3.0.3 jsonschema-4.4.0 jupyter-1.0.0 jupyter-client-7.1.2 jupyter-console-6.4.2 jupyter-core-4.9.2 jupyterlab-pygments-0.1.2 jupyterlab-widgets-1.0.2 matplotlib-inline-0.1.3 mistune-0.8.4 nbclient-0.5.12 nbconvert-6.4.2 nbdev-1.2.2 nbformat-5.1.3 nest-asyncio-1.5.4 notebook-6.4.8 packaging-21.3 pandocfilters-1.5.0 parso-0.8.3 pexpect-4.8.0 pickleshare-0.7.5 pip-22.0.3 prometheus-client-0.13.1 prompt-toolkit-3.0.28 ptyprocess-0.7.0 pure-eval-0.2.2 pycparser-2.21 pygments-2.11.2 pyparsing-3.0.7 pyrsistent-0.18.1 python-dateutil-2.8.2 pyyaml-6.0 pyzmq-22.3.0 qtconsole-5.2.2 qtpy-2.0.1 setuptools-60.9.3 six-1.16.0 stack-data-0.2.0 terminado-0.13.2 testpath-0.6.0 tornado-6.1 traitlets-5.1.1 wcwidth-0.2.5 webencodings-0.5.1 widgetsnbextension-3.5.2 zipp-3.7.0
    WARNING: Target directory /home/repl/.local/lib/python3.8/site-packages/__pycache__ already exists. Specify --upgrade to force replacement.
    WARNING: Target directory /home/repl/.local/lib/python3.8/site-packages/bin already exists. Specify --upgrade to force replacement.
    WARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.

</div>

</div>

<div class="cell code">

``` python
```

</div>

<div class="cell code" data-execution_count="75">

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('mushrooms.csv')
```

<div class="output error" data-ename="ModuleNotFoundError" data-evalue="No module named &#39;nbdev&#39;">

    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    /tmp/ipykernel_1783/4164738541.py in <module>
          4 import seaborn as sns
          5 
    ----> 6 import nbdev
          7 
          8 pd.set_option('display.max_rows', None)
    
    ModuleNotFoundError: No module named 'nbdev'

</div>

</div>

<div class="cell markdown">

Whenever I load a new dataframe the first thing I like to do is run
.describe:

</div>

<div class="cell code" data-execution_count="2">

``` python
data.describe(include='all')
```

<div class="output execute_result" data-execution_count="2">

``` json
{"is_truncated":false,"table":{"data":[{"ring-number":8124,"stalk-surface-above-ring":8124,"spore-print-color":8124,"ring-type":8124,"stalk-root":8124,"gill-attachment":8124,"cap-surface":8124,"gill-spacing":8124,"cap-shape":8124,"stalk-surface-below-ring":8124,"habitat":8124,"stalk-shape":8124,"veil-type":8124,"gill-color":8124,"stalk-color-above-ring":8124,"cap-color":8124,"stalk-color-below-ring":8124,"population":8124,"index":"count","odor":8124,"class":8124,"gill-size":8124,"veil-color":8124,"bruises":8124},{"ring-number":3,"stalk-surface-above-ring":4,"spore-print-color":9,"ring-type":5,"stalk-root":5,"gill-attachment":2,"cap-surface":4,"gill-spacing":2,"cap-shape":6,"stalk-surface-below-ring":4,"habitat":7,"stalk-shape":2,"veil-type":1,"gill-color":12,"stalk-color-above-ring":9,"cap-color":10,"stalk-color-below-ring":9,"population":6,"index":"unique","odor":9,"class":2,"gill-size":2,"veil-color":4,"bruises":2},{"ring-number":"o","stalk-surface-above-ring":"s","spore-print-color":"w","ring-type":"p","stalk-root":"b","gill-attachment":"f","cap-surface":"y","gill-spacing":"c","cap-shape":"x","stalk-surface-below-ring":"s","habitat":"d","stalk-shape":"t","veil-type":"p","gill-color":"b","stalk-color-above-ring":"w","cap-color":"n","stalk-color-below-ring":"w","population":"v","index":"top","odor":"n","class":"e","gill-size":"b","veil-color":"w","bruises":"f"},{"ring-number":7488,"stalk-surface-above-ring":5176,"spore-print-color":2388,"ring-type":3968,"stalk-root":3776,"gill-attachment":7914,"cap-surface":3244,"gill-spacing":6812,"cap-shape":3656,"stalk-surface-below-ring":4936,"habitat":3148,"stalk-shape":4608,"veil-type":8124,"gill-color":1728,"stalk-color-above-ring":4464,"cap-color":2284,"stalk-color-below-ring":4384,"population":4040,"index":"freq","odor":3528,"class":4208,"gill-size":5612,"veil-color":7924,"bruises":4748}],"schema":{"primaryKey":["index"],"pandas_version":"0.20.0","fields":[{"name":"index","type":"string"},{"name":"class","type":"string"},{"name":"cap-shape","type":"string"},{"name":"cap-surface","type":"string"},{"name":"cap-color","type":"string"},{"name":"bruises","type":"string"},{"name":"odor","type":"string"},{"name":"gill-attachment","type":"string"},{"name":"gill-spacing","type":"string"},{"name":"gill-size","type":"string"},{"name":"gill-color","type":"string"},{"name":"stalk-shape","type":"string"},{"name":"stalk-root","type":"string"},{"name":"stalk-surface-above-ring","type":"string"},{"name":"stalk-surface-below-ring","type":"string"},{"name":"stalk-color-above-ring","type":"string"},{"name":"stalk-color-below-ring","type":"string"},{"name":"veil-type","type":"string"},{"name":"veil-color","type":"string"},{"name":"ring-number","type":"string"},{"name":"ring-type","type":"string"},{"name":"spore-print-color","type":"string"},{"name":"population","type":"string"},{"name":"habitat","type":"string"}]}}}
```

</div>

</div>

<div class="cell markdown">

From the looks of things we are dealing with 8124 observations, each
containing various catagorical data. It dosn't look like we have any
missing data but let's check to be sure:

</div>

<div class="cell code" data-execution_count="3">

``` python
data.shape
```

<div class="output execute_result" data-execution_count="3">

    (8124, 23)

</div>

</div>

<div class="cell markdown">

Yep, looking good. Our next step is to figure out what all of these
letters mean - fortunetly kaggle provides a data dictionary. Let's use
it to map the letters to their meanings and run the describe again:

</div>

<div class="cell code" data-execution_count="70" data-tags="[]">

``` python
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

</div>

<div class="cell code" data-execution_count="5" data-jupyter="{&quot;source_hidden&quot;:true}" data-tags="[]">

``` python
#hide
data.describe()
```

<div class="output execute_result" data-execution_count="5">

``` json
{"is_truncated":false,"table":{"data":[{"gill_color":8124,"spore_print_color":8124,"stalk_color_below_ring":8124,"cap_color":8124,"habitat":8124,"stalk_surface_above_ring":8124,"ring_number":8124,"veil_color":8124,"gill_size":8124,"stalk_surface_below_ring":8124,"cap_shape":8124,"ring_type":8124,"stalk_root":8124,"gill_spacing":8124,"gill_attachment":8124,"cap_surface":8124,"population":8124,"index":"count","odor":8124,"stalk_color_above_ring":8124,"class":8124,"bruises":8124,"stalk_shape":8124,"veil_type":8124},{"gill_color":12,"spore_print_color":9,"stalk_color_below_ring":9,"cap_color":10,"habitat":7,"stalk_surface_above_ring":4,"ring_number":3,"veil_color":4,"gill_size":2,"stalk_surface_below_ring":4,"cap_shape":6,"ring_type":5,"stalk_root":5,"gill_spacing":2,"gill_attachment":2,"cap_surface":4,"population":6,"index":"unique","odor":9,"stalk_color_above_ring":9,"class":2,"bruises":2,"stalk_shape":2,"veil_type":1},{"gill_color":"buff","spore_print_color":"white","stalk_color_below_ring":"white","cap_color":"brown","habitat":"woods","stalk_surface_above_ring":"smooth","ring_number":"one","veil_color":"white","gill_size":"broad","stalk_surface_below_ring":"smooth","cap_shape":"convex","ring_type":"pendant","stalk_root":"bulbous","gill_spacing":"close","gill_attachment":"free","cap_surface":"scaly","population":"several","index":"top","odor":"none","stalk_color_above_ring":"white","class":"edible","bruises":"no","stalk_shape":"tapering","veil_type":"partial"},{"gill_color":1728,"spore_print_color":2388,"stalk_color_below_ring":4384,"cap_color":2284,"habitat":3148,"stalk_surface_above_ring":5176,"ring_number":7488,"veil_color":7924,"gill_size":5612,"stalk_surface_below_ring":4936,"cap_shape":3656,"ring_type":3968,"stalk_root":3776,"gill_spacing":6812,"gill_attachment":7914,"cap_surface":3244,"population":4040,"index":"freq","odor":3528,"stalk_color_above_ring":4464,"class":4208,"bruises":4748,"stalk_shape":4608,"veil_type":8124}],"schema":{"primaryKey":["index"],"pandas_version":"0.20.0","fields":[{"name":"index","type":"string"},{"name":"class","type":"string"},{"name":"cap_shape","type":"string"},{"name":"cap_surface","type":"string"},{"name":"cap_color","type":"string"},{"name":"bruises","type":"string"},{"name":"odor","type":"string"},{"name":"gill_attachment","type":"string"},{"name":"gill_spacing","type":"string"},{"name":"gill_size","type":"string"},{"name":"gill_color","type":"string"},{"name":"stalk_shape","type":"string"},{"name":"stalk_root","type":"string"},{"name":"stalk_surface_above_ring","type":"string"},{"name":"stalk_surface_below_ring","type":"string"},{"name":"stalk_color_above_ring","type":"string"},{"name":"stalk_color_below_ring","type":"string"},{"name":"veil_type","type":"string"},{"name":"veil_color","type":"string"},{"name":"ring_number","type":"string"},{"name":"ring_type","type":"string"},{"name":"spore_print_color","type":"string"},{"name":"population","type":"string"},{"name":"habitat","type":"string"}]}}}
```

</div>

</div>

<div class="cell markdown">

Much better\! Most of the columns seem pretty self explanitory. The
exception is the class column which turns out to have two unique values:
'edible' and 'poisonous'. Well as budding (or in this case sporing??)
data scientists our next step seems obvious - let's see how well we can
classify mushrooms as poisonous or edible\!

</div>

<div class="cell markdown">

## EDA

Well, let's take a look at how our data looks when broken down by class:

</div>

<div class="cell code" data-execution_count="6">

``` python
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

<div class="output display_data">

![](1b4daed9dd737038fd52393a4428877ac8d266ce.png)

</div>

</div>

<div class="cell markdown">

That's... a little overwhelming but we can certainly make out some
interesting points:

  - veil\_type is a useless feature - all of our mushrooms have partial
    veils.
  - All poisonous mushrooms appear to have white veils.
  - All mushrooms with buff coloured gills are poisonous.
  - All mushrooms with large rings are poisonous.
  - Odor seems to be a very good predictor - any mushroom with a smell
    can be classified on the alone

We could investigate some of these relationships further but I get the
strong feeling that several of these features are going to do a very
good job of predicting so; let's take a look\!

</div>

<div class="cell markdown">

## If a mushroom falls in a random forest and noones there to see...

To begin with lets try out a random forest. It will be easy to fit and
should give us an idea of how challenging our task will be. We will
split our data into a training as a test set, fit a very basic Random
Forest Classifier (RFC) use it to make some predictions and then
evaluate them. Our key metric here will be recall i.e. the proportion of
poisonous mushrooms we correctly identify (I can live with not eating an
edible mushroom. Eating a poisonous one on the otherhand... well let's
not find out\!)

</div>

<div class="cell code" data-execution_count="71">

``` python
#hide
data['class'] = data['class'].map({'poisonous':1, 'edible':0})
```

</div>

<div class="cell code" data-execution_count="8">

``` python
from sklearn.model_selection import train_test_split

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

</div>

<div class="cell markdown">

One of the advantages of RFC's is that we can be very laissez faire
about what we feed them. Unlike some of the more statistical ML
algorithms RFC's just wont use anything problematic so, besides
stripping out our target variable we will give the model our data as is.
The one thing we will need to do is encode our data numerically but
unlike many other algorithms, RFC's don't need it to be One Hot Encoded
so we shall use an Ordinal Encoder, arbitrarily assigning each unique
category to an integer for each feature. In the background I have
manually assigned class in order to ensure poisonous is given the value
of 1.

It's probably overkill for such a simple opporation but I am going to
use an SKlearn pipeline for funsies.

</div>

<div class="cell code" data-execution_count="9" data-tags="[]">

``` python
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

<div class="output execute_result" data-execution_count="9">

    Pipeline(steps=[('encoder', OrdinalEncoder()),
                    ('random_forest', RandomForestClassifier())])

</div>

</div>

<div class="cell markdown">

Now that we have fitted our model to the training set we can apply it to
the unseen test set and then do some evaluation. Lets look at the Recall
and overall Accuracy.

</div>

<div class="cell code" data-execution_count="10">

``` python
from sklearn.metrics import accuracy_score, recall_score

y_pred = pipe.predict(X_test)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Our first model achieved an accuracy of {accuracy} and a recall of {recall}')
```

<div class="output stream stdout">

    Our first model achieved an accuracy of 1.0 and a recall of 1.0

</div>

</div>

<div class="cell markdown">

... Well that went better then expected. Let's come up with a new plan\!

</div>

<div class="cell markdown">

## Less is more

Now that we know that mushrooms are very predictible (sorry
Mycologiests, looks like the machines are here for your jobs) we shall
have to find something a little bit more challenging.

Lets take a look at what our model was using.

</div>

<div class="cell code" data-execution_count="13">

``` python
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

<div class="output display_data">

![](af16cd95341ec7fbcc7724ebf04399c84e6a0625.png)

</div>

</div>

<div class="cell markdown">

As we suspected during the EDA odor is a very powerful feature. Looking
back at spore\_print\_color it's also not suprising that the model has
honed in on it. veil\_color on the other hand is suprisingly irrelivant.

We can see from the plot that there are definitly some features
(veil\_type chief among them) that we could probably do without. Lets
see what we can get away with losing. We will build a function that runs
a model, removes the least important feature and runs the model again,
continuing until it has reached some lower level of efficiancy.

</div>

<div class="cell code" data-execution_count="20">

``` python
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

</div>

<div class="cell markdown">

Our function has three exit clauses:

  - if at the very start it can't reach its target it will imediatly
    tell us.
  - Otherwise it will run untill it has no longer hits the target and
    then reports the last successful feature configuration.
  - Finally if it can hit the target with only one feature it will
    report.

</div>

<div class="cell code" data-execution_count="21">

``` python
feature_eliminator(data, 'class', pipe, min_prob=0.95)
```

<div class="output stream stdout">

    Only one feature needed to achieve 95.0% accuracy. Accuracy:98.5%, recall:96.89999999999999%

</div>

<div class="output execute_result" data-execution_count="21">

    ['odor']

</div>

</div>

<div class="cell markdown">

I knew something smelled fishy\! let's try again with some deodorized
mushrooms:

</div>

<div class="cell code" data-execution_count="22">

``` python
feature_eliminator(data, 'class', pipe, min_prob=0.95, initial_drops=['odor'])
```

<div class="output stream stdout">

    Minimum # of features with at least 95.0% accuracy: 3, accuracy:97.6%, recall:100.0%

</div>

<div class="output execute_result" data-execution_count="22">

    ['gill_size', 'ring_type', 'spore_print_color']

</div>

</div>

<div class="cell markdown">

Well... one more try with all three removed?

</div>

<div class="cell code" data-execution_count="23">

``` python
feature_eliminator(data, 'class', pipe, min_prob=0.95, initial_drops=['odor', 'gill_size', 'spore_print_color'])
```

<div class="output stream stdout">

    Minimum # of features with at least 95.0% accuracy: 3, accuracy:95.6%, recall:92.0%

</div>

<div class="output execute_result" data-execution_count="23">

    ['stalk_root', 'ring_type', 'population']

</div>

</div>

<div class="cell markdown">

That's enough of that I think\! There are obviously a multitude of very
strong predictors. We are going to have to be more creative to find a
challenge in the world of mushrooms.

</div>

<div class="cell markdown">

## Mushrooms for everyone\!

Okay, let's set out a project. We shall build a usable field guide that
is fully accessable. For this we will need:

  - A model that has no risk of poisoning its users; that is 100%
    recall.
  - The model should not use any features that might be inaccessable; in
    this case that means colors and odor.
  - The model should be as simple as possible.

Our model will be a Decision Tree; we shall try to minimise the number
of end leaves. We will use a GridsearchCV ensemble to try to find the
best parameters and try different random seeds.

Our first step will be to build a function runs our GridSearchCV with
increasing Maximum leaf values searching for one that achieves 100%
recall. It will then produce the best decision tree it found and report
its accuracy.

</div>

<div class="cell code" data-execution_count="32">

``` python
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

</div>

<div class="cell markdown">

Eagle eyed readers may have noticed something a bit naughty about our
function: We are predicting upon the entire dataset rather then just the
training set. There is a reason for this; As we are looking for 100%
recall we are protected from overfitting - if the algorithm is achieving
100% on the whole data set it must necessarily be achieving it on the
training set as well. Why doing it this way is desirable is because it
is possible for an algorithm to achieve high but not 100% recall on the
training data but then, by fluke, get 100% on the test data. This
protects us from that case.

Now, lets impliment it.

</div>

<div class="cell code" data-execution_count="33">

``` python
features = list(data.columns)
for i in ['odor', 'spore_print_color', 'veil_color', 'stalk_color_below_ring', 'stalk_color_above_ring', 'gill_color', 'cap_color', 'class']:#
    features.remove(i)

params = {'criterion':['entropy', 'gini'],
              'random_state':range(10),
              'max_features':[7,9,11,13], 
              'max_depth':[3,5,7,9,11]}

best = leaf_finder(data, 'class', features, params, min_leaf=5, max_leaf=15)
```

<div class="output stream stdout">

    Best estimator achieved accuracy of 96.5%, using 8 leaves

</div>

</div>

<div class="cell markdown">

8 leaves, not bad\! An overall accuracy of 96.5% is completly acceptable
as well. Lets take a look at the winning tree:

</div>

<div class="cell code" data-execution_count="34" data-tags="[]">

``` python
from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))
plot_tree(best, feature_names=features, class_names=['edible', 'poisonous'], filled=True)
plt.show()
```

<div class="output display_data">

![](7677162490439db542a4d6c426863ebac58b6fb3.png)

</div>

</div>

<div class="cell code" data-execution_count="72">

``` python
#hide
data['stalk_root'] = data['stalk_root'].map({'bulbous': 1, 'club': 0, 'equal': 3, 'missing': 2, 'rooted': 4})
```

</div>

<div class="cell markdown">

Can we make it any better? After a bit of experimentation I found two
stratagies. Firstly gill\_size seems to be a trap. While it is a great
initial seporator the two groups it creats seem to be hard to brake down
further. Secondly the stalk root feature is obviously a very useful one,
but the arbitraty order it has been assigned seems sub-optimal (as we
can see from the fact it has been split on three seperate times in our
tree). I remaped the values to make the splitting more efficient.

</div>

<div class="cell code" data-execution_count="73">

``` python

features = list(data.columns)
for i in ['odor', 'spore_print_color', 'veil_color', 'stalk_color_below_ring', 'stalk_color_above_ring', 'gill_color', 'cap_color', 'class', 'gill_size']:#
    features.remove(i)

params = {'criterion':['entropy', 'gini'],
              'random_state':range(10),
              'max_features':[7,9,11,13], 
              'max_depth':[3,5,7,9,11]}

best = leaf_finder(data, 'class', features, params, min_leaf=5, max_leaf=15)
```

<div class="output stream stdout">

    Best estimator achieved accuracy of 93.5%, using 6 leaves

</div>

</div>

<div class="cell code" data-execution_count="74">

``` python
from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))
plot_tree(best, feature_names=features, class_names=['edible', 'poisonous'], filled=True)
plt.show()
```

<div class="output display_data">

![](81ecb8df1783f4c7802d12b1e66002735ee056f9.png)

</div>

</div>

<div class="cell markdown">

And there we have it\! Five questions that anyone can answer and we have
sucessfully sorted out all of the inedible fungus. Trufflers rejoyce\!

</div>
