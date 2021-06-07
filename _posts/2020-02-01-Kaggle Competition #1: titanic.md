
Date: 02/01/2020

Author: Byvuong

# Context
This notebook will perform an analysis of the Kaggle Titanic datasets.
The objective is to find a classification model to predict if a person will survived or not. The data dictionary can be found here: [Kaggle - Titanic](https://www.kaggle.com/c/titanic/data). I'll be using Google colab and will explain further along how to setup the workspace.




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import urllib

%matplotlib inline
%reload_ext autoreload
%autoreload 2
np.random.seed(42)
```

# Kaggle setup for Google colab

This will mount Google drive to Google Colab, you'll gain access to any files stored in your drive. Copy paste the authorization code from the propose URL and press enter to setup.


```python
!pip install kaggle
!pip install -q kaggle-cli
from google.colab import drive
drive.mount('/content/gdrive')
```

In Google drive, I've created a few folders to store the kaggle key and dataset. The data will be stored in this directory 'v_kaggle_competitions/titanic'. The kaggle environnement path to load the kaggle key is stored in 'v_kaggle_competitions/.kaggle'.

The process to generate the kaggle access key, it's by going to your kaggle account, in settings and in the api section where you can create the token. For more infos on how to setup go to [here](https://www.kaggle.com/docs/api).


```python
# Code reference to create the hidden folder .kaggle.

# !pwd
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 /.kaggle/kaggle.json
```


```python
import os

os.chdir('/content/gdrive/My Drive/v_kaggle_competitions/titanic')
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/v_kaggle_competitions/.kaggle"
```

To consult any available competitions datasets enter the following code:


```python
!kaggle competitions list
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.6 / client 1.5.4)
    ref                                            deadline             category             reward  teamCount  userHasEntered  
    ---------------------------------------------  -------------------  ---------------  ----------  ---------  --------------  
    digit-recognizer                               2030-01-01 00:00:00  Getting Started   Knowledge       2375           False  
    titanic                                        2030-01-01 00:00:00  Getting Started   Knowledge      16059            True  
    house-prices-advanced-regression-techniques    2030-01-01 00:00:00  Getting Started   Knowledge       5099           False  
    connectx                                       2030-01-01 00:00:00  Getting Started   Knowledge        374           False  
    imagenet-object-localization-challenge         2029-12-31 07:00:00  Research          Knowledge         63           False  
    competitive-data-science-predict-future-sales  2020-12-31 23:59:00  Playground            Kudos       5603           False  
    deepfake-detection-challenge                   2020-03-31 23:59:00  Featured         $1,000,000       1423           False  
    cat-in-the-dat-ii                              2020-03-31 23:59:00  Playground             Swag        437           False  
    nlp-getting-started                            2020-03-23 23:59:00  Getting Started     $10,000       2278           False  
    bengaliai-cv19                                 2020-03-16 23:59:00  Research            $10,000        963           False  
    google-quest-challenge                         2020-02-10 23:59:00  Featured            $25,000       1402           False  
    tensorflow2-question-answering                 2020-01-22 23:59:00  Featured            $50,000       1233           False  
    data-science-bowl-2019                         2020-01-22 23:59:00  Featured           $160,000       3497           False  
    pku-autonomous-driving                         2020-01-21 23:59:00  Featured            $25,000        866           False  
    santa-2019-revenge-of-the-accountants          2020-01-16 23:59:00  Playground             Swag        106           False  
    santa-workshop-tour-2019                       2020-01-15 23:59:00  Featured            $25,000       1620           False  
    nfl-big-data-bowl-2020                         2020-01-06 23:59:00  Featured            $75,000       2038           False  
    nfl-playing-surface-analytics                  2020-01-02 23:59:00  Analytics           $75,000          0           False  
    ashrae-energy-prediction                       2019-12-19 23:59:00  Featured            $25,000       3614           False  
    Kannada-MNIST                                  2019-12-17 23:59:00  Playground        Knowledge       1214           False  


Let's download the Titanic datasets. Afterward refresh the file pane which is located on the left side of the notebook. You will be able to locate the files in '/content/gdrive/My Drive/v_kaggle_competitions/titanic'. The folder will contain two files: 'test.csv' and 'train.csv'.


```python
!kaggle competitions download -c titanic
```

# Data splitting

We will split the datasets in three parts: 60% training, 20% validation, 20% test.


```python
# to review and remove ***

from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=0.4, random_state=0)
X_validation, X_test, y_validation, y_test = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)
```


```python
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('train.csv', header=0, sep=',', quotechar='"')
df_test = pd.read_csv('test.csv', header=0, sep=',', quotechar='"')
df_x_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']
```


```python
df_test
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>3</td>
      <td>Spector, Mr. Woolf</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>A.5. 3236</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17758</td>
      <td>108.9000</td>
      <td>C105</td>
      <td>C</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>3</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>3</td>
      <td>Ware, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>359309</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 11 columns</p>
</div>



# Data exploration

For exploration, we will use the variable 'df_expl' and make a deep copy of the training set. Any manipulation from 'df_expl' won't affect the training set.


```python
df_expl = df_x_train.copy()
```


```python
df_expl.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_expl.head(10)
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_expl.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 11 columns):
    PassengerId    891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 76.7+ KB


Respectively the columns 'Age', 'Cabin' and 'Embarked' have 177, 687 and 3 missing values.

Overview of the variables:
- Nominals: 'Pclass', 'Sex', 'Embarked'
- Numericals: 'Age', 'SibSp', 'Parch', 'Fare'
- Strings: 'Name', 'Ticket', 'Cabin'


```python
# To print out the classes for nominal variable.

cols_cat = [ 'Pclass', 'Sex', 'Embarked']
for _ in cols_cat:
  print('-' * 40)
  print(f'{_} : {df_expl[_].value_counts()}')
```

    ----------------------------------------
    Pclass : 3    491
    1    216
    2    184
    Name: Pclass, dtype: int64
    ----------------------------------------
    Sex : male      577
    female    314
    Name: Sex, dtype: int64
    ----------------------------------------
    Embarked : S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64


The 'Name' column is non atomic and contains multiple informations: 
- lastname 
- title, 
- firstname, 
- and a relative in parenthesis.

More analysis have to be done before concluding those are the assign descriptions.

e.g: 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)'


```python
# https://www.kaggle.com/c/titanic/data
# sibsp refers to: # of siblings / spouses aboard the Titanic	
# parch	referts to: # of parents / children aboard the Titanic

pd.set_option('display.max_colwidth', -1)
cols_expl_title = ['Name', 'Age', 'SibSp', 'Parch']
df_expl[cols_expl_title].head(20)
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
      <th>Name</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Allen, Mr. William Henry</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moran, Mr. James</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>McCarthy, Mr. Timothy J</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Saundercock, Mr. William Henry</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Andersson, Mr. Anders Johan</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Vestrom, Miss. Hulda Amanda Adolfina</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Hewlett, Mrs. (Mary D Kingcome)</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Rice, Master. Eugene</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Williams, Mr. Charles Eugene</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Masselmani, Mrs. Fatima</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titles = [i.split(',')[1].split('.')[0].strip() for i in df_expl['Name']]
```


```python
df_expl['Title'] = pd.Series(titles)
df_expl['Title'].value_counts()
```




    Mr              517
    Miss            182
    Mrs             125
    Master          40 
    Dr              7  
    Rev             6  
    Major           2  
    Col             2  
    Mlle            2  
    Sir             1  
    Capt            1  
    Don             1  
    Ms              1  
    Jonkheer        1  
    Mme             1  
    the Countess    1  
    Lady            1  
    Name: Title, dtype: int64



Due to the small size for multiple titles, we will group the titles below 40 together as rare. 


```python
titles_common = ['Mr', 'Miss', 'Mrs', 'Master']
df_expl.loc[~df_expl['Title'].isin(titles_common)]
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>1</td>
      <td>Uruchurtu, Don. Manuel E</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17601</td>
      <td>27.7208</td>
      <td>NaN</td>
      <td>C</td>
      <td>Don</td>
    </tr>
    <tr>
      <th>149</th>
      <td>150</td>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
      <td>male</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>244310</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Rev</td>
    </tr>
    <tr>
      <th>150</th>
      <td>151</td>
      <td>2</td>
      <td>Bateman, Rev. Robert James</td>
      <td>male</td>
      <td>51.0</td>
      <td>0</td>
      <td>0</td>
      <td>S.O.P. 1166</td>
      <td>12.5250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Rev</td>
    </tr>
    <tr>
      <th>245</th>
      <td>246</td>
      <td>1</td>
      <td>Minahan, Dr. William Edward</td>
      <td>male</td>
      <td>44.0</td>
      <td>2</td>
      <td>0</td>
      <td>19928</td>
      <td>90.0000</td>
      <td>C78</td>
      <td>Q</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>249</th>
      <td>250</td>
      <td>2</td>
      <td>Carter, Rev. Ernest Courtenay</td>
      <td>male</td>
      <td>54.0</td>
      <td>1</td>
      <td>0</td>
      <td>244252</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Rev</td>
    </tr>
    <tr>
      <th>317</th>
      <td>318</td>
      <td>2</td>
      <td>Moraweck, Dr. Ernest</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>29011</td>
      <td>14.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>369</th>
      <td>370</td>
      <td>1</td>
      <td>Aubart, Mme. Leontine Pauline</td>
      <td>female</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17477</td>
      <td>69.3000</td>
      <td>B35</td>
      <td>C</td>
      <td>Mme</td>
    </tr>
    <tr>
      <th>398</th>
      <td>399</td>
      <td>2</td>
      <td>Pain, Dr. Alfred</td>
      <td>male</td>
      <td>23.0</td>
      <td>0</td>
      <td>0</td>
      <td>244278</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>443</th>
      <td>444</td>
      <td>2</td>
      <td>Reynaldo, Ms. Encarnacion</td>
      <td>female</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>230434</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Ms</td>
    </tr>
    <tr>
      <th>449</th>
      <td>450</td>
      <td>1</td>
      <td>Peuchen, Major. Arthur Godfrey</td>
      <td>male</td>
      <td>52.0</td>
      <td>0</td>
      <td>0</td>
      <td>113786</td>
      <td>30.5000</td>
      <td>C104</td>
      <td>S</td>
      <td>Major</td>
    </tr>
    <tr>
      <th>536</th>
      <td>537</td>
      <td>1</td>
      <td>Butt, Major. Archibald Willingham</td>
      <td>male</td>
      <td>45.0</td>
      <td>0</td>
      <td>0</td>
      <td>113050</td>
      <td>26.5500</td>
      <td>B38</td>
      <td>S</td>
      <td>Major</td>
    </tr>
    <tr>
      <th>556</th>
      <td>557</td>
      <td>1</td>
      <td>Duff Gordon, Lady. (Lucille Christiana Sutherland) ("Mrs Morgan")</td>
      <td>female</td>
      <td>48.0</td>
      <td>1</td>
      <td>0</td>
      <td>11755</td>
      <td>39.6000</td>
      <td>A16</td>
      <td>C</td>
      <td>Lady</td>
    </tr>
    <tr>
      <th>599</th>
      <td>600</td>
      <td>1</td>
      <td>Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")</td>
      <td>male</td>
      <td>49.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17485</td>
      <td>56.9292</td>
      <td>A20</td>
      <td>C</td>
      <td>Sir</td>
    </tr>
    <tr>
      <th>626</th>
      <td>627</td>
      <td>2</td>
      <td>Kirkland, Rev. Charles Leonard</td>
      <td>male</td>
      <td>57.0</td>
      <td>0</td>
      <td>0</td>
      <td>219533</td>
      <td>12.3500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>Rev</td>
    </tr>
    <tr>
      <th>632</th>
      <td>633</td>
      <td>1</td>
      <td>Stahelin-Maeglin, Dr. Max</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>13214</td>
      <td>30.5000</td>
      <td>B50</td>
      <td>C</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>641</th>
      <td>642</td>
      <td>1</td>
      <td>Sagesser, Mlle. Emma</td>
      <td>female</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17477</td>
      <td>69.3000</td>
      <td>B35</td>
      <td>C</td>
      <td>Mlle</td>
    </tr>
    <tr>
      <th>647</th>
      <td>648</td>
      <td>1</td>
      <td>Simonius-Blumer, Col. Oberst Alfons</td>
      <td>male</td>
      <td>56.0</td>
      <td>0</td>
      <td>0</td>
      <td>13213</td>
      <td>35.5000</td>
      <td>A26</td>
      <td>C</td>
      <td>Col</td>
    </tr>
    <tr>
      <th>660</th>
      <td>661</td>
      <td>1</td>
      <td>Frauenthal, Dr. Henry William</td>
      <td>male</td>
      <td>50.0</td>
      <td>2</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.6500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>694</th>
      <td>695</td>
      <td>1</td>
      <td>Weir, Col. John</td>
      <td>male</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>113800</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Col</td>
    </tr>
    <tr>
      <th>710</th>
      <td>711</td>
      <td>1</td>
      <td>Mayne, Mlle. Berthe Antonine ("Mrs de Villiers")</td>
      <td>female</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17482</td>
      <td>49.5042</td>
      <td>C90</td>
      <td>C</td>
      <td>Mlle</td>
    </tr>
    <tr>
      <th>745</th>
      <td>746</td>
      <td>1</td>
      <td>Crosby, Capt. Edward Gifford</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>WE/P 5735</td>
      <td>71.0000</td>
      <td>B22</td>
      <td>S</td>
      <td>Capt</td>
    </tr>
    <tr>
      <th>759</th>
      <td>760</td>
      <td>1</td>
      <td>Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)</td>
      <td>female</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>B77</td>
      <td>S</td>
      <td>the Countess</td>
    </tr>
    <tr>
      <th>766</th>
      <td>767</td>
      <td>1</td>
      <td>Brewe, Dr. Arthur Jackson</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112379</td>
      <td>39.6000</td>
      <td>NaN</td>
      <td>C</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>796</th>
      <td>797</td>
      <td>1</td>
      <td>Leader, Dr. Alice (Farnham)</td>
      <td>female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>17465</td>
      <td>25.9292</td>
      <td>D17</td>
      <td>S</td>
      <td>Dr</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Jonkheer</td>
    </tr>
    <tr>
      <th>848</th>
      <td>849</td>
      <td>2</td>
      <td>Harper, Rev. John</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>1</td>
      <td>248727</td>
      <td>33.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Rev</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Rev</td>
    </tr>
  </tbody>
</table>
</div>




```python
# uncommon title will be assign to the value 'rare'

df_expl.loc[~df_expl['Title'].isin(titles_common), 'Title'] = 'rare'
df_expl['Title'].value_counts()
```




    Mr        517
    Miss      182
    Mrs       125
    Master    40 
    rare      27 
    Name: Title, dtype: int64




```python
df_expl.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>



# Further investigation on the classes Ticket and Fare relationship


```python
df_expl[df_expl['Ticket']=='PC 17611']
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>334</th>
      <td>335</td>
      <td>1</td>
      <td>Frauenthal, Mrs. Henry William (Clara Heinsheimer)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.65</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>660</th>
      <td>661</td>
      <td>1</td>
      <td>Frauenthal, Dr. Henry William</td>
      <td>male</td>
      <td>50.0</td>
      <td>2</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.65</td>
      <td>NaN</td>
      <td>S</td>
      <td>rare</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_expl[df_expl['Fare']>130].sort_values(by='Fare')
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>660</th>
      <td>661</td>
      <td>1</td>
      <td>Frauenthal, Dr. Henry William</td>
      <td>male</td>
      <td>50.00</td>
      <td>2</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.6500</td>
      <td>NaN</td>
      <td>S</td>
      <td>rare</td>
    </tr>
    <tr>
      <th>334</th>
      <td>335</td>
      <td>1</td>
      <td>Frauenthal, Mrs. Henry William (Clara Heinsheimer)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.6500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>337</th>
      <td>338</td>
      <td>1</td>
      <td>Burns, Miss. Elizabeth Margaret</td>
      <td>female</td>
      <td>41.00</td>
      <td>0</td>
      <td>0</td>
      <td>16966</td>
      <td>134.5000</td>
      <td>E40</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>319</th>
      <td>320</td>
      <td>1</td>
      <td>Spedden, Mrs. Frederic Oakley (Margaretta Corning Stone)</td>
      <td>female</td>
      <td>40.00</td>
      <td>1</td>
      <td>1</td>
      <td>16966</td>
      <td>134.5000</td>
      <td>E34</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>269</th>
      <td>270</td>
      <td>1</td>
      <td>Bissette, Miss. Amelia</td>
      <td>female</td>
      <td>35.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17760</td>
      <td>135.6333</td>
      <td>C99</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>373</th>
      <td>374</td>
      <td>1</td>
      <td>Ringhini, Mr. Sante</td>
      <td>male</td>
      <td>22.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17760</td>
      <td>135.6333</td>
      <td>NaN</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>325</th>
      <td>326</td>
      <td>1</td>
      <td>Young, Miss. Marie Grice</td>
      <td>female</td>
      <td>36.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17760</td>
      <td>135.6333</td>
      <td>C32</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>1</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B78</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>1</td>
      <td>Lurette, Miss. Elise</td>
      <td>female</td>
      <td>58.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B80</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>708</th>
      <td>709</td>
      <td>1</td>
      <td>Cleaver, Miss. Alice</td>
      <td>female</td>
      <td>22.00</td>
      <td>0</td>
      <td>0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>297</th>
      <td>298</td>
      <td>1</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.00</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>498</th>
      <td>499</td>
      <td>1</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.00</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>305</th>
      <td>306</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.92</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>Master</td>
    </tr>
    <tr>
      <th>332</th>
      <td>333</td>
      <td>1</td>
      <td>Graham, Mr. George Edward</td>
      <td>male</td>
      <td>38.00</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17582</td>
      <td>153.4625</td>
      <td>C91</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>609</th>
      <td>610</td>
      <td>1</td>
      <td>Shutes, Miss. Elizabeth W</td>
      <td>female</td>
      <td>40.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17582</td>
      <td>153.4625</td>
      <td>C125</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>268</th>
      <td>269</td>
      <td>1</td>
      <td>Graham, Mrs. William Thompson (Edith Junkins)</td>
      <td>female</td>
      <td>58.00</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17582</td>
      <td>153.4625</td>
      <td>C125</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>856</th>
      <td>857</td>
      <td>1</td>
      <td>Wick, Mrs. George Dennick (Mary Hitchcock)</td>
      <td>female</td>
      <td>45.00</td>
      <td>1</td>
      <td>1</td>
      <td>36928</td>
      <td>164.8667</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>318</th>
      <td>319</td>
      <td>1</td>
      <td>Wick, Miss. Mary Natalie</td>
      <td>female</td>
      <td>31.00</td>
      <td>0</td>
      <td>2</td>
      <td>36928</td>
      <td>164.8667</td>
      <td>C7</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>779</th>
      <td>780</td>
      <td>1</td>
      <td>Robert, Mrs. Edward Scott (Elisabeth Walton McMillan)</td>
      <td>female</td>
      <td>43.00</td>
      <td>0</td>
      <td>1</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B3</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>730</th>
      <td>731</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.00</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>689</th>
      <td>690</td>
      <td>1</td>
      <td>Madill, Miss. Georgette Alexandra</td>
      <td>female</td>
      <td>15.00</td>
      <td>0</td>
      <td>1</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>377</th>
      <td>378</td>
      <td>1</td>
      <td>Widener, Mr. Harry Elkins</td>
      <td>male</td>
      <td>27.00</td>
      <td>0</td>
      <td>2</td>
      <td>113503</td>
      <td>211.5000</td>
      <td>C82</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>527</th>
      <td>528</td>
      <td>1</td>
      <td>Farthing, Mr. John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17483</td>
      <td>221.7792</td>
      <td>C95</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>380</th>
      <td>381</td>
      <td>1</td>
      <td>Bidois, Miss. Rosalie</td>
      <td>female</td>
      <td>42.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>NaN</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>716</th>
      <td>717</td>
      <td>1</td>
      <td>Endres, Miss. Caroline Louise</td>
      <td>female</td>
      <td>38.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>C45</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>557</th>
      <td>558</td>
      <td>1</td>
      <td>Robbins, Mr. Victor</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>NaN</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>700</th>
      <td>701</td>
      <td>1</td>
      <td>Astor, Mrs. John Jacob (Madeleine Talmadge Force)</td>
      <td>female</td>
      <td>18.00</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>C62 C64</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>299</th>
      <td>300</td>
      <td>1</td>
      <td>Baxter, Mrs. James (Helene DeLaudeniere Chaput)</td>
      <td>female</td>
      <td>50.00</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>118</th>
      <td>119</td>
      <td>1</td>
      <td>Baxter, Mr. Quigg Edmond</td>
      <td>male</td>
      <td>24.00</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>311</th>
      <td>312</td>
      <td>1</td>
      <td>Ryerson, Miss. Emily Borie</td>
      <td>female</td>
      <td>18.00</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>742</th>
      <td>743</td>
      <td>1</td>
      <td>Ryerson, Miss. Susan Parker "Suzette"</td>
      <td>female</td>
      <td>21.00</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.00</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>438</th>
      <td>439</td>
      <td>1</td>
      <td>Fortune, Mr. Mark</td>
      <td>male</td>
      <td>64.00</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.00</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>341</th>
      <td>342</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.00</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.00</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.00</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sibsp refers to: # of siblings / spouses aboard the Titanic	
# parch	referts to: # of parents / children aboard the Titanic

df_expl[df_expl['Ticket']=='PC 17757']
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>380</th>
      <td>381</td>
      <td>1</td>
      <td>Bidois, Miss. Rosalie</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.525</td>
      <td>NaN</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>557</th>
      <td>558</td>
      <td>1</td>
      <td>Robbins, Mr. Victor</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.525</td>
      <td>NaN</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>700</th>
      <td>701</td>
      <td>1</td>
      <td>Astor, Mrs. John Jacob (Madeleine Talmadge Force)</td>
      <td>female</td>
      <td>18.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.525</td>
      <td>C62 C64</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>716</th>
      <td>717</td>
      <td>1</td>
      <td>Endres, Miss. Caroline Louise</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.525</td>
      <td>C45</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_expl[df_expl['Ticket']=='PC 17582']
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>268</th>
      <td>269</td>
      <td>1</td>
      <td>Graham, Mrs. William Thompson (Edith Junkins)</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17582</td>
      <td>153.4625</td>
      <td>C125</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>332</th>
      <td>333</td>
      <td>1</td>
      <td>Graham, Mr. George Edward</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17582</td>
      <td>153.4625</td>
      <td>C91</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>609</th>
      <td>610</td>
      <td>1</td>
      <td>Shutes, Miss. Elizabeth W</td>
      <td>female</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17582</td>
      <td>153.4625</td>
      <td>C125</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_expl[df_expl['Ticket']=='PC 17611']
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>334</th>
      <td>335</td>
      <td>1</td>
      <td>Frauenthal, Mrs. Henry William (Clara Heinsheimer)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.65</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>660</th>
      <td>661</td>
      <td>1</td>
      <td>Frauenthal, Dr. Henry William</td>
      <td>male</td>
      <td>50.0</td>
      <td>2</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.65</td>
      <td>NaN</td>
      <td>S</td>
      <td>rare</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_expl[df_expl['Name'].str.contains('Dennick')]

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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>856</th>
      <td>857</td>
      <td>1</td>
      <td>Wick, Mrs. George Dennick (Mary Hitchcock)</td>
      <td>female</td>
      <td>45.0</td>
      <td>1</td>
      <td>1</td>
      <td>36928</td>
      <td>164.8667</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
  </tbody>
</table>
</div>



Here are some remarks for further exploration we can do:
- same tickets possess the same embarked value
- the 'Fare' value for the same tickets is a sum between them
- in the Name column, for a female the parenthesis is their name and before that is their husband name


```python
df_expl['Title'].value_counts()
```




    Mr        517
    Miss      182
    Mrs       125
    Master    40 
    rare      27 
    Name: Title, dtype: int64




```python
df_expl.groupby(['Sex','Title']).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="8" halign="left">PassengerId</th>
      <th colspan="8" halign="left">Pclass</th>
      <th colspan="8" halign="left">Age</th>
      <th colspan="8" halign="left">SibSp</th>
      <th colspan="8" halign="left">Parch</th>
      <th colspan="8" halign="left">Fare</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th>Title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">female</th>
      <th>Miss</th>
      <td>182.0</td>
      <td>408.884615</td>
      <td>246.775812</td>
      <td>3.0</td>
      <td>213.00</td>
      <td>381.5</td>
      <td>612.25</td>
      <td>889.0</td>
      <td>182.0</td>
      <td>2.307692</td>
      <td>0.849989</td>
      <td>1.0</td>
      <td>1.25</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>146.0</td>
      <td>21.773973</td>
      <td>12.990292</td>
      <td>0.75</td>
      <td>14.125</td>
      <td>21.0</td>
      <td>30.0</td>
      <td>63.0</td>
      <td>182.0</td>
      <td>0.714286</td>
      <td>1.431961</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>8.0</td>
      <td>182.0</td>
      <td>0.549451</td>
      <td>0.804184</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>182.0</td>
      <td>43.797873</td>
      <td>66.027199</td>
      <td>6.7500</td>
      <td>7.95105</td>
      <td>15.62085</td>
      <td>41.034400</td>
      <td>512.3292</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125.0</td>
      <td>453.160000</td>
      <td>270.762764</td>
      <td>2.0</td>
      <td>255.00</td>
      <td>438.0</td>
      <td>679.00</td>
      <td>886.0</td>
      <td>125.0</td>
      <td>2.000000</td>
      <td>0.823055</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>108.0</td>
      <td>35.898148</td>
      <td>11.433628</td>
      <td>14.00</td>
      <td>27.750</td>
      <td>35.0</td>
      <td>44.0</td>
      <td>63.0</td>
      <td>125.0</td>
      <td>0.696000</td>
      <td>0.598708</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
      <td>125.0</td>
      <td>0.832000</td>
      <td>1.274666</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>125.0</td>
      <td>45.138533</td>
      <td>45.723716</td>
      <td>7.2250</td>
      <td>15.85000</td>
      <td>26.00000</td>
      <td>57.000000</td>
      <td>247.5208</td>
    </tr>
    <tr>
      <th>rare</th>
      <td>7.0</td>
      <td>611.571429</td>
      <td>161.576460</td>
      <td>370.0</td>
      <td>500.50</td>
      <td>642.0</td>
      <td>735.50</td>
      <td>797.0</td>
      <td>7.0</td>
      <td>1.142857</td>
      <td>0.377964</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>32.857143</td>
      <td>11.171818</td>
      <td>24.00</td>
      <td>24.000</td>
      <td>28.0</td>
      <td>40.5</td>
      <td>49.0</td>
      <td>7.0</td>
      <td>0.142857</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>50.447629</td>
      <td>26.244481</td>
      <td>13.0000</td>
      <td>32.76460</td>
      <td>49.50420</td>
      <td>69.300000</td>
      <td>86.5000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">male</th>
      <th>Master</th>
      <td>40.0</td>
      <td>414.975000</td>
      <td>301.717518</td>
      <td>8.0</td>
      <td>165.75</td>
      <td>345.0</td>
      <td>764.00</td>
      <td>870.0</td>
      <td>40.0</td>
      <td>2.625000</td>
      <td>0.627878</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>36.0</td>
      <td>4.574167</td>
      <td>3.619872</td>
      <td>0.42</td>
      <td>1.000</td>
      <td>3.5</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>40.0</td>
      <td>2.300000</td>
      <td>1.910833</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.00</td>
      <td>8.0</td>
      <td>40.0</td>
      <td>1.375000</td>
      <td>0.540062</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>34.703125</td>
      <td>28.051752</td>
      <td>8.5167</td>
      <td>18.75000</td>
      <td>29.06250</td>
      <td>39.171875</td>
      <td>151.5500</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>517.0</td>
      <td>454.499033</td>
      <td>253.715526</td>
      <td>1.0</td>
      <td>226.00</td>
      <td>466.0</td>
      <td>674.00</td>
      <td>891.0</td>
      <td>517.0</td>
      <td>2.410058</td>
      <td>0.810622</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>398.0</td>
      <td>32.368090</td>
      <td>12.708793</td>
      <td>11.00</td>
      <td>23.000</td>
      <td>30.0</td>
      <td>39.0</td>
      <td>80.0</td>
      <td>517.0</td>
      <td>0.288201</td>
      <td>0.821298</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
      <td>517.0</td>
      <td>0.152805</td>
      <td>0.533615</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>517.0</td>
      <td>24.441560</td>
      <td>44.378561</td>
      <td>0.0000</td>
      <td>7.80000</td>
      <td>9.35000</td>
      <td>26.000000</td>
      <td>512.3292</td>
    </tr>
    <tr>
      <th>rare</th>
      <td>20.0</td>
      <td>523.400000</td>
      <td>258.018033</td>
      <td>31.0</td>
      <td>301.00</td>
      <td>613.5</td>
      <td>707.75</td>
      <td>887.0</td>
      <td>20.0</td>
      <td>1.400000</td>
      <td>0.502625</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>19.0</td>
      <td>45.894737</td>
      <td>12.332859</td>
      <td>23.00</td>
      <td>39.000</td>
      <td>49.0</td>
      <td>54.0</td>
      <td>70.0</td>
      <td>20.0</td>
      <td>0.350000</td>
      <td>0.670820</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.100000</td>
      <td>0.307794</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>35.143750</td>
      <td>31.729447</td>
      <td>0.0000</td>
      <td>13.00000</td>
      <td>27.13540</td>
      <td>36.525000</td>
      <td>133.6500</td>
    </tr>
  </tbody>
</table>
</div>



# Preprocessing pipeline

Respectively the columns 'Age', 'Cabin' and 'Embarked' have 177, 687 and 3 missing values.

we will use min-max scaling to scale the numerical value between 0 and 1


```python
# X_validation, X_test, y_validation, y_test 
# ColumnTransformer mueller chapter 4

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names):
        self.columns_names = columns_names
    def fit(self, X, y=None):
    # y=None specifies we don't want to affect the label set
        return self
    def transform(self, X):
        return X[self.columns_names]

class MostFrequentValue(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


num_pipeline = Pipeline([
        ("select_numeric", SelectColumns(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])

cat_pipeline = Pipeline([
        ("select_cat", SelectColumns(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentValue()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),   
    ])
```

# Modeling

## svm


```python
df_expl.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = preprocess_pipeline.fit_transform(df_x_train)
X_train
```




    array([[0.27117366, 0.125     , 0.        , ..., 0.        , 0.        ,
            1.        ],
           [0.4722292 , 0.125     , 0.        , ..., 1.        , 0.        ,
            0.        ],
           [0.32143755, 0.        , 0.        , ..., 0.        , 0.        ,
            1.        ],
           ...,
           [0.34656949, 0.125     , 0.33333333, ..., 0.        , 0.        ,
            1.        ],
           [0.32143755, 0.        , 0.        , ..., 1.        , 0.        ,
            0.        ],
           [0.39683338, 0.        , 0.        , ..., 0.        , 1.        ,
            0.        ]])




```python
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
```




    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)




```python
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
```




    0.788976279650437



## random forest


```python
from sklearn.ensemble import RandomForestClassifier

# n_estimators refers to the number of trees in the forest.
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```




    0.8126466916354558




```python
forest_clf
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor


param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)
```




    RandomizedSearchCV(cv=5, error_score=nan,
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       ccp_alpha=0.0,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       max_samples=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100,
                                                       n_jobs=None, oob_score=Fals...
                                                       warm_start=False),
                       iid='deprecated', n_iter=10, n_jobs=None,
                       param_distributions={'max_features': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7f95103794a8>,
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7f951038dac8>},
                       pre_dispatch='2*n_jobs', random_state=42, refit=True,
                       return_train_score=False, scoring='neg_mean_squared_error',
                       verbose=0)




```python
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```


```python
rnd_search.best_params_
```




    {'max_features': 7, 'n_estimators': 122}




```python
# n_estimators refers to the number of trees in the forest.
forest_clf = RandomForestClassifier(n_estimators=122, max_features=7, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```




    0.8227465667915107



# Prediction on test data


```python
X_test = preprocess_pipeline.fit_transform(df_test)
forest_clf.fit(X_train, y_train)
predictions = forest_clf.predict(X_test)
```

# Create csv to upload to Kaggle


```python
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':predictions})
submission.head()
```


```python
filename = 'Titanic Predictions 1.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
```

Your First Entry Welcome to the leaderboard! Your score represents your submission's accuracy. For example, a score of 0.7 in this competition indicates you predicted Titanic survival correctly for 70% of people.
