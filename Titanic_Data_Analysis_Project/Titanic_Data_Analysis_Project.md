
## Data Analysis Project Titanic 

In this study, I will focus my attention on the surviving rate of passengers on board the Titanic. 

I would like to understand if there was a significant difference in survived people between the one that were traveling alone on the ship and the one that were on board with other family members. To do so I will create a new variable called Alone that will indicate for 0 that the passenger was traveling alone and 1 with one or more relatives. I will make use of the variable SibSb and Parch to know if the passengers were traveling alone or not. Although we do not have information about friends or unidentified couples on board. At the begin I will focus my attention to the general case and then later I will continue this investigation for women and men separately. 

Secondly, I would like to investigate the difference in the ticket fare among the passengers that survived and the passengers that did not. I would like to consider this information as the status of the passenger in state of using class that were only 3. I could be much more helpful have the information about the salary of the passenger but we do not have it.  

Before looking for an answer to my question I will import the data using Panda functions for CSV and I will check the data imported looking for anomalies that could compromise my research. I will use the function describe looking for strange values and the function isnull to understand how many values are missing in the table.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats
%matplotlib inline
```


```python
titanic = pd.read_csv('titanic-data.csv')
```


```python
titanic.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>0</td>
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
titanic.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
sns.distplot(titanic['Age'][np.isfinite(titanic['Age'])],hist=True, rug=False, color = 'olive', axlabel = 'Fare')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of the varaibale Age among passengers',size = 13)
```




    <matplotlib.text.Text at 0x110972940>




![png](output_7_1.png)


The first thigs that I notice is that this data does not contain all the data about the passengers on board the Titanic. In fact, we have in this table 891 passengers while on board the titanic were 1316 passengers and 908 crew members. So, our study will be not exhaustive as if we were in possess of all the information.     
But in this histogram, that shows the distributions of the variable Age over passengers, we can notice a shape similar to a normal distribution, and we can suppose to have a valid sample of all the passengers on board the titanic. 
Secondly we can notice that we have some fields missing, for 177 passengers we do not have information about the age and for 687 we do not know the cabin and finally for 2 passengers we do not have the embarked port. Considered what will be the focus of this study, these missing fields will be not be an issue, they will be not be used. 
On the other hand, Survived, Sex, Fare, SibSp and Parch, the variables of interests in this study are not missing. 
Although, there is one thing that catch my attention, and it is the minimum value of Fare (0.00) and the maximum. I will therefor investigate later these values to try answering correctly at my second question.

I will now perform some modification to the table. 
I will change the value of Sex transforming it to integer (0=male, and 1=female). 
I will transform the Embarked value to an integer as well (S=1, Q=2, C=3) respecting the time frame of the embarking.  Untimely I will add a new column called 'Alone'. This will be an integer 0 or 1, calculated over the two columns SibSp and Parch (0=NotAlone, 1=Alone).  
To perform all this, I will define 3 function.


```python
def port(x):
    if x == 'S':
        x =1
    elif x == 'Q':
        x =2
    elif x == 'C':
        x =3 
    return x 

def gender(x):
    if x == 'male':
        x = 0 
    elif x == 'female':
        x = 1 
    return x

def alone(x):
    if x >= 1:
        x = 0
    else:
        x = 1
    return x 
```


```python
titanic['Embarked']= titanic['Embarked'].apply(port)
```


```python
titanic['Sex']= titanic['Sex'].apply(gender)
```


```python
titanic['Alone'] = titanic['SibSp'] + titanic['Parch']
titanic['Alone'] = titanic['Alone'].apply(alone)
```


```python
titanic.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
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
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Analysis of Alone variable 
We have now more possibilities to work with the data. 
I'm gonna plot now the distribution of passengers that were alone or not. 


```python
sns.countplot(x = 'Alone', data = titanic,palette=['Olive','DarkOrange'])
plt.xlabel('Not Alone                                         Alone')
plt.title('Count of passengers on board, travelling Not Alone and Alone ',size = 13)
```




    <matplotlib.text.Text at 0x113d52358>




![png](output_16_1.png)


We can see from the plot above that in our data we have more passengers traveling alone then passengers not. 
I will now group by the Alone variable and see if there is a significant difference in the Survived mean among these passengers.


```python
titanic.groupby('Alone')['Survived'].describe()
```




    Alone       
    0      count    354.000000
           mean       0.505650
           std        0.500676
           min        0.000000
           25%        0.000000
           50%        1.000000
           75%        1.000000
           max        1.000000
    1      count    537.000000
           mean       0.303538
           std        0.460214
           min        0.000000
           25%        0.000000
           50%        0.000000
           75%        1.000000
           max        1.000000
    Name: Survived, dtype: float64



In order to evaluate the independence of these two categorical variables, I will make use of the Chi-Square Test for Independence. The Null hypothesis is that the two variables, Alone and Survived, are independent. The alternative hypothesis is that knowing the value of the first could help predict the second. 
- H0: Alone and Survived are independent. 
- H1: Alone A and Survived are not independent.

For this test we are going to set a significant level of 0.05 (α).                   

To do so, I will create a function that could be used later on.


```python
def chitest2x2(x,y):
    a = x[x==0].count()
    b = x[x==1].count()
    c = y[y==0].count()
    d = y[y==1].count()
    Chi = (((a*d-b*c)**2)*(a+b+c+d))/((a+b)*(c+d)*(b+d)*(a+c))
    print ('The Chi Square score is:', Chi )
```

I will create 2 series for passengers on board , Alone and not Alone, with only the survived variable. 


```python
not_alone = titanic['Survived'][titanic['Alone']==0]
alone = titanic['Survived'][titanic['Alone']==1]
```


```python
chitest2x2(not_alone,alone)
```

    The Chi Square score is: 36.8501308475


In this case the P-value is the probability that a chi-square statistic having 1 degrees of freedom is more extreme than 36.85. In this case, we have P(Χ²>36.85) ≤ 0.0001. Since this value is less than the significant level (0.05), we can reject the null hypothesis, and we can suppose that there is a relationship between the two variable.

To continue in this direction, I will now divide the dataset using the variable Sex, and look if is there a difference in surviving rate when passengers where alone or not.


```python
sex_notalone = titanic[['Survived','Sex']][titanic['Alone']==0]
female_notalone = sex_notalone[sex_notalone['Sex']==1]
male_notalone = sex_notalone[sex_notalone['Sex']==0]
sex_alone = titanic[['Survived','Sex']][titanic['Alone']==1]
female_alone = sex_alone[sex_alone['Sex']==1]
male_alone = sex_alone[sex_alone['Sex']==0]
```

I will now plot the for different group to understand better how many passengers were alone or not, by their gender.


```python
g = sns.factorplot("Survived", row="Alone", col="Sex", margin_titles=True,data=titanic,kind="count",palette=['firebrick','cornflowerblue'])
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Count of passengers on board, travelling Not Alone and Alone and by Sex, Survived or not')
```




    <matplotlib.text.Text at 0x1109dea20>




![png](output_28_1.png)


First in the above plot we can see the difference among gender, eventually we can notice that the data show an opposite direction when we look at survived passengers.                    
The following data will help better understand this.


```python
print ('In our Data set : ')
print ('There were', male_notalone['Survived'].count(),'male not alone on board, only',"%.0f"%(male_notalone['Survived'][male_notalone['Survived']==1].count())
       ,'survived')
print ('There were', male_alone['Survived'].count(),'male alone on board, only',"%.0f"%(male_alone['Survived'][male_alone['Survived']==1].count())
       ,'survived')
print ('There were', female_notalone['Survived'].count(),'female not alone on board, only',"%.0f"%(female_notalone['Survived'][female_notalone['Survived']==1].count())
       ,'survived')
print ('There were', female_alone['Survived'].count(),'female alone on board, only',"%.0f"%(female_alone['Survived'][female_alone['Survived']==1].count())
       ,'survived')
```

    In our Data set : 
    There were 166 male not alone on board, only 45 survived
    There were 411 male alone on board, only 64 survived
    There were 188 female not alone on board, only 134 survived
    There were 126 female alone on board, only 99 survived


Looking at the data we can notice a difference between the two means, alone or not alone, but the result is opposite by sex.                               
The mean of survived women traveling alone is 0.786 and not alone is 0.713 while for survived men alone is 0.156 and not alone is 0.271. 

In order to evaluate the independence of these two categorical variables, I will make use of the Chi-Square Test for Independence. I will perform this test using the function defined before, and in this case for female and male separately.         
The Null hypothesis is that the two variables, Alone and Survived, are independent. The alternative hypothesis is that knowing the value of the first could help predict the second. 
H0: Alone and Survived are independent. 
H1: Alone A and Survived are not independent.
For this test we are going to set a significant level of 0.05 (α)


```python
chitest2x2(female_alone['Survived'],female_notalone['Survived'])
chitest2x2(male_alone['Survived'],male_notalone['Survived'])
```

    The Chi Square score is: 2.09723834744
    The Chi Square score is: 10.2710153203


 

In this case, we have different results: 

For female on board the titanic we have a Chi-Square score of 2.10 with a degree of freedom of 1. So our P-Value is P(Χ²>2.10) = 0.147569, but since this value is more than the significant level (0.05), we cannot reject the null hypothesis, and we can say that the two variables, Alone and Survived, for female on board the Titanic are independent. 

While for male on board the Titanic, we have a Chi-Square score of 10.27 with a degree of freedom of 1. In this case the P-Value is P(Χ²>10.27) = 0.001351 that is less than our significant level of 0.05. We can say that for Male, in this case, the two variables are not independent and we can reject the null hypothesis.

From this result, we could assume that the difference for survived variable sex is an important factor, in fact it seems so. While in general survived passengers has a mean of 0.384, the mean for survived man is 0.189 and female is 0.742, while there was much more man then women. (see cells below)     


```python
titanic.groupby('Sex')['Survived'].describe()
```




    Sex       
    0    count    577.000000
         mean       0.188908
         std        0.391775
         min        0.000000
         25%        0.000000
         50%        0.000000
         75%        0.000000
         max        1.000000
    1    count    314.000000
         mean       0.742038
         std        0.438211
         min        0.000000
         25%        0.000000
         50%        1.000000
         75%        1.000000
         max        1.000000
    Name: Survived, dtype: float64




```python
sns.countplot(x = 'Sex', data = titanic, palette=['dodgerblue','coral'])
plt.xlabel('Male                                             Female')
plt.title('Paggengers count by Sex',size = 13)
```




    <matplotlib.text.Text at 0x1144c4518>




![png](output_38_1.png)



```python
g = sns.factorplot("Survived", col="Sex", margin_titles=True,data=titanic,kind="count",palette=['firebrick','cornflowerblue'])
plt.subplots_adjust(top=0.88)
g.fig.suptitle('Passengers count, survived or not, by Sex', size = 13)
```




    <matplotlib.text.Text at 0x11450b2b0>




![png](output_39_1.png)


We can notice, thanks to the plot above, that there is an important difference in surviving between male and female. Among 577 male pasengers only 18.9 % survived, while among 314 female 74.2% survived. 
So, gender it can be considered an important factor, I will perform a chi-square Test of independence over the variables Survived as Sex at level 0.01. 
The Null hypothesis is that the two variables are independent, the alternative is that they are not independent.


```python
chitest2x2(titanic['Survived'][titanic['Sex']==0],titanic['Survived'][titanic['Sex']==1])
```

    The Chi Square score is: 263.050574071


With a Chi-Square Score of 263.05, we have a p-value is P(Χ²>263.05) ≤ 0.0001. This value is less than the significant level of 0.01 and this result allows us to do reject the null hypothesis and say that the two variables are not independent. So, we conclude that there is a relationship between, the variables Sex and Survived without considering the Alone variable analysed before.

## Alalysis of Fare variable
As said at the beginning of our study we have notice that the minimum value for fare is 0, we will go now look at this value more in detail. 


```python
titanic[titanic['Fare']==0]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
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
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>0</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>0</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0</td>
      <td>B94</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>0</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>0</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0</td>
      <td>A36</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0</td>
      <td>B102</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We have found 15 passengers where the fare is recorded 0. This could be possible for a prize for example or a gift, but it doesn't seem always the case. 4 passengers have bought 'LINE' ticket, and they were American line employees according to some research, they should have paid 7 pound each for the ticket. 8 passengers are members of the titanic guarantee group and they win the trip over 3000 employees that worked on the titanic. 
Mr Jonkheer Johan George Reuchlin traveled for free thanks to his position with the Holland America Line. 
Fry, Mr. Richard and Harrison, Mr. William were traveling with Mr Joseph Bruce Ismay as his vallet and his secretary. 
Although this sample is well redistributed over classes, just 1 over 15 survived the accident, and we will try to remove these passengers to avoid disturbance in the next study.
Another interesting value of Fare is the maximum, that we will now check before trying to answer to our question.


```python
titanic['Fare'].max()
```




    512.32920000000001



In the describe table we saw that 75% of the passenger paid less then 31 pound, this 512 could mean that there are some extra high values over fare.


```python
sns.distplot(titanic['Fare'],hist=True, rug=False, bins = 35, color = 'steelblue', axlabel = 'Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Distribution of the varaibale Fare among passengers',size = 13)
```




    <matplotlib.text.Text at 0x1147b02e8>




![png](output_48_1.png)


We can see in the histogram that there are values up to 300 and then nothing until the 500. We will now see them in details.  


```python
titanic[titanic['Fare']>=300]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
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
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>0</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Although these values seem to be correct, for the quality of the cabins reserved, I am going to try to remove these 3 passengers to permorm our study. 
I will then plot the histogramss of fare by survived or not. 


```python
tit_high_o = titanic[titanic['Fare']>0]
```


```python
tit_fare = tit_high_o[tit_high_o['Fare']<500]
```


```python
tit_fare[['Survived','Fare']].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>873.000000</td>
      <td>873.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.387171</td>
      <td>31.107631</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.487382</td>
      <td>41.331513</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>4.012500</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>14.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>31.275000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>263.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(tit_fare[tit_fare['Survived']==1]['Fare'],hist=True, rug=False, bins = 35, color = 'cornflowerblue', axlabel = 'Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Distribution of the varaibale Fare among survived passengers',size = 13)
```




    <matplotlib.text.Text at 0x114976940>




![png](output_55_1.png)



```python
sns.distplot(tit_fare[tit_fare['Survived']==0]['Fare'],hist=True, rug=False, bins = 35, color = 'darkred', axlabel = 'Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Distribution of the varaibale Fare among not survived passengers',size = 13)
```




    <matplotlib.text.Text at 0x114b49828>




![png](output_56_1.png)



```python
sns.boxplot(data = tit_fare , y = 'Fare', x = 'Survived',palette=['firebrick','cornflowerblue'])
plt.title('Boxplot of the varaible Fare between not survived and survived passengers ')
plt.xlabel('0 = Not Survived                                    1= Survived')
```




    <matplotlib.text.Text at 0x114c445f8>




![png](output_57_1.png)


Although the distributions are similar shaped, we can see more in survived passenger a higher presence in higher fares. I the box plot also is possible to notice this difference between survived and not. 
we are going use the describe function to watch at the numbers of what just observed. 


```python
tit_fare.groupby('Survived')['Fare'].describe()
```




    Survived       
    0         count    535.000000
              mean      22.696673
              std       31.589367
              min        4.012500
              25%        7.895800
              50%       10.500000
              75%       26.000000
              max      263.000000
    1         count    338.000000
              mean      44.420834
              std       50.487368
              min        6.975000
              25%       12.475000
              50%       26.000000
              75%       56.929200
              max      263.000000
    Name: Fare, dtype: float64




```python
tit_fare[['Survived','Fare']].corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>1.000000</td>
      <td>0.256172</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.256172</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the Pearson's correlation is 0.256, so the 2 variables are slightly positive correlated.      
the two means are 22.697 for not survived passengers and 44.421 for survived passengers. We are going to perform a one tail Welch's t-test, because, as we can see from the describing table just above, the two sample are of 535 and 338 respectively, and have quite different standard deviation.                
As null hypothesis, we are going to suppose that the two mean of the variable fare are equal and as alternative hypothesis that the mean of fare for survived passenger is higher than the mean of passengers that did not. 
the significant level will be 0.05. 
I will in this case a function in scipy package after have divided my dataset by the variable Survived.


```python
tit_fare_surv = tit_fare['Fare'][tit_fare['Survived']==1]
tit_fare_not_surv = tit_fare['Fare'][tit_fare['Survived']==0]
```


```python
scipy.stats.ttest_ind(tit_fare_surv,tit_fare_not_surv, axis=0, equal_var=False)
```




    Ttest_indResult(statistic=7.0831705832208023, pvalue=4.7561407741698863e-12)



The value of 7.083 as t-statistic and a p-value that is less than 0.00001, allow us to reject the null hypothesis and state that the two mean are significant different, specifically the mean of the variable Fare for survived people is higher than for not survived.

## Conclusions

The dataset used in this study is just a part of all the data available of the passengers of the Titanic. As said at the beginning of this analysis, we only have used in this study only 891 passengers of 1316 total passengers. So, we couldn't perform our analysis of all the passengers.  There were as saw before some missing values,especailly Age and Cabin,  but the variables of interest of this study were not missing at all.                
Said so, we can say that we discovered, in this study, that there is an interesting relation between being alone or not and survived ratio. Although this value that come from Parch and SibSp, do not count other kind of relations, like friendship or professional relation. It could be interesting investigate also with this kind of information. 
When we looked at Gender separately, we discover that a difference behaviour. First of all, the mean of survived male alone is less than the male not alone survived, while for female these two mean are opposite, so the mean of female alone is higher. Secondly for Male the variable Survived and alone were resulted not independent, while for female we cannot say the same. 

We can say that we discover also an significant difference between the ticket fare, among passengers that survived and not.                                    
However, it could be interesting continuing this investigation, extracting the real ticket fare per person. The fare variable that we have in this data set is the fare that has been payed for the ticket all together, so in some case this fare is the correct value when the passenger was traveling alone, otherwise is the fare for the ticket for multiple persons. We could define a new variable dividing the fare by the numbers of family members (Parch + SibSb), and this could it be the fare for each passenger. But data shows that some tickets were bought for multiple passengers that were not family related, and this would create not correct values of our dataset. With more investigation and analysis this could be avoided, using the variable ticket and to look for duplicate values and use it to find the correct fare for each passenger.

## Sources

http://stackoverflow.com/questions/11869910/pandas-filter-rows-of-dataframe-with-operator-chaining       
http://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column      
http://stackoverflow.com/questions/19410042/how-to-make-ipython-notebook-matplotlib-plot-inline         
http://stackoverflow.com/questions/33458566/how-to-choose-bins-in-matplotlib-histogram        
http://www.titanic-titanic.com                 
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html                
http://stattrek.com/chi-square-test/independence.aspx?Tutorial=AP
http://math.hws.edu/javamath/ryan/ChiSquare.html
http://www.socscistatistics.com/pvalues/chidistribution.aspx
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_ind.html
https://matplotlib.org/2.0.0/examples/color/named_colors.html
http://seaborn.pydata.org/generated/seaborn.factorplot.html
http://seaborn.pydata.org/generated/seaborn.FacetGrid.html
