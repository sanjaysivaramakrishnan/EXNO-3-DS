## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np 
df=pd.read_csv(r"C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\data.csv")
df.head()
**ORDINAL ENCODER**
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
df.Ord_1.unique()
pm=['Cold','Warm','Hot','Very Hot']
e1 = OrdinalEncoder(categories=[pm])
df['bo1']=e1.fit_transform(df[["Ord_1"]])
df
**LABEL ENCODER**
le=LabelEncoder()
dfc=df.copy()
##TYPE YOUR CODE HERE
dfc['bo1']=le.fit_transform(df["Ord_1"])
dfc.head()
**OneHotEncoder**
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\Encoding Data.csv')
df2.head()
#TYPE YOUR DATAFRAME CODE HERE
newframe =pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
newframe
#TYPE YOUR CONCAT CODE HERE
df3 = pd.concat([df2,newframe],axis=1)
df3.head()
pd.get_dummies(df2,columns=["nom_0"])
**BinaryEncoder**
from category_encoders import BinaryEncoder
df=pd.read_csv(r"C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\data.csv")
df.head()
be=BinaryEncoder()
new_frame = be.fit_transform(df['Ord_2'])
new_frame
pd.concat([df,new_frame],axis=1)
**TargetEncoder**
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
##TYPE YOUR CODE HERE
newframe = te.fit_transform(X=cc['City'],y=cc['Target'])
pd.concat([cc,newframe],axis=1)
**FEATURE TRANSFORMATION**
from scipy import stats
df=pd.read_csv(r"C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\Data_to_Transform.csv")
df.head()
df.columns
df.skew()
#Perform Log,Reciprocal,sqrt,dquare method one by one
np.log(df['Highly Positive Skew'])
np.reciprocal(df['Highly Positive Skew'])
np.sqrt(df['Highly Positive Skew'])
np.square(df['Highly Positive Skew'])
df
#perfrom boxcox and yeojhonson method for any one column
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
df.head()
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
sm.qqplot(np.reciprocal(df['Moderate Negative Skew']),line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)
df['Moderate Negative Skew'] = qt.fit_transform(df[['Moderate Negative Skew']])
_= sm.qqplot(df['Moderate Negative Skew'],line='45')

df['Highly Negative Skew_1'] = qt.fit_transform(df[['Highly Negative Skew']])
_= sm.qqplot(df['Highly Negative Skew'], line='45')
_= sm.qqplot(df['Highly Negative Skew_1'],line='45')
dt = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\titanic_dataset.csv')
dt.head()
dt['Age_1'] = qt.fit_transform(dt[['Age']])
_= sm.qqplot(dt['Age'],line='45')
_= sm.qqplot(dt['Age_1'],line='45')

```

![image](https://github.com/user-attachments/assets/5832f4d8-fcff-49fd-84b7-eacde438b6bc)
![image](https://github.com/user-attachments/assets/e46a5f9a-3a97-4a15-8cf5-ad06394436ac)
![image](https://github.com/user-attachments/assets/d23cc413-4af3-494d-8960-87d6a8e68431)
![image](https://github.com/user-attachments/assets/a81f808d-8f0f-4504-8726-59da544dd02a)
![image](https://github.com/user-attachments/assets/bcbcb436-f657-4f54-8616-2a1e4f31b058)
![image](https://github.com/user-attachments/assets/134190aa-a768-425b-b96e-57eb8aa2b95e)
![image](https://github.com/user-attachments/assets/886ef1f1-b99f-427a-9b8f-4f590254e1a7)
![image](https://github.com/user-attachments/assets/ece50f83-dc9e-469f-80a5-c41e16f0cec4)
![image](https://github.com/user-attachments/assets/a85a7644-b2bc-467c-9011-d6e9a5f80b34)
![image](https://github.com/user-attachments/assets/4c4d37c7-ba64-4605-af14-31e5c7039641)
![image](https://github.com/user-attachments/assets/6d2b9a5c-7f80-47c7-bb17-ce4b3391ca85)
![image](https://github.com/user-attachments/assets/0e03b3a1-9091-4654-a065-3dcf16b25936)
![image](https://github.com/user-attachments/assets/4467d309-14e7-42bc-81d7-c307000513b5)
![image](https://github.com/user-attachments/assets/51f6a6f2-7cd3-49f4-b431-8af6d30135f9)
![image](https://github.com/user-attachments/assets/298bb839-8c74-4634-9d94-777ec142a55e)



# RESULT:
      Thus we perform Feature Encoding and Transformation process

       
