#!/usr/bin/env python
# coding: utf-8

# ### Programming Assignment 1 Q2
# ### Bayesian (Naïve Bayes) Classifier
# 
# Group Number : 46
# 
# Student 1 : Aryan Singh 19CS30007
# 
# Student 2 : Seemant Guruprasad Achari 19CS30057
# 

# ### Tools

# In[1]:


import numpy as np
import pandas as pd

# for pretty printing summary
import json 
     

import matplotlib.pyplot as plt


# In[2]:


# A constant to reference the result column
LABEL = 'Response'
# max unique categories to consider
MAX_UNIQUE_CATEGORIES = 20

# tag for categorical attributes
CATEGORICAL_FEATURE = "categorical"

# tag for continuous attributes
CONTINUOUS_FEATURE = "continuous"


# In[3]:


# splitting dataset index maintaing the ratio of true and false examples
def train_test_split(df,train_sample=0.5,target_col= LABEL):
    all_indexes=[]
    df_grouped= df.groupby(target_col)
    for x,x_df in df_grouped:
        t = x_df.sample(frac=train_sample,).index
        all_indexes.append(t)
    g = all_indexes[0].values
    for k in all_indexes[1:]:
        g=np.hstack([g,k.values])
        #np.hstack([all_indexes[0].values,all_indexes[1].values])

    train_df = df[df.index.isin(g) ]
    test_df = df[~df.index.isin(g)]
    return train_df, test_df 


# In[5]:


# Data read using pandas
df = pd.read_csv('./Dataset_C.csv')
df = df.drop(["id"], axis = 1) # since id doesn't give any information
df.info()


# In[6]:


# Classifying the attributes as continuous or categorical feature
categoricalStringCols = []
print("Analzing dataset")
for k in df.columns:
    if k == 'id':
        continue
    if k == LABEL :
        print("Label     => ", end = "")
    else:
        print("Attribute => ", end = "")
        if(len(df[k].unique()) <= MAX_UNIQUE_CATEGORIES and type(df[k][0]) == str):
            categoricalStringCols.append(k)
    print(f"{k} has {len(df[k].unique())} unique values.", end="")
    if(len(df[k].unique()) <= MAX_UNIQUE_CATEGORIES):
        print("[ ", end = "")
        for val in df[k].unique():
            print(val, end = ", ")
        print("]", end="")
    print()
print(f"Categorical string features are {categoricalStringCols}")
print()


# In[7]:


print("---------------------Task-1-Started----------------------------")
print("Encoding of categorical feature started")
vehicle_age_dict = {
    '< 1 Year' : 0,
    '1-2 Year' : 1,
    '> 2 Years': 2
}
df["Vehicle_Age_ordinal"] = df.Vehicle_Age.map(vehicle_age_dict)

gender_dict = {
    'Male': 0,
    'Female': 1
}
df["Gender_encoded"] = df.Gender.map(gender_dict)

vehicle_damage_dict = {
    'No' : 0,
    'Yes' : 1
}

df["Vehicle_Damage_encoded"] = df.Vehicle_Damage.map(vehicle_damage_dict)
print("Encoding of categorical feature finished")


# In[8]:


df.info()


# In[9]:


# Dropping old non-encoded columns

df = df.drop(['Gender', 'Vehicle_Age', 'Vehicle_Damage'], axis = 1)
print(f"Replaced the {categoricalStringCols} columns with their encoded versions")


# In[10]:


print(df.head())


# In[11]:


def determineTypeOfFeature(df):
    print("Determining type of feature")
    feature_types = {}
    n_unique_values_treshold = MAX_UNIQUE_CATEGORIES
    for feature in df.columns:
        if feature == 'id':
            continue
        if feature != LABEL:
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types[feature] = CATEGORICAL_FEATURE
            else:
                feature_types[feature] = CONTINUOUS_FEATURE
    
    return feature_types

# Generating a constant map which we will use ahead
FEATURE_TYPES = determineTypeOfFeature(df)
for k, v in FEATURE_TYPES.items():
    print(f"Attr {k} is {v}")


# In[12]:


# Generating Random split with fixed ratio, for training and testing
train_set, test_set = train_test_split(df, .8)
print("---------------------Task-1-Completed--------------------------")


# #### Removing Outliers

# In[13]:


print("---------------------Task-2-Started----------------------------")
print(f"Inititial Dataset count {df.shape[0]}")
for feature in FEATURE_TYPES.keys():
    print()
    print(f"Feature {feature}")
    feature_mean = df[feature].mean()
    feature_std = df[feature].std()
    print(f"Mean               :{feature_mean}")
    print(f"Standard Deviation :{feature_std}")
    
    lower_bound = feature_mean - 3* feature_std
    upper_bound = feature_mean + 3*feature_std
    print(f"Lower cutoff = {lower_bound}")
    print(f"Upper cutoff = {upper_bound}")

    
    print(f"Outlier count = {df[(df[feature]<lower_bound) | (df[feature]>upper_bound)].shape[0]}")
    print("Filtering the data set")
    df = df[(df[feature]>=lower_bound) & (df[feature]<=upper_bound)]
print(f"Final dataset count after removing outliers = {df.shape[0]}")
print()


# In[14]:


toDropFeature = []
for feature in FEATURE_TYPES.keys():
    unique_count = df[feature].unique().shape[0]
    if unique_count <= 1:
        print(f"Removing feature {feature}, since the unique value left in the samples = {unique_count}")
        toDropFeature.append(feature)
        
df = df.drop(toDropFeature, axis = 1)


# In[15]:


# Generating a final constant map which we will use in futher analysis
FINAL_FEATURE_TYPES = determineTypeOfFeature(df)
print("Final features")
for k, v in FINAL_FEATURE_TYPES.items():
    print(f"Feature {k} is {v}")


# In[16]:


print("Normalizing the data set, using min-max normalization")
normalized_df=(df-df.min())/(df.max()-df.min())
print("Normalization complete")


# In[17]:


print(normalized_df.head())
print("---------------------Task-2-Completed--------------------------")


# # Task 3

# In[18]:


# Calculating Prior for all classes of label
def calculate_prior(train_data, label):
    classes = train_data[label].unique()
    prior = {}
    for c in classes:
        prior[c] = (len(train_data[train_data[label]==c])/len(train_data))
    return prior

# Calculating P(X = x | Label = Y) for categorical features
def calculate_likelihood_categorical(summary, feature, feature_value, label, Y, laplaceCorrection = False, uniqueFeatureCount = 0):
#     Data for label = Y
#     df_given_y = train_data[train_data[label] == Y]
#     p_x_given_y = len(df_given_y[df_given_y[feature] == feature_value]) / len(df_given_y)
    numerator = summary[feature][Y][feature_value][0]
    denominator = summary[feature][Y][feature_value][1]
    if(laplaceCorrection):
        numerator+=1
        denominator+=uniqueFeatureCount
    p_x_given_y = numerator / denominator
    return p_x_given_y

# Calculating P(X = x  | label = Y) for continous features
def calculate_likelihood_gaussian(summary, feature, feature_value, label, Y):
#     df_given_y = train_data[train_data[label] == Y]
#     mean, std = df_given_y[feature].mean(), df_given_y[feature].std()
    mean, std = summary[feature][Y]['mean'], summary[feature][Y]['std']

    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feature_value-mean)**2 / (2 * std**2 )))
    return p_x_given_y


# In[19]:


# class to feature to summary 
'''
summary[feature][class][mean / std] for continous
summary[feature][class][feature_value] = (tuple) (count of feature=feature value | class , count of feature class)
The second one is count so that we may use the same summary for Laplace correction
'''

def generateSummary(train_data, label):
    result = {}
    classes = train_data[label].unique()
    for feature in FINAL_FEATURE_TYPES.keys():
        if (FINAL_FEATURE_TYPES[feature] == CONTINUOUS_FEATURE):
            # store its std and mean for every label class
            feature_result = {}
            for c in classes:
                df_given_y = train_data[train_data[label] == c]
                mean, std = df_given_y[feature].mean(), df_given_y[feature].std()
                feature_result[c] = {
                    'mean': mean,
                    'std': std
                } 
            result[feature] = feature_result
        else:
#                     Condition when feature is category
            feature_result = {}
            for c in classes:
                class_result = {}
                for feature_value in train_data[feature].unique():
                    df_given_y = train_data[train_data[label] == c]
                    p_x_given_y_tuple = (len(df_given_y[df_given_y[feature] == feature_value]) , len(df_given_y))
                    class_result[feature_value] = p_x_given_y_tuple
                feature_result[c] = class_result
            result[feature] = feature_result
    return result
                    
            


# In[20]:


# count the number of unique categorical feature value
count = 0
for feature in FINAL_FEATURE_TYPES.keys():
    if FINAL_FEATURE_TYPES[feature] == CATEGORICAL_FEATURE:
        count += len(normalized_df[feature].unique())
        
UNIQUE_FEATURE_VALUES = count # we shall use this value in laplace correction


# In[21]:


print("---------------------Task-3-Started----------------------------")
print(f"unique feature value count = {UNIQUE_FEATURE_VALUES}")
summary = json.dumps(generateSummary(normalized_df, LABEL), indent=2)
print(f"Summary of normalized data : {summary}")


# In[22]:


def naive_bayes_algo(train_data, test_X, label, laplaceCorrection = False):
    # calculate prior
    prior = calculate_prior(train_data, label)
    summary = generateSummary(train_data, label)
    Y_pred = []
    # loop over every data sample
    for index, x in test_X.iterrows():
        # calculate likelihood
        classes = train_data[label].unique()
        
        likelihood = {}
        for c in classes:
            likelihood[c] = 1
            
        
        for c in classes:
            for feature in FINAL_FEATURE_TYPES.keys():
                if (FINAL_FEATURE_TYPES[feature] == CATEGORICAL_FEATURE):
                    likelihood[c] *= calculate_likelihood_categorical(summary, feature, x[feature], label, c, laplaceCorrection, UNIQUE_FEATURE_VALUES)
                else:
#                     Condition when feature is continous
                    likelihood[c] *= calculate_likelihood_gaussian(summary, feature, x[feature], label, c) 

        # calculate posterior probability (numerator only)
        post_prob = {}
        for c in classes:
            post_prob[c] = 1
        
        for c in classes:
            post_prob[c] = likelihood[c] * prior[c]
#         argmax 
        Y_pred.append(max(post_prob, key=post_prob.get))

    return np.array(Y_pred) 


# In[23]:


# Evaluates Accuracy

def GetAccuracy(Y_test, Y_pred, printDetails = True):
    
    accuracy = sum((Y_test == Y_pred)) / len(Y_test)
    if(printDetails):
        print(f"Accuracy  = {accuracy}")
    
    return accuracy

# Evaluates precision for predicting 1
def GetPrecision(Y_test, Y_pred, printDetails = True):
    marked_true_samples = sum(Y_pred == 1)
    correctly_marked = sum((Y_test == Y_pred) & (Y_test == 1))
    precision = 0
    if(marked_true_samples > 0):
        precision = correctly_marked / marked_true_samples
    if(printDetails):
        print(f"Precision = {precision}")
    
    return precision

# Evaluates recall for predicting 1
def GetRecall(Y_test, Y_pred, printDetails = True):
    true_samples = sum(Y_test == 1)
    correctly_marked = sum((Y_test == Y_pred) & (Y_test == 1))
    recall = 0
    if(true_samples >0):
        recall = correctly_marked / true_samples
    if(printDetails):
        print(f"Recall    = {recall}")
    
    return recall

def evaluateAlgo(Y_test, Y_pred, printDetails = True):
    acc = GetAccuracy(Y_test, Y_pred, printDetails)
    pre = GetPrecision(Y_test, Y_pred, printDetails)
    rec = GetRecall(Y_test, Y_pred, printDetails)
    return [acc, pre, rec]


# In[24]:


def KFolds(dataframe, k = 10):
    # shuffle the data
    shuffled_df = dataframe.sample(frac=1)    
    count = shuffled_df.shape[0]
    size_of_fold = (count + k-1)//k
    start = 0
    datas = []
    for i in range(k):
        start_idx = i*size_of_fold
        end_idx = (i+1)*size_of_fold
        datas.append(shuffled_df.iloc[start_idx: end_idx, :])
    
    return datas
    
    


# In[25]:



def kFoldValidation(dataframe, algorithm, label, k=10, laplaceCorrection = False, _folds = None):
    print(f"{k} fold validation started")
    accuracy = []
    precision = []
    recall = []
    folds = _folds
    if folds == None:
        print("Ola")
        folds = KFolds(dataframe, 10)
    for iteration in range(k):
        print(f"Iter : {iteration+1}  started")
        test = folds[iteration]
        train = pd.concat(folds[0:iteration] + folds[iteration+1:k])

        X_test = test.drop([label], axis = 1)
        Y_test = test[label]
        Y_pred = algorithm(train, test_X=X_test, label = label, laplaceCorrection = laplaceCorrection)
        acc, pre, rec = evaluateAlgo(Y_test, Y_pred)
        accuracy.append(acc)
        precision.append(pre)
        recall.append(rec)
    print()
    print(f"Mean accuracy = {np.mean(accuracy)}")
    print(f"Mean precision = {np.mean(precision)}")  
    print(f"Mean recall = {np.mean(recall)}")
    
    print(f"{k} fold validation finished")
    return folds # for task 4


# In[28]:


folds = kFoldValidation(normalized_df, naive_bayes_algo, LABEL, 10)
print("---------------------Task-3-Completed--------------------------")
print()


# In[29]:


print("---------------------Task-4-Started----------------------------")
kFoldValidation(normalized_df, naive_bayes_algo, LABEL, 10, True,  _folds = folds)
print("---------------------Task-4-Finished---------------------------")


# In[ ]:




