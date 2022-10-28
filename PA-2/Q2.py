#!/usr/bin/env python
# coding: utf-8

# ### Programming Assignment 2 Q2
# ### Supervised Learning
# 
# Group Number : 46
# 
# Student 1 : Aryan Singh 19CS30007
# 
# Student 2 : Seemant Guruprasad Achari 19CS30057
# 

# In[176]:


# importing various tools and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from sklearn import svm
from sklearn.neural_network import MLPClassifier


# In[118]:


# Column 1 is defined as the class label
LABEL = 1 

# To make outputs reproducible
np.random.seed(101)


# In[119]:


# loading the dataset

df = pd.read_csv('lung-cancer.data', header = None, na_values=["?"])
df.columns += 1


# In[120]:


print(df.head())

print("Dataset loaded successfully!")


# In[121]:


for attr,value in df.isna().sum().items():
    if(value > 0):
        print(f"Attribute {attr} has {value} missing data.")


# In[122]:


# Filling the na values with mode of the columns
fill_mode = lambda col: col.replace(np.nan, col.mode()[0])
df_without_na = df.apply(fill_mode, axis=0)


# In[123]:


for attr,value in df_without_na.isna().sum().items():
    if(value > 0):
        print(f"Attribute {attr} has {value} missing data.")
print("Missing data handled using mode!")


# In[124]:


print(df_without_na.head())


# In[125]:


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


# In[126]:


df_without_na[LABEL].value_counts(True)


# In[127]:


train_df, test_df = train_test_split(df_without_na,0.8)


# In[128]:


train_df[LABEL].value_counts()


# In[129]:


print(f"The shape of training dataset {train_df.shape}")
print(f"The shape of testing dataset {test_df.shape}")


# In[130]:


train_stats = train_df.describe()
train_stats.pop(LABEL)
train_stats = train_stats.transpose()
print(train_stats)


# In[131]:


attr_to_drop = []
for row, val in train_stats["std"].items():
    if val == 0:
        print(f"Attribute {row} has 0 std in the training set, hence droping it.")
        attr_to_drop.append(row)
train_stats_upd = train_stats.drop(attr_to_drop)
train_df_upd = train_df.drop(attr_to_drop, axis = 1)
test_df_upd = test_df.drop(attr_to_drop, axis = 1)
        


# In[132]:


train_labels = train_df_upd.pop(LABEL)
test_labels = test_df_upd.pop(LABEL)


# ### Data Normalization

# In[166]:


def standard_scalar_normalization(df, df_stats):
    # print(df.shape)
    norm_df = (df - df_stats["mean"])/df_stats["std"]
    np.nan_to_num(norm_df, False, 0.0)
    return norm_df


# In[134]:


norm_train = standard_scalar_normalization(train_df_upd, train_stats_upd)
norm_test = standard_scalar_normalization(test_df_upd, train_stats_upd)


# In[135]:


print("Normalization complete")


# In[136]:


#Create a svm Classifier
model = svm.SVC(C = 1, # reg paramater
                kernel='linear', #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
               ) # Linear Kernel

#Train the model using the training sets
model.fit(norm_train, train_labels)

#Predict the response for test dataset
y_pred = model.predict(norm_test)


# In[137]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))


# In[138]:


def GetAccuracy(trueLabel, pred, printDetails = True):
    total_samples = len(trueLabel)
    cnt = (trueLabel == pred).sum()
    accuracy = cnt / total_samples
    if(printDetails):
        print(f"Accuracy  = {cnt / total_samples}")
    
    return accuracy


# In[139]:


model_map = {}
accuracy_map = {}


# In[140]:


kernel_types = ['linear', 'poly', 'rbf']

for kernel_type in kernel_types:

    #Create a linear kernel svm Classifier
    model = svm.SVC(C = 1, 
                    kernel=kernel_type, 
                    degree = 2 # will be ignored by all kernel except poly
                    # Thus poly will become quadratic
                ) # Linear Kernel

    #Train the model using the training sets
    model.fit(norm_train, train_labels)


    y_pred = model.predict(norm_train)
    acc = GetAccuracy(train_labels, y_pred, False)
    print(f"Training Accuracy for SVM with {kernel_type} kernel = {acc}")

    #Predict the response for test dataset
    y_pred = model.predict(norm_test)
    acc = GetAccuracy(test_labels, y_pred, False)
    print(f"Testing Accuracy for SVM with {kernel_type}  kernel = {acc}")


    model_map[kernel_type] = model
    accuracy_map[kernel_type] = acc


# In[149]:


mlp_model_map= {}
mlp_model_accuracy = {}
mlp_models_sizes = {
    'Single Hidden Layer' : (16,),
    'Multi Hidden Layer' : (256, 16)
}


# In[151]:


for name, layerSize in mlp_models_sizes.items():
    mlp = MLPClassifier(learning_rate_init = 0.001, batch_size=32, hidden_layer_sizes=layerSize, solver='sgd')
    mlp.fit(norm_train, train_labels)
    pred = mlp.predict(norm_test)
    acc = GetAccuracy(test_labels, pred, False)
    print(f"Testing Accuracy for MLP with {name}  = {acc}")
    mlp_model_map[name] = mlp
    mlp_model_accuracy[name] = acc


# In[143]:


mlp = MLPClassifier(learning_rate_init = 0.001, batch_size=32, hidden_layer_sizes=(16,), solver='sgd')
mlp


# In[144]:


mlp.fit(norm_train, train_labels)


# In[145]:


pred = mlp.predict(norm_test)


# In[148]:


GetAccuracy(test_labels, pred)


# In[156]:




bestSetting = max(mlp_model_accuracy, key= mlp_model_accuracy.get)
LayerSize = mlp_models_sizes[bestSetting]
y_data = []
x_data = [0.1, 0.01, 0.001, 0.0001, 0.00001]
for l_rate in x_data:
    mlp = MLPClassifier(learning_rate_init = l_rate, batch_size=32, hidden_layer_sizes=layerSize, solver='sgd')
    mlp.fit(norm_train, train_labels)
    pred = mlp.predict(norm_test)
    acc = GetAccuracy(test_labels, pred, False)
    print(f"Testing Accuracy for MLP with {name}  = {acc}")
    y_data.append(acc)


# In[158]:


plt.figure(figsize = (10,8))
plt.plot(x_data, y_data)
plt.title("learning rate vs accuracy")
plt.xlabel("learning rate")
plt.ylabel("accuracy")
plt.xscale("log")
plt.show()


# In[167]:


def modelEvaluation(model, df):
    fill_mode = lambda col: col.replace(np.nan, col.mode()[0])
    df_without_na = df.apply(fill_mode, axis=0)
    train_df, test_df = train_test_split(df_without_na,0.8)
    train_stats = train_df.describe()
    train_stats.pop(LABEL)
    train_stats = train_stats.transpose()
    attr_to_drop = []
    for row, val in train_stats["std"].items():
        if val == 0:
            print(f"Attribute {row} has 0 std in the training set, hence droping it.")
            attr_to_drop.append(row)
            return 0.0 # since it has a useless attribute
    train_stats_upd = train_stats.drop(attr_to_drop)
    train_df_upd = train_df.drop(attr_to_drop, axis = 1)
    test_df_upd = test_df.drop(attr_to_drop, axis = 1)
    # print(train_df_upd.shape)
    train_labels = train_df_upd.pop(LABEL)
    test_labels = test_df_upd.pop(LABEL)
    norm_train = standard_scalar_normalization(train_df_upd, train_stats_upd)
    norm_test = standard_scalar_normalization(test_df_upd, train_stats_upd)

    mlp = model
    mlp.fit(norm_train, train_labels)
    pred = mlp.predict(norm_test)
    acc = GetAccuracy(test_labels, pred, False)
    print(f"Testing Accuracy for MLP = {acc}")
    return acc

    
def sfs(model, pd_data):
    '''
    # takes data frame and model as input
    # and then returns the dataframe with the optimal attributes
    '''
    # Record the name of the column that contains the instance IDs and the class
    no_of_columns = len(pd_data.columns) # number of columns
    class_column_index = 0
    class_column_colname = pd_data.columns[class_column_index]
 
    # Record the number of available attributes
    no_of_available_attributes = no_of_columns - 1
 
    # Create a dataframe containing the available attributes by removing
    # the Instance and the Class Column
    available_attributes_df = pd_data.drop(columns = [class_column_colname]) 
 
    # Create an empty optimal attribute dataframe containing only the
    # Instance and the Class Columns
    optimal_attributes_df = pd_data[[class_column_colname]]
 
    # Set the base performance to a really low number
    base_performance = -9999.0
 
    # Check whether adding a new attribute to the optimal attributes dataframe
    # improves performance
    # While there are still available attributes left
    while no_of_available_attributes > 0: 
        # Set the best performance to a low number
        best_performance = -9999.0
 
        # Initialize the best attribute variable to a placeholder
        best_attribute = "Placeholder"
 
        # For all attributes in the available attribute data frame
        for col in range(0, len(available_attributes_df.columns)):
 
            # Record the name of this attribute
            this_attr = available_attributes_df.columns[col]
         
            # Create a new dataframe with this attribute inserted
            temp_opt_attr_df = optimal_attributes_df.copy()
            temp_opt_attr_df.insert(loc=1,column=this_attr,value=(available_attributes_df[this_attr]))
 
            # Run Naive Bayes on this new dataframe and return the 
            # classification accuracy
            current_performance = modelEvaluation(model, temp_opt_attr_df)
 
            # Find the new attribute that yielded the greatest
            # classification accuracy
            if current_performance > best_performance:
                best_performance = current_performance
                best_attribute = this_attr
 
        # Did adding another feature lead to improvement?
        if best_performance > base_performance:
            base_performance = best_performance
 
            # Add the best attribute to the optimal attribute data frame
            optimal_attributes_df.insert(
                loc=1,column=best_attribute,value=(
                available_attributes_df[best_attribute]))
 
            # Remove the best attribute from the available attribute data frame
            available_attributes_df = available_attributes_df.drop(
                columns = [best_attribute]) 
 
            # Print the best attribute to the console
            print()
            print(str(best_attribute) + " added to the optimal attribute subset")
            print()
            
            # Decrement the number of available attributes by 1
            no_of_available_attributes -= 1
 
            # Print number of attributes remaining to the console
            print()
            print(str(no_of_available_attributes) + " attributes remaining")
            print()
            print()
        else:
            print()
            print("Performance did not improve this round.")
            print("End of Stepwise Forward Selection.")
            print()
            break
 
    # Return the optimal attribute set
    return optimal_attributes_df


# In[168]:


bestSetting = max(mlp_model_accuracy, key= mlp_model_accuracy.get)
bestModel = mlp_model_map[bestSetting]

opt_df = sfs(bestModel, df)


# In[172]:


print(f"The best set of features are {list(opt_df.columns)}, here we have also included the label class 1")


# In[202]:


class ensembleModel:
    def __init__(self, models):
        self.models = models
    def fit(self, train_df, train_label):
        for model in self.models:
            model.fit(train_df, train_label)
    
    def predict(self, test_df):
        predictions = []
        for model in self.models:
            pred = model.predict(test_df)
            predictions.append(pred)
        result = scipy.stats.mode(np.stack(predictions), axis=0)
        return result.mode[0]


# In[203]:


bestSetting = max(mlp_model_accuracy, key= mlp_model_accuracy.get)
bestModel = mlp_model_map[bestSetting]
modelsForEnsemble = [model_map['poly'], model_map['rbf'], bestModel]

ensemblemodel= ensembleModel(modelsForEnsemble)

ensemblemodel.fit(norm_train, train_labels)
pred = ensemblemodel.predict(norm_test)
acc = GetAccuracy(test_labels, pred, False)
print(f"Testing Accuracy for Max Vote Ensemble model = {acc}")


# In[174]:


predict1 = model_map['linear'].predict(norm_test)
predict2 = model_map['poly'].predict(norm_test)


# In[183]:


print(predict1)
print(predict2)
arrays = [predict1, predict2, predict1]


# In[184]:


result = scipy.stats.mode(np.stack(arrays), axis=0)
print(result)
result.mode


# In[198]:


predtype


# In[ ]:




