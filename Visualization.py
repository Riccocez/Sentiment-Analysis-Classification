
# coding: utf-8

# In[6]:

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from matplotlib import cm as cmap


# In[1]:

class Visualization(object):
    
    def __init(self):
        """
        * Generate Visualization object, which creates
        * diverse visualizations of a given set of information
        *
        *
        """
        pass
    
    def isdata_balanced(self,dataset):
        """
        * Plot distribution of classes in data
        """
        dataset.y_label.Rating.value_counts().plot(kind='bar',
                                          title="Classes distribution",
                                          figsize=(15,5),grid=True)
        plt.xlabel("Rating Class")
        plt.ylabel("No. of instances")
        
        return
    
    
    def counter(self,y_data,message):
        """
        * Returns shape of y_data in a visual form 
        """
        self.plot_distribution(y_data,message)
        print(message + ' shape {}'.format(Counter(y_data)))
        
        return
    
    def plot_distribution(self,y_data,message):
        """
        * Plot distribution of data
        """
        
        df_y_train = pd.DataFrame(y_data,columns=['Label'])
        df_y_train.Label.value_counts().plot(kind='bar',grid=True,title= "Distribution of " + message,
                                            figsize=(15,5))
        plt.xlabel('Class ID')
        plt.ylabel('No. of instances')
        plt.show()
        
        return
        
    def plot_confusion_matrix(self,y_test,test_preds,title,normalize=True,color='black'):
        """
        * Plot confusion matrix of y_test vs test_preds data
        """
        labels = ['1','1','2','3','4','5']
        
        cm = confusion_matrix(y_test, test_preds)
        #Normalize matrix
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(figsize=(15,6))

        colorbar = axes.matshow(cm, cmap=cmap.gist_heat_r)
        fig.colorbar(colorbar)
        
        for (i, j), z in np.ndenumerate(cm):
            axes.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',color=color)

        axes.set_xlabel('Predicted class', fontsize=11)
        axes.set_ylabel('True class', fontsize=11)
        
        axes.set_xticklabels(labels)
        axes.set_yticklabels(labels)
        
        plt.title(title)
        plt.show()
        
        cm = confusion_matrix(y_test, test_preds)


        return


# In[ ]:



