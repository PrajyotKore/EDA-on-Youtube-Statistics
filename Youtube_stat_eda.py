#!/usr/bin/env python
# coding: utf-8

# 
# This notebook is the first in a series of notebooks where I explore the Youtube Statistics Dataset. More specifically, this notebook is a basic exploratory data analysis of the data.

# In[1]:


import os 
os.getcwd()


# In[2]:


os.chdir(r"C:\Users\Prajyot_kore\OneDrive\Desktop\Python\Project\Youtube Statistics")


# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[4]:



videos_stats = pd.read_csv("videos-stats.csv")
comments = pd.read_csv("Comments.csv")

videos_stats.drop_duplicates(subset='Video ID', ignore_index=True, inplace=True)

# drop unnecessary columns
videos_stats.drop(columns="Unnamed: 0", inplace=True)
comments.drop(columns="Unnamed: 0", inplace=True)


# In[5]:


videos_stats.shape


# In[6]:


comments.shape


# In[7]:


videos_stats.head()


# In[8]:


comments.head()


# # EDA

# Now , we'll start the EDA. We'll begin by looking at some trends within the comments of each video.

# 
# # Videos with keywords arranged by maximum, minimum, average and total views

# In[9]:


stats_views_df = videos_stats[['Keyword', 'Views']].groupby(['Keyword']).agg(['max', 'min', 'mean', 'sum'])
stats_views_df.sort_values(by=('Views', 'sum'), ascending=False).style.background_gradient(cmap='Reds')


# # Note
# here it is easy to see that videos with keyword "google" ,"animal", "mrbeast","bed" and "music" is pretty much more than other keywords.

# ## Average Sentiment per Video

# In[10]:


average=comments.groupby("Video ID").aggregate({"Sentiment": "mean"})
sns.histplot(average['Sentiment'],color='lightgreen')


# # Notes

#  Now, let's look at relationship between the average sentiment and the number of likes/views a video might get

# ## Average Sentiment vs. Likes/Views

# Before we explore the relationship between average comment sentiment and likes, we need to do some preprocessing first
# 
# As each video has a different number of views, we will do this by taking the ratio i.e. Likes/Views, which is just the number of likes divided by the number of views. More specifically, it is the number of likes per 1000 views.
# 
# We can use this like/view ratio to determine if a relationship exists between average comment sentiment and the number of likes

# In[11]:


average.reset_index(inplace=True)
average['Likes/Views'] = 1000 * (videos_stats['Likes'] / videos_stats['Views'])


# In[12]:


sns.regplot(data=average, x='Sentiment', y='Likes/Views')


# In[13]:


sns.lineplot(x="Sentiment", y="Likes",data=comments);


# In[14]:


sns.heatmap(average.corr())


# In[15]:


average.corr()


# # Notes

# Here, we see that  the average sentiments of ten most liked comments of a video has correlation of O.7461 tells us much about popularity /success of said video
# 

# # Next Steps

# Trends within keywords in videos

# # Statistics Per Keyword

# In[16]:


genre=videos_stats.groupby('Keyword').aggregate({"Likes":"mean","Views":"mean","Comments":"mean"})
genre.reset_index(inplace=True)


# In[17]:


plt.figure(figsize=(40,10))
sns.barplot(data=genre, x='Keyword', y='Likes', palette='rocket')


#  Here, it is easy to see that we "mrbeast" has most likes followed by "animals" and "google".

# In[18]:


plt.figure(figsize=(40, 10))
sns.barplot(data=genre, x='Keyword', y='Views', palette='rocket')


#  Here, it is easy to see that we "google" has most views followed by "animals" and  "mrbeast".

# In[19]:


plt.figure(figsize=(40, 10))
sns.barplot(data=genre, x='Keyword', y='Comments', palette='rocket')


#  Here, it is easy to see that we "mrbeast" has most comments followed by "google" and "animals".

# # Conclusion

# 1)The Mr. Beast keyword tops the charts in almost all of these aspects.
# 
# 2)In terms of views, the google and animal keyword are the highest ranked. Moreover, these two categories are very close to each other in all aspects.
