#!/usr/bin/env python
# coding: utf-8

# In[4]:


import glob


# In[5]:


class LanguageModel:
    def __init__(self):
        self.start_words = {}
        self.end_words = {}
        
        with open("lexicon.txt", 'r') as f:
            for line in f:
                line = line.split()
                self.start_words[line[0]] = 0  # initialize frequency for each word with zero
                self.end_words[line[0]] = 0
                
        files = glob.glob('/group/teaching/asr/labs/recordings/*.txt')
        for file in files:
            with open(file, 'r') as f:
                transcription = f.readline().strip()
            
            transcription = transcription.split(" ")
            self.start_words[transcription[0]] += 1
            self.end_words[transcription[-1]] += 1
    
    def getStartWordsFrequency(self):
        return self.start_words
    
    def getEndWordsFrequency(self):
        return self.end_words


# In[7]:


lm = LanguageModel()
lm.getStartWordsFrequency()["a"]


# In[8]:


lm.getEndWordsFrequency()


# In[ ]:





# In[ ]:




