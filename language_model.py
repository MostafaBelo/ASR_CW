#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob


# In[2]:


class LanguageModel:
    def __init__(self):
        self.start_words = {}
        self.end_words = {}
        self.words = {}
        
        self.start_words_count = 0
        self.end_words_count = 0
        self.words_count = 0
        
        self.scan()
    
    def scan(self, folder = '/group/teaching/asr/labs/recordings'):
        with open("lexicon.txt", 'r') as f:
            for line in f:
                line = line.split()
                self.start_words[line[0]] = 0  # initialize frequency for each word with zero
                self.words[line[0]] = 0
                self.end_words[line[0]] = 0
                
        files = glob.glob(f'{folder}/*.txt')
        for file in files:
            with open(file, 'r') as f:
                transcription = f.readline().strip()
            
            transcription = transcription.split(" ")
            
            self.start_words[transcription[0]] += 1
            self.start_words_count += 1
            
            self.end_words[transcription[-1]] += 1
            self.end_words_count += 1
            
            for word in transcription:
                self.words[word] += 1
            self.words_count += len(transcription)
    
    def getStartWordsFrequency(self):
        return self.start_words
    
    def getEndWordsFrequency(self):
        return self.end_words
    
    def getWordsFrequency(self):
        return self.words
    
    def get_start_word_freq(self, word):
        if word in self.start_words:
            return self.start_words[word]
        return 0
    
    def get_end_word_freq(self, word):
        if word in self.end_words:
            return self.end_words[word]
        return 0
    
    def get_word_freq(self, word):
        if word in self.words:
            return self.words[word]
        return 0
    
    def get_start_word_prob(self, word):
        return self.get_start_word_freq(word) / self.start_words_count
    
    def get_end_word_prob(self, word):
        return self.get_end_word_freq(word) / self.end_words_count
    
    def get_word_prob(self, word):
        return self.get_word_freq(word) / self.words_count