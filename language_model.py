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
        
        self.bigram_words_one = {}
        
        self.start_words_count = 0
        self.end_words_count = 0
        self.words_count = 0
        self.bigram_count = 0
        
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
            
            prev = ""
            for word in transcription:
                key = f"{word}|{prev}"
                if key in self.bigram_words_one:
                    self.bigram_words_one[key] += 1
                else:
                    self.bigram_words_one[key] = 0
                
                prev = word
            
            self.bigram_count = len(self.bigram_words_one)
            
#         for w in self.start_words:
#             s = self.start_words[w]
#             m = self.words[w]
#             e = self.end_words[w]
#             print(f"{w} & {s} & {m} & {e}")
#         print("start", self.start_words)
#         print ("end", self.end_words)
#         print ("mid", self.words)
#         print("bigram", self.bigram_words_one)
    
    def getStartWordsFrequency(self):
        return self.start_words
    
    def getEndWordsFrequency(self):
        return self.end_words
    
    def getWordsFrequency(self):
        return self.words
    
    def getBigramFrequency(self):
        return self.bigram_words_one
    
    def get_start_word_freq(self, word):
        if isinstance(word, list):
            freq = 0
            for w in word:
                if w in self.start_words:
                    freq += self.start_words[w]
            return freq
        else:
            if word in self.start_words:
                return self.start_words[word]
            return 0
    
    def get_end_word_freq(self, word):
        if word in self.end_words:
            return self.end_words[word]
        return 0
    
    def get_word_freq(self, word):
        if isinstance(word, list):
            freq = 0
            for w in word:
                if w in self.words:
                    freq += self.words[w]
            return freq
        else:
            if word in self.words:
                return self.words[word]
            return 0
        
    def get_bigram_word_freq(self, word, prev):
        if isinstance(word, list) and isinstance(prev, list):
            raise Exception("Invalid arguments for bigram LM")
        
        if isinstance(word, list):
            freq = 0
            for w in word:
                freq += self.get_bigram_word_freq(w, prev)
            return freq
        
        if isinstance(prev, list):
            freq = 0
            for w in prev:
                freq += self.get_bigram_word_freq(word, w)
            return freq
        
        key = f"{word}|{prev}"
        if key in self.bigram_words_one:
            return self.bigram_words_one[key]
        return 0
    
    def get_start_word_prob(self, word):
        return self.get_start_word_freq(word) / self.start_words_count
    
    def get_end_word_prob(self, word):
        return self.get_end_word_freq(word) / self.end_words_count
    
    def get_word_prob(self, word):
        return self.get_word_freq(word) / self.words_count
    
    def get_bigram_word_prob(self, word, prev):
        return self.get_bigram_word_freq(word, prev) / self.bigram_count