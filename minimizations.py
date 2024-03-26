#!/usr/bin/env python
# coding: utf-8

# In[1]:


words = []
f = open("lexicon.txt")
data = f.read().strip()
f.close()

for line in data.split("\n"):
    line = line.split(" ")
    words.append([line[0], line[1:]])


# In[38]:


class Node:
    def __init__(self, data: str, out_lbl = ""):
        self.data = data
        self.children = []
        self.out_lbl = out_lbl
        self.words_below = []
    
    def add_child(self, child_data):
        child = Node(child_data)
        
        if not(self.is_branching()) and self.out_lbl != "":
            eps_child = Node("eps")
            eps_child.set_out_lbl(self.out_lbl)
            self.set_out_lbl("")
            eps_child.add_word_below(self.words_below[0])
            self.children.append(eps_child)
            
        self.children.append(child)
        return child
        
    def add_parent(self, parent):
        parent.add_child(self)
        
    def find_child(self, data):
        for child in self.children:
            if child.data == data:
                return child
        
        return None
    
    def get_child(self, data):
        child = self.find_child(data)
        if child != None:
            return child
        
        child = self.add_child(data)
        return child
    
    def set_out_lbl(self, out_lbl):
        self.out_lbl = out_lbl
        
    def child_count(self):
        return len(self.children)
    
    def is_branching(self):
        return self.child_count() > 1
    
    def add_word_below(self, word):
        self.words_below.append(word)
        
    def isLast(self):
        return len(self.words_below) <= 1 and self.out_lbl != ""
    def getLastWord(self):
        if self.out_lbl != "":
            return self.out_lbl
        elif self.isLast:
            return self.words_below[0]
        
        return ""
    
    def get_repr(self):
        tab_str = "----"
        res = f"{self.data}({self.out_lbl}){self.words_below}"
        for child in self.children:
            child_data = f"\n{tab_str}".join(str(child).split("\n"))
            res += f"\n{tab_str}{child_data}"
        return res
    
    def __str__(self):
        return self.get_repr()

    def __repr__(self):
        return self.get_repr()

# In[44]:


tree_root = Node("root")


# In[45]:


for word in words:
    current = tree_root
    current.add_word_below(word[0])
    for phone in word[1]:
        current = current.get_child(phone)
        current.add_word_below(word[0])
    current.set_out_lbl(word[0])