import os
import re
from collections import Counter

def main():
    fin = os.popen("awk -F',' '{print $3}' ../train/Train.csv").readlines()
    text = map(lambda x: x.strip(), fin)
    words = []
    for row in text:
        for word in row.split(' '):
            #drop the irrelavant symbols
            word = re.sub(r'[^\w]','',word)
            words.append(word)
    wordCount = Counter(words)
    
    with open('term_freq.txt','w') as fout:
       for k in sorted(wordCount, key=wordCount.get, reverse=True):
           fout.write(k+'\t'+str(wordCount[k])+'\n')
           
if __name__ == "__main__":
    main()