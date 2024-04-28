
import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Load language model
nlp = spacy.load("en_core_web_lg")
print(spacy.prefer_gpu())
#Read in Data

df = pd.read_excel('ABC.xlsx')
df = df.fillna(" ")
labels = list(df.columns)
numBooks = len(df)
#Similarity Arrays
simArr = np.zeros((numBooks,26,26))
simArrNouns = np.zeros((numBooks,26,26))
simArrVerbs = np.zeros((numBooks,26,26))
simArrAdjs = np.zeros((numBooks,26,26))
#Average Similarity Arrays
simArrTotal = np.zeros((26,26))
simArrTotal_NOUN = np.zeros((26,26))
simArrTotal_VERB = np.zeros((26,26))
simArrTotal_ADJ = np.zeros((26,26))
avgSim = np.zeros(numBooks)
#Compute similarity matrices 
for i in range(numBooks):
    for j in range(3, len(df.columns)):
        w1 = df.iloc[i,j]
        w1 = nlp(w1)
        w1_NOUNS = " ".join(token.lemma_ for token in w1 if token.pos_ == "NOUN")
        w1_VERBS = " ".join(token.lemma_ for token in w1 if token.pos_ == "VERB")
        w1_ADJ = " ".join(token.lemma_ for token in w1 if token.pos_ == "ADJ")
        for k in range(3, len(df.columns)):
           w2 = df.iloc[i,k]
           
           w2 = nlp(w2)
           w2_NOUNS = " ".join(token.lemma_ for token in w2 if token.pos_ == "NOUN")
           w2_VERBS = " ".join(token.lemma_ for token in w2 if token.pos_ == "VERB")
           w2_ADJ = " ".join(token.lemma_ for token in w2 if token.pos_ == "ADJ")
           sim = w1.similarity(w2)
           simNOUN = nlp(w1_NOUNS).similarity(nlp(w2_NOUNS))
           simVERB = nlp(w1_VERBS).similarity(nlp(w2_VERBS))
           simADJ = nlp(w1_ADJ).similarity(nlp(w2_ADJ))
           
           simArr[i,j-3,k-3] = sim
           simArrNouns[i,j-3,k-3] = simNOUN
           simArrVerbs[i,j-3,k-3] = simVERB
           simArrAdjs[i,j-3,k-3] = simADJ
           
           simArrTotal[j-3,k-3] += sim
           simArrTotal_NOUN[j-3,k-3] += simNOUN
           simArrTotal_VERB[j-3,k-3] += simVERB
           simArrTotal_ADJ[j-3,k-3] += simADJ
           

    
    fig, axes = plt.subplots(2,2)
    fig.tight_layout(pad = 3.0)
    fig.suptitle(df.iloc[i,1])

    axes[0,0].set_title("Semantic Similarity")
    axes[0,1].set_title("Semantic Similarity: Nouns")
    axes[1,0].set_title("Semantic Similarity: Verbs")
    axes[1,1].set_title("Semantic Similarity: Adjectives")
    
    hm = sns.heatmap(simArr[i,:,:],xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[0,0])
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 5,rotation = 0)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 5)
    
    hmn = sns.heatmap(simArrNouns[i,:,:],xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[0,1])
    hmn.set_xticklabels(hmn.get_xmajorticklabels(), fontsize = 5,rotation = 0)
    hmn.set_yticklabels(hmn.get_ymajorticklabels(), fontsize = 5)
    
    hmv = sns.heatmap(simArrVerbs[i,:,:],xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[1,0])
    hmv.set_xticklabels(hmv.get_xmajorticklabels(), fontsize = 5,rotation = 0)
    hmv.set_yticklabels(hmv.get_ymajorticklabels(), fontsize = 5)
    
    hma = sns.heatmap(simArrAdjs[i,:,:],xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[1,1])
    hma.set_xticklabels(hma.get_xmajorticklabels(), fontsize = 5, rotation = 0)
    hma.set_yticklabels(hma.get_ymajorticklabels(), fontsize = 5)
    
    outTxt = "Heatmaps/heatmap{num}.png"
    plt.savefig(outTxt.format(num = i))
    plt.clf()



#average the average heatmap data
simArrTotal = np.divide(simArrTotal,numBooks)
simArrTotal_NOUN = np.divide(simArrTotal_NOUN,numBooks)
simArrTotal_VERB = np.divide(simArrTotal_VERB,numBooks)
simArrTotal_ADJ = np.divide(simArrTotal_ADJ,numBooks)

fig, axes = plt.subplots(2,2)
fig.tight_layout(pad = 3.0)
fig.suptitle("Average Semantic Similarity")

axes[0,0].set_title("Semantic Similarity")
axes[0,1].set_title("Semantic Similarity: Nouns")
axes[1,0].set_title("Semantic Similarity: Verbs")
axes[1,1].set_title("Semantic Similarity: Adjectives")

hm = sns.heatmap(simArrTotal,xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[0,0],cmap="viridis")
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 5,rotation = 0)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 5)

hmn = sns.heatmap(simArrTotal_NOUN,xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[0,1],cmap="viridis")
hmn.set_xticklabels(hmn.get_xmajorticklabels(), fontsize = 5,rotation = 0)
hmn.set_yticklabels(hmn.get_ymajorticklabels(), fontsize = 5)

hmv = sns.heatmap(simArrTotal_VERB,xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[1,0],cmap="viridis")
hmv.set_xticklabels(hmv.get_xmajorticklabels(), fontsize = 5,rotation = 0)
hmv.set_yticklabels(hmv.get_ymajorticklabels(), fontsize = 5)

hma = sns.heatmap(simArrTotal_ADJ,xticklabels=labels[3:],yticklabels=labels[3:],ax=axes[1,1],cmap="viridis")
hma.set_xticklabels(hma.get_xmajorticklabels(), fontsize = 5, rotation = 0)
hma.set_yticklabels(hma.get_ymajorticklabels(), fontsize = 5)

outTxt = "Heatmaps/heatmap_AVERAGE.png"
plt.savefig(outTxt)
plt.clf()