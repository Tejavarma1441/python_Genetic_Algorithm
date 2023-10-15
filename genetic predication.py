import numpy as np
import random
def generateParents(size):
    parents = np.array(random.randint(0, 2**size - 1))
    for i in range(1, population):
        parents = np.append(parents, random.randint(0, 2**size - 1))
    return parents
def totalSize(data, size):
    s = 0
    for i in range(0, size-1):
        if(data & (1 << i) > 0):
            s += mp3s[i]
    return s
def reduceSize(rec, size):
    while totalSize(rec, size) > 700:
        index = random.randint(0, size - 1)
        if(rec & (1 << index) > 0):
            rec = rec ^ (1 << index)
    return rec
def mutate(rec, size):
    index = random.randint(0, size - 1)
    rec = rec ^ (1 << index)
    return rec
def fixChromosomes(data, size, population):
    datasize = data.shape[0]
    fitness = np.zeros((datasize,1), dtype=int)
    for i in range(0, datasize):
        rec = data[i]
        if(totalSize(rec, size) > 700):
            rec = reduceSize(rec, size)
            data[i] = rec
        fitness[i] = -1* totalSize(data[i], size)
    data = np.transpose(np.array([data]))
    generation = np.concatenate((data, fitness), axis=1)
    generation = generation[generation[:population, 1].argsort()]
    return generation
def crossover(mom, dad, size):
    index = random.randint(1, size - 1)
    mom1 = mom & (2**index -1)
    mom2 = mom & ((2**(size-index) -1) << index)
    dad1 = dad & (2**index -1)
    dad2 = dad & ((2**(size-index) -1) << index)
    return mutate(mom1|dad2, size), mutate(dad1|mom2, size)
def newGeneration(generation, size):
    top4 = generation[:4, 0]
    newGen = generation[:2,0]
    for i in range(0, 4):
        for j in range(0, 4):
            if(i != j):
                c1, c2 = crossover(top4[i], top4[j], size)
                newGen = np.append(newGen, c1)
                newGen = np.append(newGen, c2)
                #print(newGen)
    return newGen
def train(mp3Cnt, mp3s, population, generationsPerCD):
    curCD = 1
    combinedSizes = totalSize(2**mp3Cnt-1, mp3Cnt)
    doneSizes = 0.0
    while(True):
        if(mp3Cnt == 0):
            break
        parents = generateParents(mp3Cnt)
        generation = fixChromosomes(parents, mp3Cnt, population)
        ng = generation
        for i in range(generationsPerCD):
            ng = newGeneration(ng, mp3Cnt)
            ng = fixChromosomes(ng, mp3Cnt, population)
        allFileSize = totalSize(2**mp3Cnt-1, mp3Cnt)
        cdContents = ng[0,0]
        if(allFileSize < 700):
            cdContents = 2**mp3Cnt -1
        currentBestCDSize = totalSize(cdContents, mp3Cnt)
        if(currentBestCDSize >= 699 or allFileSize < 700):
            indexesToRemove = []
            for i in range(0, mp3Cnt):
                if(cdContents & (1 << i) > 0):
                    indexesToRemove.append(i)
            indexesToRemove = list(reversed(indexesToRemove))
            doneSizes += currentBestCDSize
            print("CD"+ str(curCD) + ": MP3 Count:" + str(len(indexesToRemove)) + " Size: " + str(currentBestCDSize))
            mp3Cnt = mp3Cnt - len(indexesToRemove)
            for i in range(len(indexesToRemove)):
                mp3s = np.delete(mp3s, indexesToRemove[i])
            curCD = curCD + 1
        else:
            continue
population = 10
mp3Cnt = 100
generationsPerCD = 3
maxFileSize = 100
mp3s = maxFileSize*np.random.rand(mp3Cnt, 1)

train(mp3Cnt, mp3s, population, generationsPerCD)

"""product demand"""

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
data.head()
data.isnull().sum()
data = data.dropna()
fig = px.scatter(data, x="Units Sold", y="Total Price",
                 size='Units Sold')
fig.show()
print(data.corr())
correlations = data.corr(method='pearson')
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
#features = [["Total Price", "Base Price"]]
features = np.array([[133.00, 140.00]])
model.predict(features)
