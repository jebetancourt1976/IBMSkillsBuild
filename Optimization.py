import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


rows = 10**7  
cols = 50    

random.seed(42)
np.random.seed(42)


data = []
for _ in range(rows):
    row = [random.uniform(0, 1) for _ in range(cols)] 
    target = random.choice([0, 1])
    row.append(target)
    data.append(row)


df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(cols)] + ['target'])


X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


model = LogisticRegression(penalty='l2', C=0.0001, solver='liblinear', max_iter=10)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
