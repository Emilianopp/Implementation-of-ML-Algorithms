```python
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

#set seed for random elements
np.random.seed(100)
import warnings
warnings.filterwarnings("ignore")
```

#### Source of binary tree visualization implementation: <https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python>
#### I had to edit Tree implementation as well as print algorithm to fit it with the purpose of the task


```python
class Node:

    def __init__(self, key =None , root = None , x = None, y = None):
        if key != None:
            values = list(key.values())[0]
            self.key = values['split']
        else:
            values = {'x':None,'y':None} 
            self.key = 0
        self.root = root
        self.right = None
        self.left = None
        if x == None:
         
            self.x = values['x']
        else:
            self.x = x
        if y == None:
            self.y = values['y']
        else: 
            self.y = y


    def set_left(self,left):
        self.left = left


        
    def set_right(self,right):
        self.right = right

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

        
    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
 
            line =  f'y = {self.y} '
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()

            s = f'{self.x} >= {np.round(self.key,3)} y = {self.y} '
       
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 2) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 2 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()

            s = f'{self.x} >= {np.round(self.key,3)} y = {self.y} '
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 2) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.

        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        lines, n, p, x = self.left._display_aux()

        s = f'{self.x} >= {np.round(self.key,3)} y = {self.y} '
        
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 2) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 2 + u + y) * ' ' + '\\' + (m - y - 2) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2



```

# Decision trees 

### Now these are cool 
### The Decision trees are a bit of a different beast 
### Unlike generalised linear models which often have a defined link and loss function , 
### Decision trees are a recursevely optimized model through a criteria the most popular being gini index, although entropy is also popular 




```python
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
df= pd.read_csv('IRIS.csv')
df.species[df.species == 'Iris-setosa'] = 0
df.species[df.species == 'Iris-versicolor'] = 1
df.species[df.species == 'Iris-virginica'] = 2
df.to_numpy().T.sort(axis = 0)

```

#### Create gini purity evaluation criteria


```python
def gini(p,q):
    return (1 -(p)**2 - q**2)
```

#### Calculate the gini purity for each branch split


```python

def split_score(y_low,y_high,target,j,split):
    p_low = np.sum( np.where(y_low  == target,0,1) )/((np.sum( np.where(y_low  == target,0,1) )) + (np.sum( np.where(y_low  == target,1,0) ))  ) 
    q_low = 1- p_low
    p_high = np.sum( np.where(y_high  == target,0,1) ) / ((np.sum( np.where(y_high  == target,0,1) )) + (np.sum( np.where(y_high  == target,1,0) )))
    q_high = 1 - p_high
    g_low = gini(p_low,q_low)
    g_high = gini(p_high,q_high)
    score = (g_low*len(y_low) + g_high*len(y_high) )/ (len(y_high) + len(y_low) )                                                  
    return score
```


```python
def get_split_score(i,j,target,df):
        iteration = df[[j,target]]
        iteration = iteration.sort_values(by = j)
        # Using median as I want to use a value that is actually in the data, but could have used mean
        iteration_high = iteration[iteration[j].mean() >= iteration[j]]
        iteration_low = iteration[iteration[j].mean() < iteration[j]]
        y_high = iteration_high[target]
        y_low = iteration_low[target]
        score = split_score(y_low,y_high,i,j,iteration[j].mean())
        test = np.round(iteration[j].mean(),3)
        if(test==1.704 and score == 0):
            iteration_high.to_csv('test1.csv')
            iteration_low.to_csv('test.csv')
            print('hi')
        
        return {score:{"y":i,"x":j,"split":iteration[j].mean()}}

```


```python
def split_df(df,where):
    values = list(where.values())[0]
    x=values['x']
    y=values['y']
    split= values['split']
    df_high = df[df[x] >= split ]
    df_low = df[df[x] < split ]
    return df_high,df_low
```

Most basic Descicion tree algorithm. 
leafs are only created when gini impurity is zero for a split



```python
def build_tree(df,target,root = None,max_depth=None,depth = 1):
    x_col = df.drop(columns = [target]).columns
    y_labels = df[target].unique()
    split_list = {}
    #for each outcome calculate the meanie gini at each predictor to determine best split
    for i in y_labels: 
        for j in x_col:
            split_list.update(get_split_score(i,j,target,df))

    min_score = min(split_list.keys())
    split = split_list[min_score]
    node_val = {min_score:split}
    node = Node(key = node_val)
    df_high,df_low = split_df(df,node_val)

    #IF max depth has been reached set the value of the node to the lowest gini score
    if max_depth != None:
        if max_depth == depth:
            return Node(y = node.y)
    #If gini score is 0 it means it has seperated segregated the target value to a leaf so we do not want to keep recursing after
    # Now there are various cases so in the below code I explain how to handle them
    if min_score == 0:
        low_features = df_low[target].unique()
        high_features = df_high[target].unique()
        #In the case were the data has fully segregated the output, we return a leaf node with the value of this iterations output(y)
        if (len( low_features ) == 1 and len(high_features) == 1  ):
            if(low_features[0] == high_features[0]):
                return Node(y = node.y)
        #If the low features only has one var then set that to be a leaf
        if(len( low_features ) == 1) :
            node.set_left(Node(y=low_features[0]))
        #if the "high data" still has a mix of outputs(y) split it again to fully segregate them    
        else:
            node.set_left = build_tree(df_low,target,node,depth = depth+1,max_depth=max_depth)
        #repeated steps of left nodes for right nodes
        if(len(high_features) == 1 ):
            node.set_right(Node(y=high_features[0]))
        else:
            node.set_right(build_tree(df_high,target,node,depth = depth+1,max_depth=max_depth))
            return node
    #if it is not zero recurse to build tree 
    else:
        node.set_left(build_tree(df_low,target,node,depth = depth+1,max_depth=max_depth))
        node.set_right(build_tree(df_high,target,node,depth = depth+1,max_depth=max_depth))
    
    return node




```

#### For display simplicity we will display a tree with a depth of 3 


```python
tree=build_tree(df,'species',max_depth=3)
tree.display()
```

                         __________________petal_length >= 3.759 y = 0 ___________________                    
                        /                                                                 \                  
        _petal_length >= 1.704 y = 1 ___                                   _petal_width >= 1.723 y = 2 ___   
       /                                \                                /                               \ 
    y = 0                             y = 1                             y = 1                            y = 2 
    


```python
0.054580896686159834 == 0 
```




    False


