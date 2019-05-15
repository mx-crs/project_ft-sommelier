#!/usr/bin/env python
# coding: utf-8

# # V.1 Exploring the green reds

# ### a) Write a function that will plot a scatterplot matrix of your red wine data.

# In[2]:


import random
import math

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[3]:


try:
    wine_data = pd.read_csv('resources/winequality-red.csv', sep=';')
except FileNotFoundError:
    print('Set correct path')


# In[4]:


def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):

    num_examples, num_fitures = wine_data.shape
    fiture_names = wine_data.columns

    fig, axes = plt.subplots(num_fitures, num_fitures, figsize=(25, 25))
    fig.subplots_adjust(wspace=0, hspace=0)

    good_wines = wine_data[wine_data['quality'] > good_threshold]
    bad_wines = wine_data[wine_data['quality'] < bad_threshold]
    
    for x in axes.flat:
        x.xaxis.set_visible(False)
        x.yaxis.set_visible(False)

    for i, label in enumerate(fiture_names):
        axes[i, i].annotate(label.replace(' ', '\n'), (0.5, 0.5), va='center', ha='center', size=20, xycoords='axes fraction')
    
    for i in range(num_fitures):
        for j in range(i + 1, num_fitures):
            axes[i, j].scatter(good_wines[str(fiture_names[j])], good_wines[str(fiture_names[i])], marker='.', c='g', s=7)
            axes[i, j].scatter(bad_wines[str(fiture_names[j])], bad_wines[str(fiture_names[i])], marker='.', c='r', s=7)
            axes[j, i].scatter(good_wines[str(fiture_names[i])], good_wines[str(fiture_names[j])], marker='.', c='g', s=7)
            axes[j, i].scatter(bad_wines[str(fiture_names[i])], bad_wines[str(fiture_names[j])], marker='.', c='r', s=7)

    if save_plot == True:
        fig.savefig('wine_data.png')
    
    return fig


# In[5]:


fig = plot_scatter_matrix(wine_data, 6, 5, True)
plt.show(fig)


# ### b) Now that you’ve had a chance to analyze different chemical factors in red wines and their relationship with high scores (8 or higher) and low scores (3 or lower), which factors do you think will be most useful to your perceptron for distinguishing high quality vs. low quality wines? Why?
# 
# Нам подходят все сочетания этих хим. элементов:
# - Sulphates
# - Alcohol
# - Volatile acidity
# 
# Поскольку отобразив зависимость качества вина от различных сочетаний описанных химических элементов, можно сделать
# вывод, что сочетания с вышеперечисленными химическими элементами наиболее визуально разделимые.

# # V.2 Learning to perceptron
# 
# ### a) Implement a perceptron that:
# 
# - Has randomly initialized weights and bias
# - Uses the Rosenblatt perceptron learning rule (with changeable learning rate)
# - Utilizes the heaviside step activation function (discrete version)
# 
# ### b) Implement a function for training the perseptron. Training function should take in the red wine data as a parameter and should:
# 
# - Have a way to specify number of training epochs
# - Train your perceptron until it makes no errors, if training epochs is set to 0
# - Have a way to specify learning rate
# - Return a list of python tuples containing (performance): [(current_epoch, num_errors_at_epoch_end, [array_of_weights], bias), . . . ]

# In[6]:


class Matrix:
    @staticmethod
    def add(a, b):
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Invalid parametrs of 'B' matrix")
        if type(b) == int:
            for i in range(len(a)):
                a[i] = [x + b for x in a[i]]
        elif type(b) == list:
            for i in range(len(a)):
                for j in range(len(a[i])):
                    a[i][j] += b[i][j]
    
    @staticmethod
    def minus(a, b):
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Invalid parametrs of 'B' matrix")
        res = list()
        if type(b) == int:
            for i in range(len(a)):
                res = [x - b for x in a[i]]
        elif type(b) == list:
            for i in range(len(a)):
                for j in range(len(a[i])):
                    res.append([a[i][j] - b[i][j]])
        return res
                    
    @staticmethod        
    def mul(A, B):
        if len(A[0]) != len(B):
            raise ValueError("Invalid parameters of 'B' matrix")
        t_B = Matrix.transpose(B)
        r_matrix = [[0 for _ in range(len(t_B[0]))] for row in range(len(A))]
        for i in range(len(A)):
            for l in range(len(t_B)):
                for k in range(len(t_B[0])):
                    r_matrix[i][k] += A[i][l] * t_B[l][k]
        return r_matrix
    
    @staticmethod
    def scalar_mul(A, s):
        if type(s) is not int and type(s) is not float:
            raise ValueError("'s' is not a scalar")
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] *= s
        return A
    
    @staticmethod
    def scalar_div(s, A):
        if type(s) is not int and type(s) is not float:
            raise ValueError("'s' is not a scalar")
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = s / A[i][j]
        return A
    
    @staticmethod
    def scalar_add(A, s):
        if type(s) is not int and type(s) is not float:
            raise ValueError("'s' is not a scalar")
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] += s
        return A
    
    @staticmethod
    def exp(A):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = math.exp(A[i][j])
        return A
    
    @staticmethod
    def transpose(matrix):
        if type(matrix[0]) is list:
            t_matrix = [[] for _ in range(len(matrix[0]))]
            for row in matrix:
                for i, val in enumerate(row):
                    t_matrix[i].append(val)
        else:
            t_matrix = [[] for _ in range(len(matrix))]
            for i, val in enumerate(matrix):
                t_matrix[i].append(val)
        return t_matrix


# In[13]:


class Perseptron(Matrix):
    def __init__(self, wine_data, bad_tresh=3, good_tresh=8, feature_scaling=False):
        self.feature_scaling = feature_scaling
        self.feature_set = self.get_specified_data(wine_data, bad_tresh, good_tresh, columns=[8, 10, 11])
        self.scaled_set = self.ft_feature_scaling()
        self.feature_size = len(self.feature_set), len(self.feature_set[0]) - 1
        self.weights = [random.choice([x/1000 for x in range(-1000, 1000)]) for _ in range(self.feature_size[1])]
        self.y = [[0 if q[-1] <= 3 else 1] for q in self.feature_set]
        self.bad_tresh = bad_tresh
        self.good_tresh = good_tresh

    
    def get_specified_data(self, wine_data, bad_tresh, good_tresh, columns):
        tmp = [[1] + list(exmpl) for exmpl in wine_data.values[:,columns]]
        data_set = list()
        for i in range(len(tmp)):
            if tmp[i][-1] <= bad_tresh or tmp[i][-1] >= good_tresh:
                data_set.append(tmp[i])
        return data_set
    
    
    def ft_feature_scaling(self):
        scaled_set = []
        f1 = {
            "max": max([row[1] for row in self.feature_set]),
            "min": min([row[1] for row in self.feature_set])
        }
        f2 = {
            "max": max([row[2] for row in self.feature_set]),
            "min": min([row[2] for row in self.feature_set])
        }
        if self.feature_scaling:
            for row in self.feature_set:
                scaled_set.append([row[0]] + [(row[1]-f1['min']) / (f1['max']-f1['min'])] +
                                  [(row[2]-f2['min']) / (f2['max']-f2['min'])] + row[3:])
        return scaled_set
    
    
    def hypothesis(self, feature_set, weights):
        return Matrix.mul(feature_set, weights)

    
#     def sigmoid_func(self, h):
#         return Matrix.scalar_div(1, Matrix.scalar_add(Matrix.exp(Matrix.scalar_mul(h, -1)), 1))
    
    
    def heaviside_func(self, h):
        return [[0 if x < 0 else 1 for x in row] for row in h]
    
    
    def activation_func(self, h):
        return self.heaviside_func(h)
    
    
    def calc_error_f1_score(self, output_y):
        c_dict = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
        }

        for i, val in enumerate(self.y):
            if val[0] and output_y[i][0]:
                c_dict['tp'] += 1
            elif not val[0] and output_y[i][0]:
                c_dict['fp'] += 1
            elif val[0] and not output_y[i][0]:
                c_dict['fn'] += 1
        try:
            precision = c_dict['tp'] / (c_dict['tp'] + c_dict['fp'])
        except ZeroDivisionError:
            return 0
        try:
            recall = c_dict['tp'] / (c_dict['tp'] + c_dict['fn'])
        except ZeroDivisionError:
            return 0
        f1_score = 2 * ((precision*recall) / (precision+recall))
        return int(f1_score)
    
    
    def calc_error_simple(self, output_y):
        errors = 0
        for i, val in enumerate(self.y):
            if val[0] != output_y[i][0]:
                errors += 1
        return errors
    
    
    def train(self, l_rate=0.01, epochs=0):
        h = 0
        epoch = 0
        result = list()
        if self.feature_scaling:
            feature_set = [row[0:-1] for row in self.scaled_set]
        else:
            feature_set = [row[0:-1] for row in self.feature_set]
        while True:
            epoch += 1
            h = self.hypothesis(feature_set, self.weights)
            inter_res = self.activation_func(h)
            num_errors = self.calc_error_simple(inter_res)
            result.append((epoch, num_errors, self.weights[1:], self.weights[0]))
            if epochs == 0:
                if num_errors == 0:
                    return result
            elif epochs == epoch:
                return result
            for i, Xi in enumerate(feature_set):
                update = (self.y[i][0] - inter_res[i][0]) * l_rate
                for j, w in enumerate(self.weights):
                    if j == 0:
                        self.weights[j] += update
                    else:
                        self.weights[j] += update * Xi[j]


# ### c) Write a function that will take the output of the perceptron training function and the red wine data and generate two plots in one figure:
# 
# - The first plot should plot the number of errors the perceptron made as a function of epoch. Be careful with how you calculate errors
# - The second plot should plot the decision boundary of your perceptron and also show 'good' and 'bad' wine data points on the final training epoch. This second plot should also shade 'good' and 'bad' areas
# - The function should allow the user to specify a specific epoch and see what the decision boundary of the perceptron was on that epoch. If a negative epoch is given, cause the plots to show the last epoch

# In[14]:


def decision_boundary(weights, range_):
    slope = (-weights[0]/weights[2]) / (weights[0]/weights[1])
    b = (-weights[0]/weights[2])
    return [x * slope + b for x in range_]

def plot_performance(performance, wine_data, good_tresh, bad_tresh, epoch=-1, save_plot=False):
    """
    Plot the performance of our perceptron or adaline.
    This function will produce a two plot figure:
    1) Classification Errors vs. Epochs
    2) Decision boundery for two factors
    """
    good_wines = {
        'pH': [],
        'alcohol': []
    }
    bad_wines = {
        'pH': [],
        'alcohol': []
    }
        
    for val in wine_data:
        if val[3] >= good_tresh:
            good_wines['pH'].append(val[1])
            good_wines['alcohol'].append(val[2])
        elif val[3] <= bad_tresh:
            bad_wines['pH'].append(val[1])
            bad_wines['alcohol'].append(val[2])
    
#     Data for first plot
    epochs = [curr_epoch[0] for curr_epoch in performance]
    errors = [err[1] for err in performance]
    weights = []
    
#     Determine an epoch for second plot
    if epoch == -1 or epoch > performance[-1][0]:
        weights = [performance[-1][-1]] + performance[-1][-2]
        epoch = performance[-1][0]
    elif epoch >= 0:
        weights = [performance[epoch-1][-1]] + performance[epoch-1][-2]
        epoch = performance[epoch-1][0]
    
#     Decision boundary
    x = min([row[1] for row in wine_data]), max([row[1] for row in wine_data])
    y = decision_boundary(weights, range(15))

#     For adjustment of asix
    ymnmx = min([row[2] for row in wine_data]), max([row[2] for row in wine_data])
    scy = (ymnmx[1] - ymnmx[0]) / len(wine_data)
    scx = (x[1] - x[0]) / len(wine_data)
    
#     Plotting
    fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(12, 4))
    
    plt1.set_title('Errors as a function of epoch')
    plt1.set_xlabel('epoch')
    plt1.set_ylabel('classification errors')
    
    plt2.set_title(f'Decision boundary on epoch: {epoch}')
    plt2.set_xlabel('alcohol')
    plt2.set_ylabel('pH')
    plt2.set_xlim(ymnmx[0]-scy, ymnmx[1]+scy)
    plt2.set_ylim(x[0]-scx, x[1]+scx)

    plt1.plot(epochs, errors)
    plt2.fill_between(decision_boundary(weights, range(-15, 15)),
                      range(-15, 15), -1, facecolor='#6EDD5E', alpha=0.3)
    plt2.fill_between(decision_boundary(weights, range(-15, 15)), range(-15, 15),
                     int(max(wine_data[1])), facecolor='#FFB4FA', alpha=0.3)
    plt2.scatter(bad_wines['alcohol'], bad_wines['pH'], s=13, color='#FF00F0', label=f'bad wines (>{good_tresh-1} score)')
    plt2.scatter(good_wines['alcohol'], good_wines['pH'], s=13, color='g', label=f'good wines (<{bad_tresh+1} score)')
    plt2.plot(decision_boundary(weights, range(-15, 15)), range(-15, 15), '--', label='Decision Boundary')
    plt2.legend(loc=1, bbox_to_anchor=(1.52, 1))


# In[24]:


perseptron = Perseptron(wine_data)
plot_performance(perseptron.train(), perseptron.feature_set, perseptron.good_tresh, perseptron.bad_tresh)


# ### d) Adding the feature scaling and using the function that was created in part "c)" to plot and verify that the perceptron is learning more efficiently.

# In[23]:


perseptron = Perseptron(wine_data, feature_scaling=True)
plot_performance(perseptron.train(), perseptron.scaled_set, perseptron.good_tresh, perseptron.bad_tresh)


# In[ ]:




