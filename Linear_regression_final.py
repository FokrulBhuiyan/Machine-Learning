'''The code is properly run with spyder or any IDE'''
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)
# Building the model
m = 0
c = 0
Y_pred = 0
L = 0.05  # The learning Rate

'''
input Values(save as data.csv)
1,10
1,10
2,20
2,20
2,20
3,30
3,39
4,37
5,48
5,20
6,40
'''

data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

def animate(i):
    if(i<100):
        print(i)
        plt.cla()
        global m
        global c
        global L
        global Y_pred
        
        # Performing Gradient Descent 
        n = float(len(X)) # Number of elements in X
        Y_pred = m*X + c  # The current predicted value of Y
        D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        # Draw the graph
        ax1.scatter(X, Y)
        ax1.plot([X, X], [Y, Y_pred], color='green')
        ax1.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
        plt.title('y='+"%.2f" % m+'X+'+"%.2f" % c)  
        plt.xlabel('X Data')
        plt.ylabel('Y Data')
        plt.show()

ani = animation.FuncAnimation(fig, animate, interval=100) 
ax1.scatter(X, Y)
plt.title('y='+"%.2f" % m+'X+'+"%.2f" % c)  
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.show()
