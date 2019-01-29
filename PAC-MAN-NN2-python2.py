import numpy as np
import random
from copy import deepcopy
from keras.models import Model, Sequential
from keras.layers.core import Activation, Dropout, Dense, Merge
from keras.layers import Input, Conv2D, Flatten
from keras import optimizers
from keras import initializations
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
import math

seed = 7
np.random.seed(seed)
OO = float("inf")
rows = 7
cols = 18
init = [5,16]
permgrid =  [['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'],  # 0
             ['*', '.', '$', '$', '$', '$', '.', '$', '.', '$', '$', '$', '$', '$', '$', '$', '$', '*'],  # 1
             ['*', '$', '*', '$', '*', '$', '*', '*', '.', '*', '*', '$', '$', '.', '*', '*', '$', '*'],  # 2
             ['*', '.', '.', '$', '*', '.', '*', '$', '$', '$', '$', '$', '*', '$', '$', '$', '$', '*'],  # 3
             ['*', '$', '*', '$', '*', '$', '*', '.', '*', '*', '*', '$', '*', '$', '$', '*', '$', '*'],  # 4
             ['*', '.', '$', '$', '.', '$', '$', '$', '$', '$', '$', '$', '$', '$', '$', '$', '.', '*'],  # 5
             ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']]  # 6
grid = deepcopy(permgrid)
batch_size = 20 # batch size
alpha = 0.2
epsilon = 0.5
gamma = 0.8
iterNo = 20
actionsNo= 4
ghosts_nb = 0
dx = [0,0,-1,1]
dy = [1,-1,0,0]
#              0   1   2   3     4
actionName = ['R','L','U','D']
ghosts = []

ccc = 0

def build_network():

    model = Sequential()
    model.add(Conv2D(nb_filter=4, nb_row=3, nb_col=3, activation='relu', dim_ordering='th',input_shape=(4, rows, cols)))
    #model.add(Dropout(0.5))
    model.add(Conv2D(nb_filter=32, nb_row=3, nb_col=3, activation='tanh'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(output_dim=256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(output_dim=actionsNo, activation='linear'))

    model.compile(loss='mse', optimizer=optimizers.adam(lr=0.01))
    
    return model

# returns next state results from executing some action in some state
def execAction(s, a):
    sPrime = [s[0] + dx[a], s[1] + dy[a]]
    return sPrime

# returns a list of all feasible, valid, action in some state
def genFeasibleActions(s):
    a = []
    for i in range(actionsNo):
        if not(s[0]+dx[i]<0 or s[0]+dx[i]>=rows or s[1]+dy[i]<0 or s[1]+dy[i]>=cols):
            if grid[s[0]+dx[i]][s[1]+dy[i]]!='*':
                a.append(i)
    return a
    
def preprocess_inp(s):
#    if ccc<10:
#        print 'state', s
#        print 'ghosts', ghosts    
#        print 'grid'    
#        for r in grid:
#            print r
#        print '->'*20
    walls, biscs, g_pos, a_pos= [np.zeros((rows, cols)) for i in range(4)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]=='*':
                walls[i][j]=1
            if grid[i][j]=='$':
                biscs[i][j]=1
    
    for i in range(ghosts_nb):
        g_pos[ghosts[i][0]][ghosts[i][1]]=1
    
    a_pos[s[0]][s[1]] = 1
    
    inp = [walls, biscs, g_pos, a_pos]

    return inp

# returns the value of the Q of some state-action pair
def getQ(s, a):
    #print '*getQ',s,' ', a     
    inp = preprocess_inp(s)
#    if ccc<10:
#        print 'input:\n'
#        for arr in inp:
#            print '*'*20
#            for r in arr:
#                print r
    ret = model.predict(np.array([inp]))[0][a]
    #print "predict:\n", 'inp:', [inp]
#    print 'getQ (',s,',',a,')'
#    print 'return', ret
    return ret

# returns the maximum Q over all valid actions from a given state
def argMaxQa(s):
    retA = 0
    maxQ = -OO
    A = genFeasibleActions(s)
#    print 'A:', A
    inp = preprocess_inp(s)
    qs =  model.predict(np.array([inp]))[0]
    #if ccc<10: 
#    print 'returns:', qs
#    print 'argMax', s
    for a in A:
#        print actionName[a], ' value:', qs[a]
        if qs[a]>maxQ:
            maxQ = qs[a]
            retA = a
#    print 'max', actionName[retA]
    return retA

# returns an action according to epsilon-greedy
def chooseAction(s):
    a = genFeasibleActions(s)
    r = np.random.rand()
    if r < epsilon:
        #print("Exploration")
        return random.choice(a)
    else:
        #print("Exploitation")
        return argMaxQa(s)

# returns the reward of some some state-action-nextState
def getReward(s, a, sPrime):
    if grid[sPrime[0]][sPrime[1]]=='$':
        return 10
    else:
        return -1


# updates the guess of the weights
def update_network(inputs, targets):
    new_targets = []
    #print 'targets_shape', targets.shape
    for s, t in zip(inputs, targets):        
        #print 's', s
        qs =  model.predict(np.array([s]), batch_size=1)[0]
        # qs [q(s,'l'), ......]
        qs[int(t[1])] = t[0]
        #print 't', t
        new_targets.append(qs)
        
    new_targets = np.array(new_targets)
    #print 'inputs_shape:', inputs.shape, ', new_targets_shape:', new_targets.shape
    model.fit(inputs, new_targets, nb_epoch=20, batch_size=batch_size, verbose=0)

 
# move the ghosts randomly
def moveGhosts():
    global ghosts
    for i in range(len(ghosts)):
        A = genFeasibleActions(ghosts[i])
        a = random.choice(A)
        ghosts[i][0] += dx[a]
        ghosts[i][1] += dy[a]
    

# counts the current number of the biscuits of the grid
def cntBiscuits():
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]=='$':
                cnt+=1
    return cnt

# executes the Q learning
def qLearning():
    err = 0
    avgReward = 0
    global alpha, ccc
    global epsilon
    global grid
    global ghosts
    cnt = 0
    targets = []
    inputs = []
    for it in range(iterNo):
        steps=0
        #alpha = max(alpha-0.0008, 0.00)
        epsilon = max(epsilon-0.05, 0.00)
        ghosts = [] #  2 ghosts
        
        grid = deepcopy(permgrid)
        bisc = cntBiscuits() #initial number of biscuits
        s = deepcopy(init)
        totalReward = 0
        while bisc>0: #loop until no biscuits are remaining
            #epsilon = max(epsilon-0.0002, 0.00)
            steps+=1            
            cnt += 1
#            print (str(cnt)+' ')*50
            xxx = 1
            ccc +=1
            
#            if ccc<1000: print '*'*5 + 'state' + '*'*5
            a = chooseAction(s)
#            if ccc<1000:
#                for r in grid:
#                    print r
#                print 'state:', s
#            print 'action:', actionName[a]
#                print 'ghosts:', ghosts
            sPrime = execAction(s, a)
       #     print 'not there!'
            # Collecting samples
            # input: batch of collected states (grid+ghosts+state+action)

            
            inp = preprocess_inp(s)
            inputs.append(inp)           
            
            qsa = getQ(s, a)
            
            # The next few lines check if the agent collides with a ghost
            v = []
            for i in range(len(ghosts)):
                if ghosts[i]==sPrime:
                    v.append(i)
            
            moveGhosts()
            #if it + 1 == 100:
                #print("Ghosts After:",ghosts)
            for i in range(len(ghosts)):
                if i in v and ghosts[i]==s:
                    xxx = 0

            # if the agent collides with a ghost, end the episode
            #print(" ,sPrime:", sPrime, end='')
            r = getReward(s, a, sPrime)
            if not xxx or (sPrime in ghosts):
                r = -100
                
            #print(" ,Reward:", r)
            
            totalReward += r
            
            if grid[sPrime[0]][sPrime[1]]=='$':
                #if it + 1 == iterNo:
                    #print("Ate a biscuit")
                grid[sPrime[0]][sPrime[1]] = '.'
                bisc-=1            
#
#            
#            if ccc<1000: 
#                print '*'*5 + 'next state' + '*'*5
#                print 'next state:', sPrime
#                print 'ghosts:', ghosts
#                
            aPMax = argMaxQa(sPrime)
            qsaPMax = getQ(sPrime, aPMax)
#            if ccc<10: print 'enough'
#            
            #if bisc==0 or not xxx or (sPrime in ghosts):
                #qsaPMax = 0
            
            # q learning update rule
#            q(s,a) = q(s,a) + alpha * (reward + gamma*argmaxa`q(s`,a`) - q(s,a))            
#            
#            # gradient descent general formula
#            a(n+1) =   a(n) - alpha * gradient(f(a(n)))          
#            
#            # the loss function we want to minimize
#            mean sqaured error loss = (target - predict)^2
#            gradient = -0.5 * (target - predict)
#            predict : q(s,a)
#            q(s,a)
#            
#            target : reward + gamma*argmaxa`q(s`,a`)
#            
#            q(s,a) = q(s,a) + learning_rate * 0.5 * (q(s,a) - (reward + gamma*argmaxa`q(s`,a`)))
#            
#            the input: state(grid, positions ghosts and the agent) the output: q(s,a) for every action           
            
#            if ccc>200:
#                assert (ccc==20)
            # Collecting samples
            # input: batch of collected states (grid+ghosts+state+action)
            # target: r + gamma * qsaPMax
            # it should be the reward if it's a terminal state
#            print r, '+', gamma , '*', qsaPMax
            targets.append([r+gamma*qsaPMax,a])
        
                
            #print cnt, ' ', batch_size
            # update the network
            if (cnt)%batch_size==0:
#                if ccc<1000: print 'updating net..' + '.'*40
                update_network(np.array(inputs), np.array(targets))
                inputs, targets=[],[]

            if not xxx or (sPrime in ghosts):
                #print("Collided with a ghost")
                break
            #print(w)
            s = deepcopy(sPrime)

        if bisc!=0:
            err+=1
        print("Iteration No.:", it+1)
        print('steps:', steps)
        #print("weights: ", end='')
        #print(weights)
        print("Bisuits remaining: ", bisc)
        print("Total Reward:", totalReward)
        
        avgReward += totalReward
    avgReward /= iterNo
    print("Average Reward:", avgReward)
    print("Success times:", iterNo-err)
    print("Success rate:", (iterNo-err)/iterNo)

model = build_network()

print "network build is OK"

qLearning()
