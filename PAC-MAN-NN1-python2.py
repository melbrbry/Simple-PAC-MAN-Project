import numpy as np
import random
from copy import deepcopy
from keras.models import Model, Sequential
from keras.layers.core import Activation, Dropout, Dense, Merge
from keras.layers import Input
from keras import optimizers
from keras import initializations
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
import math

seed = 7
np.random.seed(seed)
OO = int(1e18)
rows = 7
cols = 18
init = [5,16]
permgrid =  [['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'],  # 0
             ['*', '.', '$', '$', '$', '$', '.', '$', '.', '$', '$', '$', '$', '$', '$', '$', '$', '*'],  # 1
             ['*', '$', '*', '$', '*', '$', '*', '*', '.', '*', '*', '$', '$', '.', '*', '*', '$', '*'],  # 2
             ['*', '.', '.', '$', '*', '.', '*', '$', '$', '$', '$', '$', '*', '$', '$', '$', '$', '*'],  # 3
             ['*', '$', '*', '$', '*', '$', '*', '.', '*', '*', '*', '$', '*', '$', '$', '*', '$', '*'],  # 4
             ['*', '.', '$', '$', '.', '$', '$', '$', '$', '$', '$', '$', '$', '$', '$', '.', '.', '*'],  # 5
             ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']]  # 6
grid = deepcopy(permgrid)
batch_size = 20 # batch size
alpha = 0.2
epsilon = 1.0
gamma = 0.8
iterNo = 10
actionsNo= 4
ghosts_nb = 0
dx = [0,0,-1,1]
dy = [1,-1,0,0]
#              0   1   2   3 
actionName = ['R','L','U','D']
ghosts = []
inp_len = rows*cols + 2*ghosts_nb+3


def build_network():
    
    inp = Input(shape=(inp_len,))
    dense = Dense(32, activation='relu')(inp)
    drop = Dropout(0.2)(dense)
    dense2 = Dense(1)(drop)    
    model = Model(input=inp, output=dense2)        
    model.compile(loss='mse',optimizer=optimizers.SGD(lr=0.01))
    
    return model

# returns next state results from executing some action in some state
def execAction(s, a):
    sPrime = [s[0] + dx[a], s[1] + dy[a]]
    return sPrime

# returns a list of all feasible, valid, actions in some state
def genFeasibleActions(s):
    a = []
    for i in range(actionsNo):
        if not(s[0]+dx[i]<0 or s[0]+dx[i]>=rows or s[1]+dy[i]<0 or s[1]+dy[i]>=cols):
            if grid[s[0]+dx[i]][s[1]+dy[i]]!='*':
                a.append(i)
    return a

# preparing the input to the network
def preprocess_inp(s, a):
#    print 's:', s
#    print 'a:', a
#    print 'ghosts:', ghosts
#    print 'grid:'
#    for row in grid:
#        print row

    inp = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            inp.append(ord((grid[i][j])))
            
    state = [s[0],s[1],a]
    for x in ghosts:
        state.append(x[0])
        state.append(x[1])
    
    inp += state
    
    inp = [1.*x/60 for x in inp]      
    inp = preprocessing.scale(inp)
    
    return inp

# returns the value of the Q of some state-action pair
def getQ(s, a):
    #print '*getQ',s,' ', a 
    inp = preprocess_inp(s, a)
    ret = model.predict(np.array([inp]))[0][0]
    #print "predict:\n", 'inp:', [inp]
    #print 'return', ret
    return ret

# returns the maximum Q over all valid actions from a given state
def argMaxQa(s):
    retA = 0
    maxQ = -OO
    A = genFeasibleActions(s)
    #print 'A:', A
    for a in A:
        actionVal = getQ(s,a)
        if actionVal>maxQ:
            maxQ = actionVal
            retA = a
        #print 'a:', actionName[a], 'Q-val:', actionVal
    #print 'max', actionName[retA]
    return retA

# returns an action according to epsilon-greedy
def chooseAction(s):
    a = genFeasibleActions(s)
    r = random.random()
    if r < epsilon:
     #   print("Exploration")
        return random.choice(a)
    else:
      #  print("Exploitation")
        return argMaxQa(s)

# returns the reward of some some state-action-nextState
def getReward(s, a, sPrime):
    if grid[sPrime[0]][sPrime[1]]=='$':
        return 1
    else:
        return 0


# updates the guess of the weights
def update_network(inputs, targets):    
    model.fit(inputs, targets, nb_epoch=10, batch_size=batch_size, verbose=0)
 
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
    err = avgReward = cnt = 0
    global alpha, epsilon, grid, ghosts
    inputs, targets = [],[]
    
    for it in range(iterNo):
        #alpha = max(alpha-0.02, 0.00)
        
        ghosts = [] #  2 ghosts
        grid = deepcopy(permgrid)
        bisc = cntBiscuits() #initial number of biscuits
        s = deepcopy(init)
        totalReward = 0
        steps=0
        while not (bisc==0): #loop until no biscuits are remaining
            cnt += 1
            steps+=1
            epsilon = max(epsilon-0.001, 0.00)
            #print 'cnt', cnt
            xxx = 1
        
            a = chooseAction(s)
            
#            print("State:",s)
#            print("Ghosts:")
#            print(ghosts)
#            print("Action:", actionName[a])            
                
            sPrime = execAction(s, a)
            
            inp = preprocess_inp(s, a)
            #print inp
            inputs.append(inp)
            
            #print 'sPrime', sPrime
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

            r = getReward(s, a, sPrime)
            if not xxx or (sPrime in ghosts):
                r = -10
            #print(" ,Reward:", r)
            totalReward += r
            #qsa = getQ(s, a)
            
            if grid[sPrime[0]][sPrime[1]]=='$':
                #if it + 1 == iterNo:
                    #print("Ate a biscuit")
                grid[sPrime[0]][sPrime[1]] = '.'
                bisc-=1             
            
            aPMax = argMaxQa(sPrime)
            qsaPMax = getQ(sPrime, aPMax)


            # Collecting samples
            # input: batch of collected states (grid+ghosts+state+action)
            # target: r + gamma * qsaPMax
            
            
 #           if bisc==0 or not xxx or (sPrime in ghosts):
#                qsaPMax = 0            
            
            # it should only be the reward if it's a terminal state
            targets.append(r+gamma*qsaPMax)            

                            
            #print cnt, ' ', batch_size
            # update the network
            if (cnt)%batch_size==0:
                #print 'updating net..'
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
