# Simple version of Pac-Man
# Brief Description:
# The agent has to eat as many biscuits as it can, and tries to avoid been eaten by the ghosts
# The agent has only one life, every biscuit of +1 rewards, eaten by ghosts -50 and 0 otherwise
# The agent have 5 actions: NoOp, Up, Down, Left, Right
# At each step the N, typically 4, ghosts take a random action between these 5.
# Solution:
# Q learning is used with linear gradient descent function approximation
# Three features describe any state-action pair:
# 1) the number of ghosts in or adjacent to the next state
# 2) if the next state will result in eating a biscuit
# 3) the distance to the nearest biscuit from the next state
# Notes:
# The solution is not meant to get the "optimal" solution, we are here satisfied with a "sub-optimal" one.
# The weights probably are not converging to some solution...
# To be continued..

import random
import queue
from copy import deepcopy
import numpy as np
OO = float("inf")
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
#         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
grid = deepcopy(permgrid)

mb = 20 # mini-batch size, if one -> SGD
alpha = 0.2
epsilon = 0.05
gamma = 0.8
iterNo = 500
ftNo = 4
actionsNo=4
dx = [0,0,-1,1]
dy = [1,-1,0,0]
actionName = ['R','L','U','D']
weights = [0.0, -2.0, 2.0, -2.0]
ghosts = []

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

# using BFS to return the nearest biscuit from a given cell (other than the cell itself)
def nearestBisc(s):
    #print("nearestBisc from", s)
    #for i in range(rows):
     #   print(grid[i])
    vis = [[0 for j in range(cols)] for i in range(rows)]
    q = queue.Queue()
    q.put([0, s])
    while not q.empty():
        cur = q.get()
        #print(cur)
        if cur[1]!=s and grid[cur[1][0]][cur[1][1]]=='$':
            #print(cur[0])
            return cur[0]
        vis[cur[1][0]][cur[1][1]] = 1
        #print("vis:", [cur[1][0],cur[1][1]])
        A = genFeasibleActions(cur[1])
        #print("Childs:")
        for a in A:
            sPrime = execAction(cur[1], a)
            if not vis[sPrime[0]][sPrime[1]]:
                #print(sPrime)
                q.put([cur[0]+1, sPrime])
    return -1

# returns a list of the values of the features assigned to some state-action pair
def getFeatures(s, a):
    f = [1.0]
    sPrime = execAction(s, a)
    # number of ghosts 1 step from me
    A = genFeasibleActions(sPrime)
    cnt=0
    for a1 in A:
        #print("sPrime", sPrime)
        #print("a1", actionName[a1])
        sDPrime = execAction(sPrime, a1)
        #print("sDPrime", sDPrime)
        if sDPrime in ghosts:
            cnt+=1
            #print("Ghost")
    if sPrime in ghosts:
        cnt+=1
    f.append(cnt)
    # if agent can eat a biscuit
    if not cnt and grid[sPrime[0]][sPrime[1]]=='$': #the bug was in this line (not cnt) wasn't there
        f.append(1)
    else:
        f.append(0)
    # nearest Biscuits
    dist = nearestBisc(sPrime)
    dist /= float(rows+cols+1)
    f.append(dist)
    # divide all by 10.0
    for i in range(ftNo):
        f[i] /= 10.0
    #print("f:", f)
    return f

# returns the value of the Q of some state-action pair
def getQ(s, a):
    ret = 0
    f = getFeatures(s, a)
    for i in range(ftNo):
        ret += weights[i]*f[i]
    return ret

# returns the maximum Q over all valid actions from a given state
def argMaxQa(s):
    retA = 0
    maxQ = -OO
    A = genFeasibleActions(s)
    for a in A:
        if getQ(s, a)>maxQ:
            maxQ = getQ(s, a)
            retA = a
    return retA

# returns an action according to epsilon-greedy
def chooseAction(s):
    a = genFeasibleActions(s)
    r = random.random()
    if r < epsilon:
        #print("Exploration")
        return random.choice(a)
    else:
        #print("Exploitation")
        return argMaxQa(s)

# returns the reward of some some state-action-nextState
def getReward(s, a, sPrime):
    if grid[sPrime[0]][sPrime[1]]=='$':
        return 1
    else:
        return 0

# updates the guess of the weights
def updateWeights(mbAcc):
    global weights
    for i in range(len(weights)):
        weights[i] += alpha*(1.0/mb)*mbAcc[i]

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
    mbAcc = [0, 0, 0, 0]
    err = 0
    avgReward = 0
    global alpha
    global epsilon
    global grid
    global ghosts
    cnt = 0
    steps = 0
    for it in range(iterNo):
        #alpha = max(alpha-0.0008, 0.00)
        #epsilon = max(epsilon-0.001, 0.00)
        # it seems that alpha=0.2 and epsilon=0.05 perform best
        # Improvement No.1
        # Ok, Let's reduce the learning rate gradually
        # As it has been shown that SGD shows better convergence behaviour, similar to batch GD, when we use this trick
        # Improvement No.2
        # Let's try to decrease the epsilon too gradually over time.
        # Because towards the end I know that I'm closer to the solution so I don't need to do stupid move in the name
        # of the exploration.
        # Improvement No.3
        # Ok... Let's try to update the weights in mini-batches method, instead of SGD.
        # We'll update our guess of the weights every 20 samples.
        
        prev_reward = prev_qsa = -1
        prev_features = []          
        
        #if it+1==25:
            #epsilon=0
            #alpha=0
        ghosts = [[3,3], [3,3]] #  2 ghosts
        #         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
        grid = deepcopy(permgrid)
        bisc = cntBiscuits() #initial number of biscuits
        s = []
        s.append(init[0])
        s.append(init[1])
        totalReward = 0
        while not (bisc==0): #loop until no biscuits are remaining
            xxx = 1
            steps+=1
            #if it+1 == 100:
                #for ii in range(rows):
                   #print(grid[ii])
                #print("Biscuits Number:", bisc)
            a = chooseAction(s)
            #if it+1 == 100:
                #print("State:",s)
                #print("Ghosts:", end='')
                #print(ghosts)
                #print("Action:", actionName[a])
                #print(weights)
            sPrime = execAction(s, a)

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
                r = -10
            #print(" ,Reward:", r)
            totalReward += r
            qsa = getQ(s, a)
            f = getFeatures(s, a)
            if grid[sPrime[0]][sPrime[1]]=='$':
                #if it + 1 == iterNo:
                    #print("Ate a biscuit")
                grid[sPrime[0]][sPrime[1]] = '.'
                bisc-=1
            #aPMax = argMaxQa(sPrime)
            #qsaPMax = getQ(sPrime, aPMax)
            
            # the end of the eposide
            end = 0
            if bisc==0 or not xxx or (sPrime in ghosts):
                #qsaPMax = 0
                end = 1
            
            start = 0            
            if len(prev_features)==0:
                start = 1
            
            if start!=1:
                for i in range(len(weights)):
                    mbAcc[i] += (prev_reward + gamma * qsa - prev_qsa) * prev_features[i]
                cnt+=1
                
                if end==1:
                    if (cnt)%mb==0:
                        updateWeights(mbAcc)
                        for i in range(len(weights)):
                            mbAcc[i] = 0
                    cnt+=1
                    for i in range(len(weights)):
                        mbAcc[i] += (r - qsa) * f[i]
                    
            if (cnt)%mb==0:
                updateWeights(mbAcc)
                for i in range(len(weights)):
                    mbAcc[i] = 0

            if not xxx or (sPrime in ghosts):
                #print("Collided with a ghost")
                break
            #print(w)
            s[0] = sPrime[0]
            s[1] = sPrime[1]
            
            prev_reward = r
            prev_qsa = qsa
            prev_features = deepcopy(f)
            

        if bisc!=0:
            err+=1
        #print("Iteration No.:", it+1)
        #print("weights: ", end='')
        #print(weights)
        #print("Bisuits remaining: ", bisc)
        #print("Total Reward:", totalReward)
        avgReward += totalReward
    avgReward /= iterNo
    print("Average Reward:", avgReward)
    print("Success times:", iterNo-err)
    print("Success rate:", (iterNo-err)/iterNo*100  )
    print("Average Steps:", steps/iterNo)


qLearning()