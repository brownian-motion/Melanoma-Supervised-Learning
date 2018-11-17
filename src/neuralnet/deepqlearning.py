
#Algorithm
    #Initialize replay memory D to capacity N
    #Initialize action-value function Q with random weights theta
    #Initialize action-value function Qhat with weights theta= theta-
    #For episode 1 = M, do
        #for sequence s1 = {x1} and preprocessed sequence theta(?)1 = theta(s1)(?)
        #for t = 1, do
            #with probability epsilon select a random action alpha t
            #otherwise alpha t  = argmax(alpha)Q((st):alpha;theta )

def init_replay_memory(memory,capacity):
    replay_memory = capacity
    return replay_memory

def init_q():
    #placeholder for randomized weights
    theta = 0

def init_qhat(theta):
    theta_neg = theta
