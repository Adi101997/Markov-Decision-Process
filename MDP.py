#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np


# In[2]:


#Function for reading and extracting variables from input file
def extract_variables(file_input):
    text_lines=file_input.readlines()
    l=[]
    ip={}
    for line in text_lines:
        if line!='\n' and line[0]!='#':
            l.append(line.strip('\n'))
    for text in l:
        key,val=text.split(':')
        ip[key.strip()]=val.strip()
    return ip


# In[3]:


#Function for extracting values from variable
def variable_values(ip):
    size=tuple([int(matrix_size) for matrix_size in ip["size"].split()])[::-1]
    walls=[]
    terminal_s=[]
    t_probabilities=[]
    for t_s in ip['terminal_states'].split(','):
        terminal_s.append((size[0]-int(t_s.split()[1]), int(t_s.split()[0])-1,float(t_s.split()[2])))
    
    reward_nonTerminal = float(ip["reward"])
    for wall in ip["walls"].split(","):
        walls.append((size[0]-int(wall.split()[1]), int(wall.split()[0])-1))

    for probability in ip["transition_probabilities"].split():
        t_probabilities.append(float(probability))
    gamma = float(ip["discount_rate"])
    epsilon = float(ip["epsilon"])
    
    return size, walls, terminal_s, reward_nonTerminal, t_probabilities, gamma, epsilon


# In[4]:


#Class for utility
class mdp_utility():
    def __init__(self, size, walls, terminal_s):
        self.size=size
        self.walls=walls
        self.terminal_s=terminal_s
        vals=np.zeros(self.size)
        for wall in self.walls:
            vals[wall[0],wall[1]]=None
        self.vals=vals
    
    #Function for copying the values
    def copy_val(self,Un):
        self.vals=np.copy(Un.vals)
    
    #Getter function
    def get_val(self, state):
        x=state[0]
        y=state[1]
        return self.vals[x][y]
    
    #Setter function
    def put_val(self,state,v):
        x=state[0]
        y=state[1]
        self.vals[x][y]=v
    
    #Function for printing the grid matrix
    def grid_print(self, q, final_flag=False):
        if final_flag==True:
            print('\nFinal Value after Convergence:')  
        else:
            print('\nIteration:', q,)
        for row in self.vals:
            row=np.around(row,7)
            print(*row, sep=' ')
    


# In[5]:


#Class for the grid
class MDP_Grid:
    def __init__(self, size, walls, terminal_s, reward_nonTerminal, t_probabilities, gamma):
        self.size=size
        self.walls=walls
        self.terminal_s=terminal_s
        self.reward_nonTerminal=reward_nonTerminal
        self.t_probabilities=t_probabilities
        self.gamma=gamma
        self.states=[]
        self.grid=[]
        for i in range(0,self.size[0]):
            for j in range(0, self.size[1]):
                self.grid.append({"state":(i,j),"type":"Desirable",
                                  "reward":self.reward_nonTerminal, "val":0})
        self.grid=np.array(self.grid)
        self.grid=np.reshape(self.grid,(self.size[0], self.size[1]))
        for wall in self.walls:
            self.grid[wall[0]][wall[1]]["val"]=None
            self.grid[wall[0]][wall[1]]["type"]="wall"
            self.grid[wall[0]][wall[1]]["reward"]=None
        for terminal in self.terminal_s:
            self.grid[terminal[0]][terminal[1]]["val"]=None
            self.grid[terminal[0]][terminal[1]]["type"]="terminal"
            self.grid[terminal[0]][terminal[1]]["reward"]=terminal[2]
        
        self.actions_possible={'N':['N','E','W','S'],
                              'E':['E','S','N','W'],
                              'S':['S','E','W','N'],
                              'W':['W','S','N','E']}
        
    #Function for getting new states after transition   
    def transition(self,action,state):
        i,j=state
        if action=="N":
            if i>0 and (i-1,j) not in self.walls:
                next_i=i-1
                next_j=j
            else:
                next_i=i
                next_j=j
        if action=="S":
            if i<self.size[0]-1 and (i+1,j) not in self.walls:
                next_i=i+1
                next_j=j
            else:
                next_i=i
                next_j=j
        if action=="E":
            if j<self.size[1]-1 and (i,j+1) not in self.walls:
                next_i=i
                next_j=j+1
            else:
                next_i=i
                next_j=j
        if action=="W":
            if j>0 and (i,j-1) not in self.walls:
                next_i=i
                next_j=j-1
            else:
                next_i=i
                next_j=j
        next_state=self.grid[next_i][next_j]["state"]
        return next_state
    
    #Funtion for getting the reward
    def get_reward(self, s):
        for g in self.grid:
            for i in range(0,len(g)):
                if g[i]["state"]==s:
                    return g[i]["reward"]
    
    #Function for getting the Q values
    def QValue(self,a,state,U):
        util=[]
        transition_states=[]
        utility=0
        if a=='N':
            for act in self.actions_possible[a]:
                transition_states.append(self.transition(act,state))
            for prob,transitionState in zip(self.t_probabilities,transition_states):
                utility=utility + prob*(self.get_reward(transitionState)+
                                        self.gamma*(U.get_val(transitionState)))
            return (a,utility)
        if a=='S':
            for act in self.actions_possible[a]:
                transition_states.append(self.transition(act,state)) 
            for prob,transitionState in zip(self.t_probabilities,transition_states):
                utility=utility + prob*(self.get_reward(transitionState)+
                                        self.gamma*(U.get_val(transitionState)))
            return (a,utility)
        if a=='E':
            for act in self.actions_possible[a]:
                transition_states.append(self.transition(act,state))
            for prob,transitionState in zip(self.t_probabilities,transition_states):
                utility=utility + prob*(self.get_reward(transitionState)+
                                        self.gamma*(U.get_val(transitionState)))
            return (a,utility)
        if a=='W':
            for act in self.actions_possible[a]:
                transition_states.append(self.transition(act,state))
            for prob,transitionState in zip(self.t_probabilities,transition_states):
                utility=utility + prob*(self.get_reward(transitionState)+
                                        self.gamma*(U.get_val(transitionState)))
            return (a,utility)
    
    #Function for getting the policy
    def policy_mdp(self, U):
        pi = np.empty(self.size, str)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.grid[row][col]["type"]=="terminal":
                    pi[row][col]="T"
                if self.grid[row][col]["type"]=="wall":
                    pi[row][col]="-"
                if self.grid[row][col]["type"]=="Desirable":
                    q_vals=[]
                    for a in self.actions_possible.keys():
                        q_vals.append(self.QValue(a,self.grid[row][col]["state"],U))
                    pi[row][col]=sorted(q_vals, key = lambda x: x[1], reverse=True)[0][0]
        return pi
    
    def policy_eval(self, U, pi,):
        for c in range(30):
            for row in range(self.size[0]):
                for col in range(self.size[1]):
                    s=self.grid[row][col]
                    a=pi[row][col]
                    if s["type"]=="Desirable":
                        U.put_val(s["state"], self.QValue(a,s["state"],U)[1])
        return U
    
    #Function for value iteration
    def val_iteration(self, epsilon):
        U = mdp_utility(self.size, self.walls, self.terminal_s)
        U_n = mdp_utility(self.size, self.walls, self.terminal_s)
        iteration = 0
        while True:
            U.grid_print(iteration)
            delta=0
            for element in self.grid:
                for e in element:
                    if e["type"]=="Desirable":
                        q_vals=[]
                        for a in self.actions_possible.keys():
                            q_vals.append(self.QValue(a,e["state"],U)[1])
                        U_n.put_val(e["state"],max(q_vals))
                        if abs(U_n.get_val(e["state"]) - U.get_val(e["state"]))>delta:
                            delta=abs(U_n.get_val(e["state"]) - U.get_val(e["state"]))
            U.copy_val(U_n)
            iteration+=1
            if delta<=epsilon*(1-self.gamma)/self.gamma:
                U.grid_print(iteration, True)
                return U, self.policy_mdp(U)
            
    #Function policy iteration        
    def policy_iteration(self):
        U = mdp_utility(self.size, self.walls, self.terminal_s)
        pi = np.random.choice(list(self.actions_possible.keys()), self.size)
        for wall in self.walls:
            pi[wall[0],wall[1]]="-"
        for t_s in terminal_s:
            pi[t_s[0],t_s[1]]="T"
        while True:
            U=self.policy_eval(U, pi)
            unchanged=True
            for row in range(self.size[0]):
                for col in range(self.size[1]):
                    if self.grid[row][col]["type"]=="Desirable":
                        q_vals=[]
                        for a in self.actions_possible.keys():
                            q_vals.append(self.QValue(a,self.grid[row][col]["state"],U))
                        star_a=sorted(q_vals, key = lambda x: x[1], reverse=True)[0][0]
                        if self.QValue(star_a,self.grid[row][col]["state"],U)[1]>self.QValue(pi[row][col],self.grid[row][col]["state"],U)[1]:
                            pi[row][col]=star_a
                            unchanged=False
            if unchanged==True:
                break
        return pi     


# In[6]:


#Function for representing policy
def policy_representation_matrix(matrix):
    for i in range(matrix.shape[0]):
    #for j in range(policy_val_iteration.shape[1]):
        print(*matrix[i],sep='  ')
        print("\n") 


# In[7]:


#Calling the functions for extracting variables
#Change the path for input file according to your folder structure
file_input = open("C:/UIC_CS/AI/Homework/Assignment_7_grad/Assignment_7_grad/expected_output(1)/mdp_input_book.txt")
ip = extract_variables(file_input)
size, walls, terminal_s, reward_nonTerminal, t_probabilities, gamma, epsilon=variable_values(ip)


# In[8]:


mdp_grid = MDP_Grid(size, walls, terminal_s, reward_nonTerminal, t_probabilities, gamma)
print("############## Value Iteration ##############")
U, policy_val_iteration = mdp_grid.val_iteration(epsilon)
print('\nFinal Policy')
policy_representation_matrix(policy_val_iteration)
print("############# Policy Iteration #############")
pi = mdp_grid.policy_iteration()
policy_representation_matrix(pi)


# In[ ]:




