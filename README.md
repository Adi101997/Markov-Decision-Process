# Markov-Decision-Process
In this project we implement Markov Decision Process for a stochastic environment

- MDP problem for a generic grid world (refer to figure 17.1 of AIMA Textbook 4th edition page 563) and use value iteration (figure 17.6) to print the values of states in each iteration. 
- After termination of the value iteration, the final policy id returned.
- The description of MDP (T, R, gamma, and epsilon) is loaded from the text file as input.

# Instructions

1) mdp_input.txt file has been used for input.  
2) Save the code in the same directory as that of input file.

## Format followed for input in the file is:
#size of the gridworld
size : 5 4

#list of location of walls
walls : 2 2 , 2 3

#list of terminal states (row,column,reward)
terminal_states : 5 3 -3 , 5 4 +2, 4 2 +1

#reward in non-terminal states
reward : -0.04

#transition probabilites
transition_probabilities : 0.8 0.1 0.1 0

discount_rate : 0.85

epsilon : 0.001



