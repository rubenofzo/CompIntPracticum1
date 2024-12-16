# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""


import util
from game import Directions
import sys


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(searchSpace: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    stack = util.Stack()
    startState = searchSpace.getStartState()
    # We store tuples in the stack to store the state with the path to the state
    stack.push((startState, []))  
    startState = searchSpace.getStartState()
    # we keep a set to track visited states 
    visited = set()  

    while not stack.isEmpty():
        currentNode, path = stack.pop()

        # Skip already visited states to prevent loops
        if currentNode in visited:
            continue  
    
        # Return the path if the goal state is reached
        if searchSpace.isGoalState(currentNode):
            return path  
        
        visited.add(currentNode)
        successors = searchSpace.getSuccessors(currentNode)
        for succ, action, _ in successors:
            # Append the action to the path and push state and path to stack
            stack.push((succ, path + [action]))  

    return None  # Return None if no solution is found

def breadthFirstSearch(searchSpace: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    startState = searchSpace.getStartState()
    queue.push(startState)

    # Dictionary which stores a state as a key 
    # and has vlaues with tuples of the predecesor of the state and the Action to get to this state 
    succesorDictionary = {startState : None}
    return BfsSearchLoop(queue, searchSpace, succesorDictionary, startState)

#Returns a path from the startstate to the goalstate
def BfsSearchLoop(queue, searchSpace: SearchProblem, succesorDictionary: dict, startState):
    # loop as long as there are still states to explore
    if queue.isEmpty(): 
        return None
    currentState = queue.pop() 

    # if the state we are exploring is the goal -> return path to goal
    if searchSpace.isGoalState(currentState):
       return returnPath(currentState, succesorDictionary, startState) 
    
    # otherwise we expand the node
    for succ,action,_ in searchSpace.getSuccessors(currentState): 
      #check if the entry has not already been seen
      if not succ in succesorDictionary: 
         #add tuple of predecessor and action to get to successor state with key successor 
         succesorDictionary.update({succ : (currentState,action)}) 
         #add the succesor state to the queue
         queue.push(succ) 
    #recurse
    return BfsSearchLoop(queue, searchSpace, succesorDictionary, startState)

#returns the path of actions from startstate to finalstate
def returnPath(finalState, succDictionary: dict, startState):
    path = []
    # begin with searching on the final state
    currentlySearching = succDictionary.get(finalState) 
    while not currentlySearching[0] == startState: # stop with searching if on the beginning state
        path.append(currentlySearching[1]) #add action to path
        currentlySearching = succDictionary.get(currentlySearching[0]) #search for the next entry in the dictionary
    path.append(currentlySearching[1]) # add the first step
    return path[::-1] # return the reversed path

def uniformCostSearch(searchSpace: SearchProblem):
    """Search the node of least total cost first."""
    priorityQueue = util.PriorityQueue()
    startState = searchSpace.getStartState()

    # we store tuples with state and cost of state into the Pqueue together with a sort value.
    # So like: ((State,CostValueOfState),SortValue) 
    # the Pqueue automaticaly sorts on this SortValue 
    priorityQueue.push((startState,1),0) 

    # Dictionary which stores a state as a key 
    # and has vlaues with tuples of the predecesor of the state and the Action to get to this state 
    succDictionary = {startState : None}

    return uniformCostSearchLoop(priorityQueue, searchSpace, succDictionary, startState)

def uniformCostSearchLoop(priorityQueue: util.PriorityQueue, searchSpace: SearchProblem, succDictionary: dict, startState):
    # loop as long as there are still nodes to explore
    if priorityQueue.isEmpty(): 
        return None
    currentState,currentCost = priorityQueue.pop()

    # if the state we are exploring is the goal -> return path to goal
    if searchSpace.isGoalState(currentState):
       return returnPath(currentState, succDictionary, startState) 
    
    # otherwise we expand the node
    for succ,action,cost in searchSpace.getSuccessors(currentState): 
       #the complete cost to get to this point in the path
       completeCost = cost + currentCost

       # add tuple of predecessor, action and cost to get to successor state with key successor this is useful later
       if succesorIsValid(succ, succDictionary, completeCost): 
          # add tuple of predecessor and action to get to successor state with key successor 
          succDictionary.update({succ : (currentState,action,completeCost)}) 
          # add the succesor state to the queue toghether with the updated cost
          priorityQueue.push((succ,completeCost),completeCost)
        
    #recurse 
    return uniformCostSearchLoop(priorityQueue, searchSpace, succDictionary, startState)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(searchSpace: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()
    startState = searchSpace.getStartState()

    # we store tuples with state and cost of state into the Pqueue together with the evaluationValue (cost + heuristic).
    # So like: ((State,CostValueOfState),evaluationValue) 
    # the Pqueue automaticaly sorts on the evaluationValue, giving it the desired a star behaviour
    priorityQueue.push((startState,1),evaluationFunction(startState, 0,heuristic, searchSpace))

    # Dictionary which stores a state as a key 
    # and has vlaues with tuples of the predecesor of the state and the Action to get to this state 
    succDictionary = {startState : None}
    return aStarLoop(priorityQueue, searchSpace, succDictionary, heuristic, startState)

def aStarLoop(priorityQueue: util.PriorityQueue, searchSpace: SearchProblem, succDictionary: dict, heuristic, startState):
    # loop as long as there are still nodes to explore
    while not priorityQueue.isEmpty():
        currentState,currentCost = priorityQueue.pop()

        # if the state we are exploring is the goal -> return path to goal
        if searchSpace.isGoalState(currentState):
            return returnPath(currentState, succDictionary, startState) 
        
        # otherwise we expand the node
        for succ,action,cost in searchSpace.getSuccessors(currentState): 
            completeCost = cost + currentCost #the complete cost to get to this point in the path
            
       #Check if the succesor is valid: if its not already in the dictionary or has been found with lower cost
            if succesorIsValid(succ, succDictionary, completeCost):
                #add tuple of predecessor, action and cost to get to successor state with key successor this is useful later
                succDictionary.update({succ : (currentState,action,completeCost)}) 

                #add the succesor state to the queue toghether with the updated heuristic/cost
                evaluationValue = evaluationFunction(succ, completeCost, heuristic, searchSpace) 
                priorityQueue.push((succ,completeCost),evaluationValue) 
    
    #if a end goal is never found return none
    return None

def evaluationFunction(state, cost, heuristic, searchSpace):
    return heuristic(state,searchSpace)+cost # cost is the heuristic value plus the cost to get in this state

# Function to check if a succcesor should be searched on
def succesorIsValid(succ, succDictionary,cost):
    # if the succesor has not been found before it is valid to search on
    if succ not in succDictionary: 
       return True 
    dicEntry = succDictionary.get(succ)
    if dicEntry is None: # this line makes sure that the start state is not added again
         return False
    _,_,c = dicEntry
    return c > cost     # if the succesor has a lower cost then the current entry it is valid to search on
   


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
