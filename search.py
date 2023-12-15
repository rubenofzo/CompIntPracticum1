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

def depthFirstSearch(problem: SearchProblem):
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
    startState = problem.getStartState()
    stack.push((startState, []))  # Using a tuple to store state and its path

    visited = set()  # To keep track of visited states

    while not stack.isEmpty():
        currentNode, path = stack.pop()

        if currentState in visited:
            continue  # Skip already visited states to prevent loops

        visited.add(currentNode)

        if problem.isGoalState(currentNode):
            return path  # Return the path if the goal state is reached

        successors = problem.getSuccessors(currentNode)
        for succ, action, _ in successors:
            stack.push((succ, path + [action]))  # Append the action to the path

    return None  # Return None if no solution is found

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    startState = problem.getStartState()
    queue.push(startState)
    succDictionary = {startState : None} #Dictionary which stores the State and Action to get to this state
    return StandardSearchLoop(queue, problem, succDictionary, startState)

#Standard search loop which can be used for bfs(queue) and dfs(stack)
#Returns a path from the startstate to the goalstate (fastest if bfs is used)
def StandardSearchLoop(queue, problem: SearchProblem, succDictionary: dict, startState):
    # loop as long as there are still states to explore
    if queue.isEmpty(): 
        return None
    currentState = queue.pop() 
    # if the state we are exploring is the goal -> return path to goal
    if problem.isGoalState(currentState):
       return returnPath(currentState, succDictionary, startState) 
    # else we expand the node
    for succ,action,_ in problem.getSuccessors(currentState): 
      if not succ in succDictionary: #If the entry is not already in the dictionary
         #add tuple of predecessor and action to get to successor state with key successor 
         succDictionary.update({succ : (currentState,action)}) 
         queue.push(succ) #add the succesor state to the queue
    return StandardSearchLoop(queue, problem, succDictionary, startState)

def returnPath(finalState, succDictionary: dict, startState):
    path = []
    currentlySearching = succDictionary.get(finalState) # begin with searching on the final state
    while not currentlySearching[0] == startState: # stop with searching if on the beginning state which has action == NONE
        path.append(currentlySearching[1]) #add action to path
        currentlySearching = succDictionary.get(currentlySearching[0]) #search for the next entry in the dictionary
    path.append(currentlySearching[1]) # add the first step
    return path[::-1] # return the reversed path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    startState = problem.getStartState()
    pqueue.push(startState, 0) #we store tuples with state and depth of state into the Pqueue
    succDictionary = {startState : None} #Dictionary which stores the State and Action to get to this state
    return uniformCostSearchHelp(pqueue, problem, succDictionary, startState)

def uniformCostSearchHelp(pqueue: util.PriorityQueue, problem: SearchProblem, succDictionary: dict, startState):
     # loop as long as there are still nodes to explore
    if pqueue.isEmpty(): 
        return None
    currentState = pqueue.pop() 
    # if the state we are exploring is the goal -> return path to goal
    if problem.isGoalState(currentState):
       #print("path found",currentState)
       return returnPath(currentState, succDictionary, startState) 
    # else we expand the node
    for succ,action,cost in problem.getSuccessors(currentState): 
       if succesorIsValid(succ, succDictionary, cost): #If the entry is not already in the dictionary
          #add tuple of predecessor and action to get to successor state with key successor 
          succDictionary.update({succ : (currentState,action,cost)}) 
          #add the succesor state to the queue toghether with the updated cost
          pqueue.push(succ,cost) 
    return uniformCostSearchHelp(pqueue, problem, succDictionary, startState)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    priorityQueue.push((startState,1),evaluationFunction(startState, 0,heuristic, problem)) #we store tuples with state and cost of state into the Pqueue
    succDictionary = {startState : None} #Dictionary which stores the State and Action to get to this state
    return aStarLoop(priorityQueue, problem, succDictionary, heuristic, startState)

def aStarLoop(priorityQueue: util.PriorityQueue, problem: SearchProblem, succDictionary: dict, heuristic, startState):\
    # loop as long as there are still nodes to explore
    if priorityQueue.isEmpty(): 
        return None
    currentState,currentCost = priorityQueue.pop()
    # if the state we are exploring is the goal -> return path to goal
    if problem.isGoalState(currentState):
        return returnPath(currentState, succDictionary, startState) 
    # else we expand the node
    for succ,action,cost in problem.getSuccessors(currentState): 
        completeCost = cost + currentCost #the complete cost to get to this point in the path
        #If the succesor is not already in the dictionary or if the succesor has lower costs then
        if succesorIsValid(succ, succDictionary, completeCost):
            #add tuple of predecessor and action to get to successor state with key successor 
            succDictionary.update({succ : (currentState,action,completeCost)}) 
            # heuristic value is determined
            evaluationValue = evaluationFunction(succ, completeCost, heuristic, problem) 
            #add the succesor state to the queue toghether with the updated cost
            priorityQueue.push((succ,completeCost),evaluationValue) 
    return aStarLoop(priorityQueue, problem, succDictionary, heuristic, startState)

def evaluationFunction(state, cost, heuristic, problem):
    return heuristic(state,problem)+cost # cost is the heuristic cost plus the cost to get in this state

# Function to check if a succcesor should be searched on
def succesorIsValid(succ, succDictionary,cost):
    # if the succesor had not been found before it is valid to search on
    if succ not in succDictionary: 
       return True 
    dicEntry = succDictionary.get(succ)
    if dicEntry is not None: # this line makes sure that the start state is not added again
      _,_,c = dicEntry
      return c > cost     # if the succesor has a lower cost then the current entry it is valid to search on
    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
