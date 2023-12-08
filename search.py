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
    newStack = util.Stack()
    newStack.push(problem.getStartState())
    return depthFirstSearchHelper(newStack, problem)

def depthFirstSearchHelper(stack: util.Stack, problem: SearchProblem):
    if stack.isEmpty:
        util.raiseNotDefined
    else: 
        t = stack.pop
        if problem.isGoalState(t):
            return t
        else:
            for succ in problem.getSuccesors(t):
                stack = stack.push(succ)
            return depthFirstSearchHelper(stack, problem)



def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    startState = problem.getStartState()
    queue.push(startState)
    succDictionary = {} #Dictionary which stores the State and Action to get to this state
    succDictionary.update({startState : None})
    return breadthFirstSearchHelper(queue, problem, succDictionary, startState)

def breadthFirstSearchHelper(queue: util.Queue, problem: SearchProblem, succDictionary: dict, startState):
    # loop as long as there are still states to explore
    if not queue.isEmpty(): 
        currentState = queue.pop() 
         # if the state we are exploring is the goal -> return path to goal
        if problem.isGoalState(currentState):
            return returnPath(currentState, succDictionary, startState) #[Directions.WEST]
        # else we expand the node
        successors = problem.getSuccessors(currentState)
        for succ,actions,costs in successors: 
          if succDictionary.get(succ)==None: #If the entry is not already in the dictionary
            #add tuple of predisseccor and action to get to succesor state with key succesor state
            print((currentState,actions))
            succDictionary.update({succ : (currentState,actions)}) 
            queue.push(succ) #add the succesor state to the queue
        return breadthFirstSearchHelper(queue, problem, succDictionary, startState)
    return None

def returnPath(finalState, succDictionary: dict, startState):
    path = []
    currentlySearching = succDictionary.get(finalState) # begin with searching on the final state
    while not currentlySearching[0] == startState: # stop with searching if on the beginning state which has action == NONE
        path.append(currentlySearching[1]) #add action to path
        currentlySearching = succDictionary.get(currentlySearching[0]) #search for the next entry in the dictionary
        print(currentlySearching)
    path.append(currentlySearching[1]) # add the first step
    return path[::-1] # return the reversed path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
