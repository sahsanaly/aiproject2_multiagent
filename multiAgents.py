# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # important variables for current and successor states
        score = successorGameState.getScore()
        newGhostPositions = successorGameState.getGhostPositions()
        currentNumFood = currentGameState.getNumFood()
        newNumFood = successorGameState.getNumFood()
        
        # variables for lists and bias variables
        closestFood = 0
        foodDistance = []
        nonZeroAdjustment = 1   # to eliminate math error for dividing by zero
        ghostNearby = 5         # distance if the ghost is nearby
        tooClose = 1            # distance if the ghost is too close to pacman

        foodList = newFood.asList()     # coordinates of food as a list
        
        # if there is food in the game state, include the food in the list, then
        # increase the reciprocal value(distance) of the closest food
        if len(foodList) > 0:
            for food in foodList:
                foodDistance.append(manhattanDistance(newPos, food))
            closestFood = min(foodDistance)
            score += 1 / closestFood
        
        # if the successor number of food is lesser than the current ones, then
        # increase the score by the reciprocal of the successor number of food
        if newNumFood < currentNumFood:
            if newNumFood > 0:
                score += 1 / newNumFood
        
        # For every ghost, calculate the successor ghost posiition
        # If the successor ghost is nearby (distance of 5), or too close (distance of 1), 
        # increase the reciprocal of the distance to the score
        # If the ghost is scared then increase the score by the reciprocal of the food number
        # And if the ghostdistance is too close when it is scared, increament the score as well, 
        # which allows the pacman to come close to the ghost only if it's scared
        for newGhostPosition in newGhostPositions:
            successorGhostDistance = manhattanDistance(newPos, newGhostPosition)
            
            if successorGhostDistance <= ghostNearby:
                score -= 1 / (successorGhostDistance + nonZeroAdjustment) 
            if successorGhostDistance <= tooClose:
                score -= 1 / (successorGhostDistance + nonZeroAdjustment) 
            for i in range(successorGameState.getNumAgents()-1):
                if newScaredTimes[i] > 0:
                    score += 1 / (newNumFood + nonZeroAdjustment)
                    if successorGhostDistance < newScaredTimes[i]:
                        score += 1 / (successorGhostDistance + nonZeroAdjustment)
            
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # returns the action state
        return self.minimax(gameState, 0, 0)[1]
    
    # This minimax function returns the score if player wins, loses or the depth
    # reached the self.depth, otherwise calls the max or min function depending on
    # which agent it is. Player's agent index is zero
    def minimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""
        
        if agentIndex == 0:
            result =  self.maxValue(gameState, depth, agentIndex)
        else:
            result = self.minValue(gameState, depth, agentIndex)

        return result
        
    # The max valuation function initializes the starting value to negative
    # infinity. It then compares all its successors value, which comes out of the min 
    # function and selects the maximum value and returns it along with its action/move
    def maxValue(self, gameState, depth, agentIndex):
        v, move  = float('-inf'), ""
        for a in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, a)
            v2 = self.minimax(successor, depth, agentIndex+1)[0]
            if v2 > v:
                v, move = v2, a
        return v, move
    
    # The min valuation function initializes the starting value to positive
    # infinity. It then compares all its successors value, which comes out of the max or
    # other min function and selects the minimum value and returns it along with its
    # action/move. It also checks if there is any more agent left in the cycle, if not 
    # then it increases the depth and and defines the agent index back to 0
    def minValue(self, gameState, depth, agentIndex):
        v, move = float('inf'), ""
        for a in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, a)
            successor_agentIndex = agentIndex + 1
            successor_depth = depth
            if successor_agentIndex == gameState.getNumAgents():
                successor_agentIndex = 0
                successor_depth += 1
            v2 = self.minimax(successor, successor_depth, successor_agentIndex)[0]
            if v2 < v:
                v, move = v2, a
        return v, move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        return self.alphaBeta(gameState, 0, 0, alpha, beta)[1]
    
    def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""
        
        if agentIndex == 0:
            result =  self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            result = self.minValue(gameState, depth, agentIndex, alpha, beta)

        return result
        
    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        v, move  = float('-inf'), ""
        for a in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, a)
            v2 = self.alphaBeta(successor, depth, agentIndex+1, alpha, beta)[0]
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v > beta:
                return v, move
        return v, move
    
    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        v, move = float('inf'), ""
        for a in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, a)
            successor_agentIndex = agentIndex + 1
            successor_depth = depth
            if successor_agentIndex == gameState.getNumAgents():
                successor_agentIndex = 0
                successor_depth += 1
            v2 = self.alphaBeta(successor, successor_depth, successor_agentIndex, alpha, beta)[0]
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v < alpha:
                return v, move
        return v, move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, 0)[1]
    
    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""
        
        if agentIndex == 0:
            result =  self.maxValue(gameState, depth, agentIndex)
        else:
            result = self.chanceValue(gameState, depth, agentIndex)

        return result
        
    def maxValue(self, gameState, depth, agentIndex):
        v, move  = float('-inf'), ""
        v2 = []
        for a in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, a)
            v2 = self.expectimax(successor, depth, agentIndex+1)[0]
            if v2 > v:
                v, move = v2, a
        return v, move
    
    def chanceValue(self, gameState, depth, agentIndex):
        v, move = float('inf'), ""
        v2 = []
        chanceNode = []
        for a in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, a)
            successor_agentIndex = agentIndex + 1
            successor_depth = depth
            if successor_agentIndex == gameState.getNumAgents():
                successor_agentIndex = 0
                successor_depth += 1
            v2.append(self.expectimax(successor, successor_depth, successor_agentIndex)[0])
            chanceNode.append(a)
        v = sum(v2) / len(v2)
        move = random.choice(chanceNode)
        return v, move

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    # This function is exactly the same as the evalution function. The only distance
    # difference is that it does not consider the actions and the successor states but 
    # only the current ones. The rest of the evaluation function is exactly the same
    
    currentPosition = currentGameState.getPacmanPosition()
    remainingFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()
    ghostPositions = currentGameState.getGhostPositions()
    closestFood = 0
    foodDistance = []
    rewardBias = 20
    nonZeroAdjustment = 1
    ghostNearby = 5
    tooClose = 1
    
    foodList = remainingFood.asList()
    if len(foodList) > 0:
        for food in foodList:
            foodDistance.append(manhattanDistance(currentPosition, food))
        closestFood = min(foodDistance)
        score += 1 / closestFood
    
    for ghostPosition in ghostPositions:
        ghostDistance = manhattanDistance(currentPosition, ghostPosition)
        
        if ghostDistance <= ghostNearby:
            score -= 1 / (ghostDistance + nonZeroAdjustment) 
        if ghostDistance <= tooClose:
            score -= 1 / (ghostDistance + nonZeroAdjustment) 
        for i in range(currentGameState.getNumAgents()-1):
            if scaredTimes[i] > 0:
                score += rewardBias
                if ghostDistance < scaredTimes[i]:
                    score += 1 / (ghostDistance + nonZeroAdjustment)
                    
    return score

# Abbreviation
better = betterEvaluationFunction
