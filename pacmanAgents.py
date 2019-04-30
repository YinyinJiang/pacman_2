# pacmanAgents.py
# ---------------
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


# Name: Yinyin Jiang
# Net ID: yj1438

from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        best_score = -100000000
        chooseAction = None
        possible = state.getAllPossibleActions()
        action_seq = []
        for i in range(0, 5):
            action_seq.append(possible[random.randint(0, len(possible) - 1)])

        while True:
            for i in range(0, len(action_seq)):
                if random.randint(0, 1) == 1:
                    action_seq[i] = possible[random.randint(0, len(possible) - 1)]
            tempState = state
            for i in range(0, len(action_seq)):
                if tempState is not None and tempState.isWin() + tempState.isLose() == 0:
                    tempState = tempState.generatePacmanSuccessor(action_seq[i])
                else:
                    break
            if tempState is None:
                break

            eval_score = gameEvaluation(state, tempState)
            if eval_score > best_score:
                best_score = eval_score
                chooseAction = action_seq[0]
        print best_score
        return chooseAction


class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        chooseAction = None
        best_score = -10000000
        possible = state.getAllPossibleActions()
        chromo_seq = self.generateChromosomes(possible)

        while True:
            chromo_scores, flag = self.rankSelection(state, chromo_seq)

            if not flag:
                break

            if chromo_scores[len(chromo_scores) - 1][1] > best_score:
                best_score = chromo_scores[len(chromo_scores) - 1][1]
                chooseAction = chromo_scores[len(chromo_scores) - 1][0][0]

            probabilities = self.generatepPobabilitySet(chromo_scores)
            parents = self.selectParents(probabilities, chromo_scores)
            # offspring become new generation
            chromo_seq = self.generateOffspring(parents, possible)
        return chooseAction

    def generateChromosomes(self, possible):
        chromo_seq = []
        for i in range(0, 8):
            action_seq = []
            for j in range(0, 5):
                action_seq.append(possible[random.randint(0, len(possible) - 1)])
            chromo_seq.append(action_seq)
        return chromo_seq

    def rankSelection(self, state, chromo_seq):
        bool = True
        scores = []
        for i in range(0, len(chromo_seq)):
            tempState = state
            for j in range(0, len(chromo_seq[i])):
                if tempState is not None and tempState.isWin() + tempState.isLose() == 0:
                    tempState = tempState.generatePacmanSuccessor(chromo_seq[i][j])
                else:
                    break
            if tempState is None:
                bool = False
                break
            scores.append(gameEvaluation(state, tempState))

        chromo_scores = zip(chromo_seq, scores)
        chromo_scores.sort(key=lambda x: x[1])
        return chromo_scores, bool

    def generatepPobabilitySet(self, chromo_scores):
        probability = []
        for i in range(1, len(chromo_scores) + 1):
            count = 0
            while count < i:
                probability.append(i)
                count = count + 1
        return probability

    def selectParents(self, probabilities, chromo_scores):
        parents = []
        while len(parents) < 8:
            pairs = []
            var = probabilities[random.randint(0, len(probabilities) - 1)]
            pairs.append(chromo_scores[var - 1][0])
            while len(pairs) < 2:
                var2 = probabilities[random.randint(0, len(probabilities) - 1)]
                if pairs[0] != chromo_scores[var2 - 1][0]:
                    pairs.append(chromo_scores[var2 - 1][0])
            parents.extend(pairs)
        return parents

    def generateOffspring(self, parents, possible):
        children = []
        # crossover
        for i in range(0, len(parents), 2):
            if random.randint(1, 10) > 7:
                children.append(parents[i])
                children.append(parents[i + 1])
            else:
                child1 = []
                child2 = []
                for j in range(0, len(parents[0])):
                    if random.randint(0, 1) == 0:
                        child1.append(parents[i][j])
                        child2.append(parents[i + 1][j])
                    else:
                        child1.append(parents[i + 1][j])
                        child2.append(parents[i][j])
                children.append(child1)
                children.append(child2)
        # mutation
        for k in range(0, len(children)):
            if random.randint(1, 10) == 1:
                children[k][random.randint(0, len(children[k]) - 1)] = possible[
                    random.randint(0, len(possible) - 1)]
        return children


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        legal = state.getLegalPacmanActions()

        # root
        v0 = Node(0, 0, [], [], None, legal)

        while True:
            v1 = self.treePolicy(v0)

            reward, flag2 = self.defaultPolicy(state, v1)
            if not flag2:
                break

            self.backUp(v1, reward)
        return self.mostVisitedChild(v0).action_seq[0]

    # selection
    def treePolicy(self, v0):
        while not (v0.children and v0.unvisited):
            if v0.unvisited:
                return self.expansion(v0)
            else:
                v0 = self.bestChild(v0, 1.0)
        return v0

    def expansion(self, v0):
        v1 = Node(0, 0, [], [], None, [])
        un_act = v0.unvisited[random.randint(0, len(v0.unvisited) - 1)]
        v0.unvisited.remove(un_act)
        v1.action_seq.append(un_act)
        v1.parent = v0
        v0.children.append(v1)
        return v1

    def bestChild(self, v, c):
        arg_max = 0
        best_child = v
        for i in range(0, len(v.children)):
            v1 = v.children[i]
            Q_v1 = v1.Q_reward
            N_v1 = v1.N_numOfVisit
            N_v = v.N_numOfVisit
            tmp_max = Q_v1 / N_v1 + c * math.sqrt(2 * math.log(N_v) / N_v1)
            if tmp_max > arg_max:
                arg_max = tmp_max
                best_child = v1
        return best_child

    # simulation
    def defaultPolicy(self, state, v1):
        next_state = state
        tempActionSeq = v1.action_seq

        for i in range(0, len(tempActionSeq)):
            if next_state is None:
                return None, False
            elif next_state.isWin():
                return 1000, True
            elif next_state.isLose():
                return -1000, True
            else:
                next_state = next_state.generatePacmanSuccessor(tempActionSeq[i])

        possible = next_state.getAllPossibleActions()
        for i in range(0, 5):
            ran_action = possible[random.randint(0, len(possible) - 1)]
            if next_state is None:
                return None, False
            elif next_state.isWin():
                return 1000, True
            elif next_state.isLose():
                return -1000, True
            else:
                next_state = next_state.generatePacmanSuccessor(ran_action)
        if next_state is None:
            return None, False
        return gameEvaluation(state, next_state), True

    def backUp(self, v, reward):
        while v is not None:
            v.N_numOfVisit = v.N_numOfVisit + 1
            v.Q_reward = v.Q_reward + reward
            v = v.parent

    def mostVisitedChild(self, v0):
        largest_N = 0
        res = []
        for i in range(0, len(v0.children)):
            if v0.children[i].N_numOfVisit > largest_N:
                largest_N = v0.children[i].N_numOfVisit
                res = []
                res.append(v0.children[i])
            elif v0.children[i].N_numOfVisit == largest_N:
                res.append(v0.children[i])
        return res[random.randint(0, len(res) - 1)]


class Node(object):
    def __init__(self, Q_reward, N_numOfVisit, action_seq, children, parent, unvisited):
        self.Q_reward = Q_reward
        self.N_numOfVisit = N_numOfVisit
        self.action_seq = action_seq
        self.children = children
        self.parent = parent
        self.unvisited = unvisited