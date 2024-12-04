# searchAgents.py
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import networkx as ntx
import matplotlib.pyplot as plt
import util
import time
import search
import pacman
import random

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

class FreeMovements(Agent):
    "An agent that walks around."

    def getAction(self, state):
        max_movements = 100

        all_moves = {
            "1": Directions.WEST,
            "2": Directions.EAST,
            "3": Directions.NORTH,
            "4": Directions.SOUTH
        }

        for i in range(max_movements):
            randmove = random.randint(1,4)
            direction = all_moves[str(randmove)]
            "The agent receives a GameState (defined in pacman.py)."
            if direction in state.getLegalPacmanActions():
                return direction
            #else:
            #    return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        
        # Inicializa paredes, posição inicial e cantos
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = [(1, 1), (1, top), (right, 1), (right, top)]

        # Garante que todos os cantos são acessíveis
        for corner in self.corners:
            if startingGameState.hasWall(corner[0], corner[1]):
                raise Exception(f"Canto inacessível: {corner}")

        # Construir o grafo completo da grade
        full_graph = build_grid_graph(self.walls)

        # Seleciona os nós relevantes (posição inicial + cantos)
        key_nodes = [self.startingPosition] + list(self.corners)

        # Extrai o subgrafo relevante
        self.graph = build_corners_subgraph(full_graph,key_nodes)

        # Pre calcula o menor caminho entre os cantos
        self.shortest_paths = preprocess_all_pairs_paths(self.graph, self.corners)

        #visualize_grid_graph(self.graph)
                
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded

    def getStartState(self):
        """
        Returns the start state (Pacman's position and an empty set of visited corners).
        """
        return (self.startingPosition, tuple(False for _ in self.corners))

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state.
        """
        position, visited_corners = state
        return all(visited_corners)

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
        """
        successors = []
        position, visited_corners = state
        x, y = position

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if not self.walls[nextx][nexty]:  # Check if the next position is not a wall
                next_position = (nextx, nexty)
                next_visited_corners = list(visited_corners)

                # Mark the corner as visited if applicable
                if next_position in self.corners:
                    corner_index = self.corners.index(next_position)
                    next_visited_corners[corner_index] = True

                successors.append(((next_position, tuple(next_visited_corners)), action, 1))

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

def cornersHeuristic(state, problem):
    """
    Estima o custo restante para visitar todos os cantos não visitados.
    Armazena dinamicamente os caminhos calculados para otimizar futuras chamadas.

    Args:
        state: Um estado contendo a posição atual (x, y) e os cantos já visitados.
        problem: Uma instância do CornersProblem com o grafo e dados precomputados.

    Returns:
        Um valor heurístico estimando o custo restante para completar o problema.
    """
    position, visited_corners = state

    # Identifica os cantos que ainda não foram visitados
    unvisited_corners = [corner for corner, is_visited in zip(problem.corners, visited_corners) if not is_visited]

    # Se todos os cantos foram visitados, o custo estimado é 0
    if not unvisited_corners:
        return 0

    # Calcula a distância ao canto mais próximo
    nearest_corner_distance = float('inf')
    for corner in unvisited_corners:
        if (position, corner) not in problem.shortest_paths:
            # Calcula dinamicamente se necessário e armazena
            try:
                dist = ntx.shortest_path_length(problem.graph, source=position, target=corner, weight="weight")
                # Salvar nas duas direções
                problem.shortest_paths[(position, corner)] = dist
                problem.shortest_paths[(corner, position)] = dist
            except Exception:
                dist = float('inf')  # Define como infinito caso não haja caminho válido
        else:
            dist = problem.shortest_paths[(position, corner)]

        # Atualiza a menor distância encontrada
        nearest_corner_distance = min(nearest_corner_distance, dist)

    # Calcula o custo da MST para conectar os cantos não visitados
    if len(unvisited_corners) > 1:
        # Usa o subgrafo da grade, incluindo apenas os cantos não visitados
        subgraph = problem.graph.subgraph(unvisited_corners)
        mst_cost = ntx.minimum_spanning_tree(subgraph, weight="weight").size(weight="weight")
    else:
        mst_cost = 0

    # Retorna a soma da distância ao canto mais próximo e o custo da MST
    return nearest_corner_distance + mst_cost



    """
    Estima o custo restante para visitar todos os cantos não visitados.

    Usa:
    - Distância ao canto mais próximo.
    - Caminhos precomputados na sub-árvore para conectar os cantos restantes.

    Args:
        state: Um estado contendo a posição atual (x, y) e os cantos já visitados.
        problem: Uma instância do CornersProblem com os menores caminhos precomputados.

    Returns:
        Um valor heurístico (estimativa do custo restante).
    """

    # Extrai a posição atual e os cantos visitados do estado
    position, visited_corners = state

    # Identifica os cantos que ainda não foram visitados
    unvisited_corners = [corner for corner, is_visited in zip(problem.corners, visited_corners) if not is_visited]

    # Se todos os cantos foram visitados, o custo estimado é 0
    if not unvisited_corners:
        return 0

    # Calcula a distância até o canto mais próximo usando os caminhos precomputados
    nearest_corner_distance = min(
        problem.shortest_paths[(position, corner)] for corner in unvisited_corners
    )

    # Se houver mais de um canto não visitado, calcula o custo para conectá-los
    if len(unvisited_corners) > 1:
        # Soma os custos dos caminhos entre os cantos não visitados
        mst_cost = sum(
            problem.shortest_paths[(corner1, corner2)]
            for i, corner1 in enumerate(unvisited_corners)
            for corner2 in unvisited_corners[i + 1:]
        ) / 2  # Evita contar o caminho duas vezes
    else:
        # Apenas um canto não visitado, o custo adicional é zero
        mst_cost = 0

    # Retorna o valor heurístico como a soma da distância ao canto mais próximo e o custo restante
    return nearest_corner_distance + mst_cost

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your search may have a but our your heuristic is not admissible!  On the
    other hand, inadmissible heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    return 0

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

def build_grid_graph(walls):
    """
    Constrói um grafo a partir de uma grade (grid), onde cada célula livre é um nó 
    e arestas conectam células adjacentes que também são livres.

    Args:
        walls: Objeto que representa as paredes da grade. Pode ser acessado como walls[x][y],
               retornando True para uma parede e False para uma célula livre.

    Returns:
        graph: Um grafo NetworkX representando a grade, onde os nós são células livres
               e as arestas representam conexões entre células adjacentes.
    """
    # Inicializa um grafo vazio
    graph = ntx.Graph()

    # Obtém as dimensões da grade
    width, height = walls.width, walls.height

    # Itera sobre cada célula na grade
    for x in range(width):
        for y in range(height):
            if not walls[x][y]:  # Apenas considera células livres (sem paredes)
                # Adiciona um nó para a célula atual
                graph.add_node((x, y))

                # Verifica os vizinhos da célula atual (cima, baixo, esquerda, direita)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy  # Calcula a posição do vizinho
                    # Adiciona uma aresta se o vizinho está dentro da grade e é uma célula livre
                    if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                        graph.add_edge((x, y), (nx, ny))  # Conecta a célula atual ao vizinho

    # Retorna o grafo resultante
    return graph


def visualize_grid_graph(graph):
    """
    Visualiza um grafo em forma de grade usando matplotlib e NetworkX.
    
    Args:
        graph: Um objeto NetworkX representando o grafo a ser visualizado.
    """
    
    pos = {node: (node[1], -node[0]) for node in graph.nodes}  # Flip y for better visualization
    
    plt.figure(figsize=(8, 8))
    
    ntx.draw(graph, pos, node_size=10, with_labels=False, edge_color='gray', alpha=0.7)
    
    x_coords = [pos[node][0] for node in graph.nodes]
    y_coords = [pos[node][1] for node in graph.nodes]
    plt.xticks(range(min(x_coords), max(x_coords) + 1), fontsize=8)
    plt.yticks(range(min(y_coords), max(y_coords) + 1), fontsize=8)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.title("Grid Graph Visualization", fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def preprocess_all_pairs_paths(graph, nodes):
    """
    Precomputa os caminhos mais curtos entre todos os pares de nós fornecidos.

    Args:
        graph: Um objeto networkx.Graph que representa o grafo da grade.
        nodes: Lista como posição inicial e cantos para calcular os caminhos.

    Returns:
        Um dicionário onde as chaves são pares de nós (tuplas) e os valores são os comprimentos dos caminhos mais curtos entre esses nós.
    """
    shortest_paths = {}  # Dicionário para armazenar os caminhos mais curtos entre pares de nós.

    # Computa o caminho mais curto para todos os pares de nós na lista fornecida.
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):  # Para evitar redundância, considera pares únicos.
            node2 = nodes[j]  # Pega o segundo nó do par.
            # Usa o método algoritmo de dijkstra para calcular o comprimento do caminho mais curto entre node1 e node2.
            path_length = ntx.shortest_path_length(graph, source=node1, target=node2, weight="weight")

            # Armazena o comprimento do caminho mais curto no dicionário.
            # Adiciona em ambas as direções, pois o grafo é não-direcionado.
            shortest_paths[(node1, node2)] = path_length
            shortest_paths[(node2, node1)] = path_length
            
    return shortest_paths  # Retorna o dicionário de caminhos mais curtos.

def build_corners_subgraph(full_graph, corners):
    """
    Constrói um subgrafo contendo todos os menores caminhos entre os cantos.

    Args:
        full_graph: Grafo completo representando o labirinto.
        corners: Lista com os 4 cantos (tuplas de coordenadas).

    Returns:
        subgraph: Um subgrafo com os menores caminhos entre os cantos.
    """
    subgraph = ntx.Graph()

    # Calcula os menores caminhos entre todos os pares de cantos
    for i, corner1 in enumerate(corners):
        for corner2 in corners[i + 1:]:
            try:
                # Encontra o menor caminho entre os dois cantos
                path = ntx.shortest_path(full_graph, source=corner1, target=corner2, weight="weight")

                # Adiciona todos os nós e arestas do caminho ao subgrafo
                ntx.add_path(subgraph, path)
            except ntx.NetworkXNoPath:
                # Se não houver caminho, levanta uma exceção
                raise ValueError(f"Sem caminho entre {corner1} e {corner2} no grafo completo!")

    return subgraph

