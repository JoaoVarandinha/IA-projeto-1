#!/usr/bin/python3
# slitherlink.py: Template para implementação do projeto de Inteligência Artificial 2025/2026.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 112:
# 113425 Gustavo Pinhão da Costa Capucho
# 114010 João Manuel Varanda Gaspar

import random, copy
from sys import stdin
from collections import defaultdict

import utils
from utils import *

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class SlitherlinkState:
    state_id = 0

    def __init__(self, board, affected_cells=None, affected_vertexes=None):
        self.board = board
        self.id = SlitherlinkState.state_id
        SlitherlinkState.state_id += 1
        self.affected_cells = affected_cells
        self.affected_vertexes = affected_vertexes
    
    def __lt__(self, other):
        return self.id > other.id
    
    def __eq__(self, other):
        return isinstance(other, SlitherlinkState) and self.board.edges == other.board.edges
    
    def __hash__(self):
        return hash(frozenset(self.board.edges.items()))

class Board:
    """Representação interna de um tabuleiro de Slitherlink."""

    def __init__(self, clues:list, edges:dict=None):
        self.clues = clues
        self.rows = len(clues)
        self.columns = len(clues[0])

        # Initialize the board if no edges are passed (meaning we're in the initial state)
        if edges is None:
            self.edges = {}
            self._initialize_board()
        else:
            self.edges = edges

    def _initialize_board(self):
        # Get all edges to 0
        sides = ['up', 'right', 'down', 'left']
        for r in range(self.rows):
            for c in range(self.columns):
                for side in sides:
                    self.edges[(r, c, side)] = 0

        # Preprocess the board by filling in all forced plays
        self._preprocess_board()

    def _preprocess_board(self):
        self._preprocess_corners()
        for r in range(self.rows):
            for c in range(self.columns):
                if self.clues[r][c] == 0:
                    self._change_cell(r, c, 2)

                elif self.clues[r][c] == 3:
                    self._preprocess_threes(r, c)

    def _preprocess_corners(self):
        # Top left corner
        if self.clues[0][0] == 1:
            self.set_edge(0, 0, 'up', 2)
            self.set_edge(0, 0, 'left', 2)
        elif self.clues[0][0] == 2:
            self.set_edge(0, 1, 'up', 1)
            self.set_edge(1, 0, 'left', 1)
        elif self.clues[0][0] == 3:
            self.set_edge(0, 0, 'up', 1)
            self.set_edge(0, 0, 'left', 1)

        # Top right corner
        if self.clues[0][self.columns - 1] == 1:
            self.set_edge(0, self.columns - 1, 'up', 2)
            self.set_edge(0, self.columns - 1, 'right', 2)
        elif self.clues[0][self.columns - 1] == 2:
            self.set_edge(0, self.columns - 2, 'up', 1)
            self.set_edge(1, self.columns - 1, 'right', 1)
        elif self.clues[0][self.columns - 1] == 3:
            self.set_edge(0, self.columns - 1, 'up', 1)
            self.set_edge(0, self.columns - 1, 'right', 1)

        # Bottom right corner
        if self.clues[self.rows - 1][self.columns - 1] == 1:
            self.set_edge(self.rows - 1, self.columns - 1, 'right', 2)
            self.set_edge(self.rows - 1, self.columns - 1, 'down', 2)
        elif self.clues[self.rows - 1][self.columns - 1] == 2:
            self.set_edge(self.rows - 2, self.columns - 1, 'right', 1)
            self.set_edge(self.rows - 1, self.columns - 2, 'down', 1)
        elif self.clues[self.rows - 1][self.columns - 1] == 3:
            self.set_edge(self.rows - 1, self.columns - 1, 'right', 1)
            self.set_edge(self.rows - 1, self.columns - 1, 'down', 1)

        # Bottom left corner
        if self.clues[self.rows - 1][0] == 1:
            self.set_edge(self.rows - 1, 0, 'down', 2)
            self.set_edge(self.rows - 1, 0, 'left', 2)
        elif self.clues[self.rows - 1][0] == 2:
            self.set_edge(self.rows - 1, 1, 'down', 1)
            self.set_edge(self.rows - 2, 0, 'left', 1)
        elif self.clues[self.rows - 1][0] == 3:
            self.set_edge(self.rows - 1, 0, 'down', 1)
            self.set_edge(self.rows - 1, 0, 'left', 1)

    def _preprocess_threes(self, r:int, c:int):
        # Check for a 3 to the right of the cell, checking above is redundant because
        # other 3s will run this check anyway, same for all other following checks
        if c < self.columns - 1:
            if self.clues[r][c + 1] == 3:
                self.set_edge(r, c, 'left', 1)
                self.set_edge(r, c, 'right', 1)
                self.set_edge(r, c + 1, 'right', 1)
                if r > 0:
                    self.set_edge(r - 1, c, 'right', 2)
                if r < self.rows - 1:
                    self.set_edge(r + 1, c, 'right', 2)

        # Check for a 3 on the bottom right of the cell
        if r < self.rows - 1 and c < self.columns - 1:
            if self.clues[r + 1][c + 1] == 3:
                self.set_edge(r, c, 'up', 1)
                self.set_edge(r, c, 'left', 1)
                self.set_edge(r + 1, c + 1, 'right', 1)
                self.set_edge(r + 1, c + 1, 'down', 1)
        
        # Check for a 3 below the cell
        if r < self.rows - 1:
            if self.clues[r + 1][c] == 3:
                self.set_edge(r, c, 'up', 1)
                self.set_edge(r, c, 'down', 1)
                self.set_edge(r + 1, c, 'down', 1)
                if c > 0:
                    self.set_edge(r, c - 1, 'down', 2)
                if c < self.columns - 1:
                    self.set_edge(r, c + 1, 'down', 2)

        # Check for a 3 on the bottom left of the cell
        if r < self.rows - 1 and c > 0:
            if self.clues[r + 1][c - 1] == 3:
                self.set_edge(r, c, 'up', 1)
                self.set_edge(r, c, 'right', 1)
                self.set_edge(r + 1, c - 1, 'left', 1)
                self.set_edge(r + 1, c - 1, 'down', 1)

    def _get_other_entry(self, r:int, c:int, side:int) -> tuple:
        if side == 'up' and r > 0:
            return (r - 1, c, 'down')
        
        if side == 'right' and c < self.columns - 1:
            return (r, c + 1, 'left')
        
        if side == 'down' and r < self.rows - 1:
            return (r + 1, c, 'up')
        
        if side == 'left' and c > 0:
            return (r, c - 1, 'right')
        
        # In case there is no other entry for this edge we just return a set tuple that transmits that to other functions
        return (-1, -1, ' ')

    def _get_edge_vertexes(self, r:int, c:int, side:int) -> tuple:
        """Devolve as coordenadas dos 2 vértices que formam uma dada aresta, cada dado vértice é
        considerado como o vértice do canto superior esquerdo da sua célula (mesmo que esta não
        exista), assim as suas coordenadas são as coordenadas da sua célula e as coordenadas dos
        outros vértices são calculadas com base neste facto. Os valores destas coordenadas são
        valores de 0 a self.rows e 0 a self.columns inclusive."""
        if side == 'up':
            return (r, c), (r, c + 1)

        if side == 'right':
            return (r, c + 1), (r + 1, c + 1)

        if side == 'down':
            return (r + 1, c + 1), (r + 1, c)

        if side == 'left':
            return (r + 1, c), (r, c)

    def set_edge(self, r:int, c:int, side:str, status:int, affected_cells:set=None, affected_vertexes:set=None):
        # Ignore if edge has already been set
        if self.edges[(r, c, side)] != 0:
            return
        
        # Set the edge to the value defined by status
        self.edges[(r, c, side)] = status

        # Add the edges corresponding vertexes and cell to the affected sets to check in is_board_valid
        if affected_vertexes is not None:
            v1, v2 = self._get_edge_vertexes(r, c, side)
            affected_vertexes.add(v1)
            affected_vertexes.add(v2)
        if affected_cells is not None:
            affected_cells.add((r, c))

        # Since the dictionary has 2 entries for each edge we change the other entry too
        other_r, other_c, other_side = self._get_other_entry(r, c, side)
        if other_r != -1:
            self.edges[(other_r, other_c, other_side)] = status
            # Here we dont need to add the vertexes because they're the same as in the other entry
            if affected_cells is not None:
                affected_cells.add((other_r, other_c))
            

        # Get the edges matching the current status in this cell and the clue to see if we can fill in any forced plays
        status_edges = self.get_status_edges(r, c, status)
        clue = self.clues[r][c]
        if other_r != -1:
            other_status_edges = self.get_status_edges(other_r, other_c, status)
            other_clue = self.clues[other_r][other_c]

        if status == 1:
            if status_edges == clue or (clue == -1 and status_edges == 3):
                self._change_cell(r, c, 2, affected_cells, affected_vertexes)

            if other_r != -1 and (other_status_edges == other_clue or (other_clue == -1 and other_status_edges == 3)):
                self._change_cell(other_r, other_c, 2, affected_cells, affected_vertexes)

        elif status == 2:
            if status_edges == (4 - clue):
                self._change_cell(r, c, 1, affected_cells, affected_vertexes)

            if other_r != -1 and other_status_edges == (4 - other_clue):
                self._change_cell(other_r, other_c, 1, affected_cells, affected_vertexes)

        (v1_r, v1_c), (v2_r, v2_c) = self._get_edge_vertexes(r, c, side)
        self._propagate_vertexes(v1_r, v1_c, affected_cells, affected_vertexes)
        self._propagate_vertexes(v2_r, v2_c, affected_cells, affected_vertexes)

    def _propagate_vertexes(self, vr:int, vc:int, affected_cells:set=None, affected_vertexes:set=None):
        ones, zeros, available_edges = self._get_lines_at_vertex(vr, vc, True)

        if ones == 2 and zeros > 0:
            for r, c, side in available_edges:
                self.set_edge(r, c, side, 2, affected_cells, affected_vertexes)

        elif ones == 1 and zeros == 1:
            r, c, side = available_edges[0]
            self.set_edge(r, c, side, 1, affected_cells, affected_vertexes)

        elif ones == 0 and zeros == 1:
            r, c, side = available_edges[0]
            self.set_edge(r, c, side, 2, affected_cells, affected_vertexes)

        elif ones == 0 and zeros == 2:
            (r1, c1, side1), (r2, c2, side2) = available_edges
            other_r1, other_c1, _ = self._get_other_entry(r1, c1, side1)
            other_r2, other_c2, _ = self._get_other_entry(r2, c2, side2)
            if (r1, c1) == (r2, c2):
                (common_cell_r, common_cell_c) = (r1, c1)
            elif (r1, c1) == (other_r2, other_c2):
                (common_cell_r, common_cell_c) = (r1, c1)
            elif (other_r1, other_c1) == (r2, c2):
                (common_cell_r, common_cell_c) = (r2, c2)
            elif(other_r1, other_c1) == (other_r2, other_c2) and other_r1 != -1:
                (common_cell_r, common_cell_c) = (other_r1, other_c1)
            else:
                return
            
            if self.clues[common_cell_r][common_cell_c] == 1:
                self.set_edge(r1, c1, side1, 2, affected_cells, affected_vertexes)
                self.set_edge(r2, c2, side2, 2, affected_cells, affected_vertexes)

            elif self.clues[common_cell_r][common_cell_c] == 3:
                self.set_edge(r1, c1, side1, 1, affected_cells, affected_vertexes)
                self.set_edge(r2, c2, side2, 1, affected_cells, affected_vertexes)

            elif self.clues[common_cell_r][common_cell_c] == 2:
                # If any of these conditions are 0 we can't take conclusions, if it's 2 propagation
                # already happened, if it's any other value is_board_valid will kill the timeline
                if self.get_status_edges(common_cell_r, common_cell_c, 1) == 1:
                    self.set_edge(r1, c1, side1, 2, affected_cells, affected_vertexes)
                    self.set_edge(r2, c2, side2, 2, affected_cells, affected_vertexes)
                elif self.get_status_edges(common_cell_r, common_cell_c, 2) == 1:
                    self.set_edge(r1, c1, side1, 1, affected_cells, affected_vertexes)
                    self.set_edge(r2, c2, side2, 1, affected_cells, affected_vertexes)

    def _change_cell(self, r:int, c:int, status:int, affected_cells:set=None, affected_vertexes:set=None):
        self.set_edge(r, c, 'up', status, affected_cells, affected_vertexes)
        self.set_edge(r, c, 'right', status, affected_cells, affected_vertexes)
        self.set_edge(r, c, 'down', status, affected_cells, affected_vertexes)
        self.set_edge(r, c, 'left', status, affected_cells, affected_vertexes)

    def get_status_edges(self, r:int, c:int, status:int) -> int:
        return ((self.edges[(r, c, 'up')] == status) +
                (self.edges[(r, c, 'right')] == status) +
                (self.edges[(r, c, 'down')] == status) +
                (self.edges[(r, c, 'left')] == status))

    def is_board_valid(self, state:SlitherlinkState) -> bool:
        # If no cells or vertexes were affected then we're waiting on the AI to make it's first move
        # still and there's no need to check if the board is in a valid state
        if state.affected_cells is None and state.affected_vertexes is None:
            return True
        
        # Start by checking if the propagation didn't disrespect the clues
        for r, c in state.affected_cells:
            clue = self.clues[r][c]
            if clue != -1:
                if self.get_status_edges(r, c, 1) > clue or self.get_status_edges(r, c, 2) > (4 - clue):
                    return False
            else:
                if self.get_status_edges(r, c, 1) == 4:
                    return False
                
        # Check for branching paths or dead ends
        for vr, vc in state.affected_vertexes:
            ones, zeros = self._get_lines_at_vertex(vr, vc, False)
            if (ones == 1 and zeros == 0) or ones >= 3:
                return False
            
        return True

    def is_move_valid(self, r:int, c:int, side:str, status:int) -> bool:
        (v1_r, v1_c), (v2_r, v2_c) = self._get_edge_vertexes(r, c, side)
        ones1, zeros1 = self._get_lines_at_vertex(v1_r, v1_c, False)
        ones2, zeros2 = self._get_lines_at_vertex(v2_r, v2_c, False)

        # If there's already 2 lines around the vertex we reject the move because we don't want to put a 3rd line there
        if status == 1 and (ones1 >= 2 or ones2 >= 2):
            return False
        
        # If we already have a 1 connected to the vertex and theres only one edge left to fill we can't
        # put a 2 or we reach a dead end
        if status == 2 and ((ones1 == 1 and zeros1 == 1) or (ones2 == 1 and zeros2 == 1)):
            return False
        
        return True

    def _get_lines_at_vertex(self, vr:int, vc:int, get_indexes:bool) -> int:
        ones = zeros = 0
        indexes = []

        # Edge above the vertex
        if vr > 0 and vc < self.columns:
            if self.edges[(vr - 1, vc, 'left')] == 1:
                ones += 1
            elif self.edges[(vr - 1, vc, 'left')] == 0:
                zeros += 1
                indexes.append((vr - 1, vc, 'left'))
        elif vr > 0 and vc == self.columns:
            if self.edges[(vr - 1, vc - 1, 'right')] == 1:
                ones += 1
            elif self.edges[(vr - 1, vc - 1, 'right')] == 0:
                zeros += 1
                indexes.append((vr - 1, vc - 1, 'right'))

        # Edge to the right of the vertex
        if vc < self.columns and vr < self.rows:
            if self.edges[(vr, vc, 'up')] == 1:
                ones += 1
            elif self.edges[(vr, vc, 'up')] == 0:
                zeros += 1
                indexes.append((vr, vc, 'up'))
        elif vc < self.columns and vr == self.rows:
            if self.edges[(vr - 1, vc, 'down')] == 1:
                ones += 1
            elif self.edges[(vr - 1, vc, 'down')] == 0:
                zeros += 1
                indexes.append((vr - 1, vc, 'down'))

        # Edge below the vertex
        if vr < self.rows and vc < self.columns:
            if self.edges[(vr, vc, 'left')] == 1:
                ones += 1
            elif self.edges[(vr, vc, 'left')] == 0:
                zeros += 1
                indexes.append((vr, vc, 'left'))
        elif vr < self.rows and vc == self.columns:
            if self.edges[(vr, vc - 1, 'right')] == 1:
                ones += 1
            elif self.edges[(vr, vc - 1, 'right')] == 0:
                zeros += 1
                indexes.append((vr, vc - 1, 'right'))

        # Edge to the left of the vertex
        if vc > 0 and vr < self.rows:
            if self.edges[(vr, vc - 1, 'up')] == 1:
                ones += 1
            elif self.edges[(vr, vc - 1, 'up')] == 0:
                zeros += 1
                indexes.append((vr, vc - 1, 'up'))
        elif vc > 0 and vr == self.rows:
            if self.edges[(vr - 1, vc - 1, 'down')] == 1:
                ones += 1
            elif self.edges[(vr - 1, vc - 1, 'down')] == 0:
                zeros += 1
                indexes.append((vr - 1, vc - 1, 'down'))

        if get_indexes:
            return ones, zeros, indexes
        else:
            return ones, zeros

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        clues = []
        for line in stdin:
            line = line.strip('\r\n')
            part = line.split()

            if not part:
                continue

            row = [int(p) if p.isdigit() else -1 for p in part]
            clues.append(row)

        return Board(clues)

class Slitherlink(Problem):
    def __init__(self, board: Board, gui=None):
        """O construtor especifica o estado inicial."""
        state = SlitherlinkState(board)
        super().__init__(state)
        self.gui = gui
        self.action_order = self._build_action_order(board)

    def _build_action_order(self, board:Board):
        order = []
        seen_edges = set()

        # No need to go through clues with the number 0 as those are locked down during preprocessing
        for target in [3, 2, 1, -1]:
            for r in range(board.rows):
                for c in range(board.columns):
                    if board.clues[r][c] == target:
                        for side in ['up', 'right', 'down', 'left']:
                            v1, v2 = board._get_edge_vertexes(r, c, side)
                            true_edge = tuple(sorted((v1, v2)))
                            if true_edge not in seen_edges:
                                seen_edges.add(true_edge)
                                order.append((r, c, side))

        return order

    def actions(self, state: SlitherlinkState) -> list:
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        board = state.board
        if not board.is_board_valid(state):
            return []
        
        for r, c, side in self.action_order:
            if board.edges[(r, c, side)] == 0:
                valid_actions = []

                if board.is_move_valid(r, c, side, 2):
                    valid_actions.append((r, c, side, 2))

                if board.is_move_valid(r, c, side, 1):
                    valid_actions.append((r, c, side, 1))

                return valid_actions
        
        return []

    def result(self, state: SlitherlinkState, action) -> SlitherlinkState:
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, column, side, status = action

        new_edges = state.board.edges.copy()
        new_board = Board(state.board.clues, new_edges)

        affected_vertexes = set()
        affected_cells = set()

        new_board.set_edge(row, column, side, status, affected_cells, affected_vertexes)

        return SlitherlinkState(new_board, affected_cells, affected_vertexes)

    def goal_test(self, state: SlitherlinkState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        board = state.board
            
        if not board.is_board_valid(state):
            return False

        active_edges = set()
        for (r, c, side), status in board.edges.items():
            if status == 1:
                v1, v2 = board._get_edge_vertexes(r, c, side)
                active_edges.add(tuple(sorted((v1, v2))))

        start_edge = list(active_edges)[0]
        visited_edges = {start_edge}

        start_vertex = start_edge[0]
        current_vertex = start_edge[1]

        while current_vertex != start_vertex:
            moved = False
            for edge in active_edges:
                if edge not in visited_edges:
                    if edge[0] == current_vertex:
                        visited_edges.add(edge)
                        current_vertex = edge[1]
                        moved = True
                        break
                    elif edge[1] == current_vertex:
                        visited_edges.add(edge)
                        current_vertex = edge[0]
                        moved = True
                        break

            if not moved:
                return False

        return len(visited_edges) == len(active_edges)

if __name__ == "__main__":
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    initial_board = Board.parse_instance()

    problem = Slitherlink(initial_board)

    goal_node = depth_first_tree_search(problem)

    if goal_node is not None:
        winning_board = goal_node.state.board
        sides = ['up', 'right', 'down', 'left']

        for r in range(winning_board.rows):
            row_output = ""
            for c in range(winning_board.columns):
                for side in sides:
                    if winning_board.edges[(r, c, side)] == 1:
                        row_output += "1"
                    else:
                        row_output += "0"

                if c != winning_board.columns - 1:
                    row_output += "\t"

            print(row_output)

    else:
        print("Goal not found")
