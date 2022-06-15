import pygame
import sys
from pygame.locals import *
import random
from copy import deepcopy
import math
from time import sleep
pygame.font.init()


##COLORS##
#             R    G    B
WHITE = (255, 255, 255)
BLUE = (0,   0, 255)
RED = (255,   0,   0)
BLACK = (0,   0,   0)
GOLD = (255, 215,   0)
HIGH = (160, 190, 255)

##DIRECTIONS##
NORTHWEST = "northwest"
NORTHEAST = "northeast"
SOUTHWEST = "southwest"
SOUTHEAST = "southeast"


class Bot:
    def __init__(self, game, color, mid_eval=None, end_eval=None, weights0=None, weights1=None, weights2=None, phases=3):
        if mid_eval == 'piece2val':
            self._mid_eval = self._piece2val
        elif mid_eval == 'piece_and_board':
            self._mid_eval = self._piece_and_board2val
        elif mid_eval == 'piece_and_row':
            self._mid_eval = self._piece_and_row2val
        elif mid_eval == 'piece_and_board_pov':
            self._mid_eval = self._piece_and_board_pov2val
        if end_eval == 'sum_of_dist':
            self._end_eval = self._sum_of_dist
        elif end_eval == 'farthest_piece':
            self._end_eval = self._farthest_piece
        else:
            self._end_eval = None
        self.game = game
        self.color = color
        self.eval_color = color
        if self.color == BLUE:
            self.adversary_color = RED
        else:
            self.adversary_color = BLUE
        self._current_eval = self._mid_eval
        self._end_eval_time = False
        self._count_nodes = 0
        self.weights_phase0 = weights0
        self.weights_phase1 = weights1
        self.weights_phase2 = weights2
        self.phase = 0
        self.phases = phases
        if weights0 is None:
            self.weights_phase0 = {'nr_enemy_pawns' : -1,'nr_enemy_kings' : -1,'nr_safe_pawns' : 1,'nr_safe_kings' : 1, 'dis_friendly_promotion' : -1,"nr_movable_friendly_pawns" : 1,"nr_movable_friendly_kings" : 1, 'num_defended_pieces' : 1,"nr_lower_pieces":1,"nr_middle_pawns":1,"nr_middle_kings":1,"nr_higher_pawns":1,"nr_diagonal_pawns":1,"nr_diagonal_kings":1}
        if weights1 is None:
            self.weights_phase1 = {'nr_enemy_pawns' : -1,'nr_enemy_kings' : -1,'nr_safe_pawns' : 1,'nr_safe_kings' : 1, 'dis_friendly_promotion' : -1,"nr_movable_friendly_pawns" : 1,"nr_movable_friendly_kings" : 1, 'num_defended_pieces' : 1,"nr_lower_pieces":1,"nr_middle_pawns":1,"nr_middle_kings":1,"nr_higher_pawns":1,"nr_diagonal_pawns":1,"nr_diagonal_kings":1}
        if weights2 is None:
            self.weights_phase2 = {'nr_enemy_pawns' : -1,'nr_enemy_kings' : -1,'nr_safe_pawns' : 1,'nr_safe_kings' : 1, 'dis_friendly_promotion' : -1,"nr_movable_friendly_pawns" : 1,"nr_movable_friendly_kings" : 1, 'num_defended_pieces' : 1,"nr_lower_pieces":1,"nr_middle_pawns":1,"nr_middle_kings":1,"nr_higher_pawns":1,"nr_diagonal_pawns":1,"nr_diagonal_kings":1}

    def step(self, board, return_count_nodes=False):
        self._count_nodes = 0
        if(self._end_eval is not None and self._end_eval_time == False):
            if self._all_kings(board):
                #print('END EVAL is on')
                self._end_eval_time = True
                self._current_eval = self._end_eval

        self._evo_step(board)

        if return_count_nodes:
            return self._count_nodes

    def _action(self, current_pos, final_pos, board):
        if current_pos is None:
            self.game.end_turn()
            # board.repr_matrix()
            # print(self._generate_all_possible_moves(board))
        # print(current_pos, final_pos, board.location(current_pos[0], current_pos[1]).occupant)
        if self.game.hop == False:
            if final_pos != None and board.location(final_pos[0], final_pos[1]).occupant != None:
                if board.location(final_pos[0], final_pos[1]).occupant.color == self.game.turn:
                    current_pos = final_pos

            elif current_pos != None and final_pos in board.legal_moves(current_pos[0], current_pos[1]):

                board.move_piece(
                    current_pos[0], current_pos[1], final_pos[0], final_pos[1])

                if final_pos not in board.adjacent(current_pos[0], current_pos[1]):
                    board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) //
                                    2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)

                    self.game.hop = True
                    current_pos = final_pos
                    final_pos = board.legal_moves(
                        current_pos[0], current_pos[1], True)
                    if final_pos != []:
                        # print("HOP in Action", current_pos, final_pos)
                        self._action(current_pos, final_pos[0], board)
                    self.game.end_turn()

        if self.game.hop == True:
            if current_pos != None and final_pos in board.legal_moves(current_pos[0], current_pos[1], self.game.hop):
                board.move_piece(
                    current_pos[0], current_pos[1], final_pos[0], final_pos[1])
                board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) //
                                   2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)

            if board.legal_moves(final_pos[0], final_pos[1], self.game.hop) == []:
                self.game.end_turn()
            else:
                current_pos = final_pos
                final_pos = board.legal_moves(
                    current_pos[0], current_pos[1], True)
                if final_pos != []:
                    # print("HOP in Action", current_pos, final_pos)
                    self._action(current_pos, final_pos[0], board)
                self.game.end_turn()
        if self.game.hop != True:
            self.game.turn = self.adversary_color

    def _action_on_board(self, board, current_pos, final_pos, hop=False):
        if hop == False:
            if board.location(final_pos[0], final_pos[1]).occupant != None and board.location(final_pos[0], final_pos[1]).occupant.color == self.game.turn:
                current_pos = final_pos

            elif current_pos != None and final_pos in board.legal_moves(current_pos[0], current_pos[1]):

                board.move_piece(
                    current_pos[0], current_pos[1], final_pos[0], final_pos[1])

                if final_pos not in board.adjacent(current_pos[0], current_pos[1]):
                    # print("REMOVE", current_pos, final_pos)
                    board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) //
                                       2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)
                    hop = True
                    current_pos = final_pos
                    final_pos = board.legal_moves(current_pos[0], current_pos[1], True)
                    if final_pos != []:
                        # print("HOP in Action", current_pos, final_pos)
                        self._action_on_board(board, current_pos, final_pos[0],hop=True)
        else:
            # print(current_pos, final_pos)
            if current_pos != None and final_pos in board.legal_moves(current_pos[0], current_pos[1], hop):
                board.move_piece(current_pos[0], current_pos[1], final_pos[0], final_pos[1])
                board.remove_piece(current_pos[0] + (final_pos[0] - current_pos[0]) // 2, current_pos[1] + (final_pos[1] - current_pos[1]) // 2)

            if board.legal_moves(final_pos[0], final_pos[1], self.game.hop) == []:
                return
            else:
                current_pos = final_pos
                final_pos = board.legal_moves(current_pos[0], current_pos[1], True)
                if final_pos != []:
                    # print("HOP in Action", current_pos, final_pos)
                    self._action_on_board(board, current_pos, final_pos[0],hop=True)

    def _generate_move(self, board):
        for i in range(8):
            for j in range(8):
                if(board.legal_moves(i, j, self.game.hop) != [] and board.location(i, j).occupant != None and board.location(i, j).occupant.color == self.game.turn):
                    yield (i, j, board.legal_moves(i, j, self.game.hop))

    def _generate_all_possible_moves(self, board):
        possible_moves = []
        for i in range(8):
            for j in range(8):
                if(board.legal_moves(i, j, self.game.hop) != [] and board.location(i, j).occupant != None and board.location(i, j).occupant.color == self.game.turn):
                    possible_moves.append(
                        (i, j, board.legal_moves(i, j, self.game.hop)))
        return possible_moves

    def _evo_step(self,board):
        if (self.phase != 2 and self.phases == 3) or (self.phase != 1 and self.phases == 2):
            detect_game_phase = self._detect_game_phase(board)
        move, choice, _ = self._evo(board)
        self._action(move, choice, board)

        return

    def _evo(self, board):
        max_value = -float("inf")
        best_pos = None
        best_action = None
        for pos in self._generate_move(board):
            for action in pos[2]:
                        board_clone = deepcopy(board)
                        self.color, self.adversary_color = self.adversary_color, self.color
                        self.game.turn = self.color
                        self._action_on_board(board_clone, pos, action)
                        self._count_nodes += 1
                        step_score = self._calculate_features_score(board_clone)
                        self.color, self.adversary_color = self.adversary_color, self.color
                        self.game.turn = self.color

                        if step_score > max_value:
                            max_value = step_score
                            best_pos = pos
                            best_action = (action[0], action[1])
                        elif step_score == max_value and random.random() <= 0.5:
                            max_value = step_score
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
                        if(step_score == -float("inf") and best_pos is  None):
                            best_pos = (pos[0], pos[1])
                            best_action = (action[0], action[1])
        return best_pos, best_action, max_value

    def _calculate_features_score(self, board):
        # Calculate feature score
        score = 0
        nr_enemy_pawns = self._nr_enemy_pawns(board)
        nr_enemy_kings = self._nr_enemy_pawns(board)
        nr_safe_pawns = self._nr_safe_pawns(board)
        nr_safe_kings = self._nr_safe_kings(board)
        dis_friendly_promotion = self._dis_friendly_promotion(board)
        nr_movable_friendly_pawns = self._nr_movable_friendly_pawns(board)
        nr_movable_friendly_kings = self._nr_movable_friendly_kings(board)
        num_defended_pieces = self._num_defended_pieces(board)
        nr_lower_pieces = self._nr_lower_pieces(board)
        nr_middle_pawns = self._nr_middle_pawns(board)
        nr_middle_kings = self._nr_middle_kings(board)
        nr_higher_pawns = self._nr_higher_pawns(board)
        nr_diagonal_pawns = self. _nr_diagonal_pawns(board)
        nr_diagonal_kings = self. _nr_diagonal_kings(board)
        
        
        if self.phase == 0:
            weights = self.weights_phase0
        elif self.phase == 1:
            weights = self.weights_phase1
        elif self.phase == 2:
            weights = self.weights_phase2

        if nr_enemy_pawns + nr_enemy_kings == 0:
            return 999
        
        score += nr_enemy_pawns * weights['nr_enemy_pawns']

        score += nr_enemy_kings * weights['nr_enemy_kings']
        
        score += nr_safe_pawns * weights['nr_safe_pawns']
        
        score += nr_safe_kings * weights['nr_safe_kings']
        
        score += dis_friendly_promotion * weights['dis_friendly_promotion']

        score += nr_movable_friendly_pawns * weights['nr_movable_friendly_pawns']

        score += nr_movable_friendly_kings * weights['nr_movable_friendly_kings']

        score += num_defended_pieces * weights['num_defended_pieces']

        score += nr_lower_pieces * weights['nr_lower_pieces']

        score += nr_middle_pawns * weights['nr_middle_pawns']

        score += nr_middle_kings * weights['nr_middle_kings']

        score += nr_higher_pawns * weights['nr_higher_pawns']

        score += nr_diagonal_pawns * weights['nr_diagonal_pawns']

        score += nr_diagonal_kings * weights['nr_diagonal_kings']

        return score
    
    def _detect_game_phase(self,board):
        nr_friendly_pieces = 0
        nr_friendly_pawns = 0
        nr_friendly_kings = 0

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == False):
                        nr_friendly_pawns += 1
                        nr_friendly_pieces += 1
                    else:
                        nr_friendly_kings += 1
                        nr_friendly_pieces += 1
        
        if nr_friendly_pawns > 3 and nr_friendly_kings == 0:
            self.phase = 0
            return
        if nr_friendly_pawns > 3 and nr_friendly_kings > 0:
            self.phase = 1
            return
        if nr_friendly_pawns < 3:
            self.phase = 2 if self.phases == 3 else 1
            return
        
        else:
            return "ERROR: unkown phase"
     
    def _nr_enemy_pawns(self, board):
        nr_enemy_pawns = 0
        max_nr_enemy_pawns = 12

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == False):
                        nr_enemy_pawns += 1

        return nr_enemy_pawns/max_nr_enemy_pawns

    def _nr_enemy_kings(self, board):
        nr_enemy_kings = 0
        max_nr_enemy_kings = 12

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == True):
                        nr_enemy_kings += 1

        return nr_enemy_kings/max_nr_enemy_kings

    def _dis_friendly_promotion(self, board):
        distance = 0
        max_distance = 42

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color == self.eval_color):
                        if(self.eval_color == BLUE):
                            distance += j
                        else:
                            distance += (8 - (j+1))

        return distance/max_distance if distance > 0 else 1/max_distance

    def _num_defended_pieces(self, board):
        count = 0
        max_count = 6

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color == self.eval_color and i != 0 and i != 7):
                        if(occupant.color == BLUE and j != 7):
                            #Blue
                            try:
                                if (board.location(i - 1, j + 1).occupant is not None and board.location(i + 1, j + 1).occupant is not None and board.location(i - 1, j + 1).occupant.color == self.eval_color and board.location(i + 1, j + 1).occupant.color == self.eval_color):
                                    count += 1
                            except:
                                pass

                        elif(occupant.color == RED and j != 0):
                            #Red
                            try:
                                if(board.location(i - 1, j - 1).occupant is not None and board.location(i + 1, j - 1).occupant is not None and board.location(i - 1, j - 1).occupant.color == self.eval_color and board.location(i + 1, j - 1).occupant.color == self.eval_color):
                                    count += 1
                            except:
                                pass

        return count/max_count if count > 0 else 1/max_count

    
    def _nr_safe_pawns(self, board):
        nr_safe_pawns = 0
        max_nr_safe_pawns = 8

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == False and (i == 0 or i == 7)):
                        nr_safe_pawns += 1
        return nr_safe_pawns / max_nr_safe_pawns
    
    def _nr_safe_kings(self, board):
        nr_safe_kings = 0
        max_nr_safe_kings = 8

        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == True and (i == 0 or i == 7)):
                        nr_safe_kings += 1
        return nr_safe_kings/max_nr_safe_kings

    def _nr_movable_friendly_pawns(self,board):
        _nr_movable_friendly_pieces = 0
        _max_nr_movable_friendly_pieces = 12
        
        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == False):
                        if(len(board.legal_moves(i,j)) != 0):
                            _nr_movable_friendly_pieces += 1
        
        return _nr_movable_friendly_pieces/_max_nr_movable_friendly_pieces

    def _nr_movable_friendly_kings(self,board):
        _nr_movable_friendly_pieces = 0
        max_nr_movable_friendly_pieces = 12
        
        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color != self.eval_color and occupant.king == True):
                        if(len(board.legal_moves(i,j)) != 0):
                            _nr_movable_friendly_pieces += 1
        
        return _nr_movable_friendly_pieces/max_nr_movable_friendly_pieces

    def _nr_lower_pieces(self,board):
        nr_lower_pieces= 0
        max_nr_lower_pieces = 8

        if(self.eval_color == BLUE):
            for i in range(8):
                for j in range(6,8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color):
                            nr_lower_pieces+=1
        else:
            for i in range(8):
                for j in range(2):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color):
                            nr_lower_pieces+=1

        return nr_lower_pieces/max_nr_lower_pieces
    
    def _nr_middle_pawns(self,board):
        nr_middle_pawns= 0
        max_nr_middle_pawns = 8

        if(self.eval_color == BLUE):
            for i in range(2,6):
                for j in range(3,5):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color and occupant.king == False):
                            nr_middle_pawns+=1
        else:
            for i in range(2,6):
                for j in range(3,5):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color and occupant.king == False):
                            nr_middle_pawns+=1

        return nr_middle_pawns / max_nr_middle_pawns

    def _nr_middle_kings(self,board):
        nr_middle_kings= 0
        max_nr_middle_kings = 8

        if(self.eval_color == BLUE):
            for i in range(2,6):
                for j in range(3,5):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color and occupant.king == True):
                            nr_middle_kings+=1
        else:
            for i in range(2,6):
                for j in range(3,5):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color and occupant.king == True):
                            nr_middle_kings+=1

        return nr_middle_kings/max_nr_middle_kings
    
    def _nr_higher_pawns(self,board):
        nr_higher_pawns= 0
        max_nr_higher_pawns = 12

        if(self.eval_color == BLUE):
            for i in range(8):
                for j in range(3):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color and occupant.king == False):
                            nr_higher_pawns+=1
        else:
            for i in range(8):
                for j in range(5,8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if(occupant.color == self.eval_color and occupant.king == False):
                            nr_higher_pawns+=1

        return nr_higher_pawns / max_nr_higher_pawns

    def _nr_diagonal_pawns(self,board):
        nr_diagonal_pawns= 0
        max_nr_diagonal_pawns = 8

        if(self.eval_color == BLUE):
            for i in range(8):
                for j in range(8):
                    if i == j:
                        occupant = board.location(i, j).occupant
                        if(occupant is not None):
                            if(occupant.color == self.eval_color and occupant.king == False):
                                nr_diagonal_pawns+=1
        else:
            for i in range(8):
                for j in range(8):
                    if i == j:
                        occupant = board.location(i, j).occupant
                        if(occupant is not None):
                            if(occupant.color == self.eval_color and occupant.king == False):
                                nr_diagonal_pawns+=1
                                
        return nr_diagonal_pawns / max_nr_diagonal_pawns

    def _nr_diagonal_kings(self,board):
        nr_diagonal_kings= 0
        max_nr_diagonal_kings = 8

        if(self.eval_color == BLUE):
            for i in range(8):
                for j in range(8):
                    if i == j:
                        occupant = board.location(i, j).occupant
                        if(occupant is not None):
                            if(occupant.color == self.eval_color and occupant.king == True):
                                nr_diagonal_kings+=1
        else:
            for i in range(8):
                for j in range(8):
                    if i == j:
                        occupant = board.location(i, j).occupant
                        if(occupant is not None):
                            if(occupant.color == self.eval_color and occupant.king == True):
                                nr_diagonal_kings+=1

        return nr_diagonal_kings / max_nr_diagonal_kings
    
    def _piece2val(self, board):
        score = 0
        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color == self.eval_color):
                        score += occupant.value
                    else:
                        score -= occupant.value
        return score

    def _piece_and_row2val(self, board):
        score = 0
        if(self.eval_color == RED):
            for i in range(8):
                for j in range(8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if occupant.color == self.eval_color:
                            score += 5 + j + 2 * (occupant.king)
                        else:
                            score -= 5 + (8 - j) + 2 * (occupant.king)
        else:
            for i in range(8):
                for j in range(8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if occupant.color == self.eval_color:
                            score += 5 + (8 - j) + 2 * (occupant.king)
                        else:
                            score -= 5 + j + 2 * (occupant.king)
        return score

    def _piece_and_board2val(self, board):
        score = 0
        if(self.eval_color == RED):
            for i in range(8):
                for j in range(8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if occupant.color == self.eval_color and occupant.king:
                            score += 10
                        elif occupant.color != self.eval_color and occupant.king:
                            score -= 10
                        elif occupant.color == self.eval_color and j < 4:
                            score += 5
                        elif occupant.color != self.eval_color and j < 4:
                            score -= 7
                        elif occupant.color == self.eval_color and j >= 4:
                            score += 7
                        elif occupant.color != self.eval_color and j >= 4:
                            score -= 5
        else:
            for i in range(8):
                for j in range(8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        if occupant.color == self.eval_color and occupant.king:
                            score += 10
                        elif occupant.color != self.eval_color and occupant.king:
                            score -= 10
                        elif occupant.color == self.eval_color and j < 4:
                            score += 7
                        elif occupant.color != self.eval_color and j < 4:
                            score -= 5
                        elif occupant.color == self.eval_color and j >= 4:
                            score += 7
                        elif occupant.color != self.eval_color and j >= 4:
                            score -= 5
        return score

    def _piece_and_board_pov2val(self, board):
        score = 0
        num_pieces = 0
        if(self.eval_color == RED):
            for i in range(8):
                for j in range(8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        num_pieces += 1
                        if occupant.color == self.eval_color and occupant.king:
                            score += 10
                        elif occupant.color != self.eval_color and occupant.king:
                            score -= 10
                        elif occupant.color == self.eval_color and j < 4:
                            score += 5
                        elif occupant.color != self.eval_color and j < 4:
                            score -= 7
                        elif occupant.color == self.eval_color and j >= 4:
                            score += 7
                        elif occupant.color != self.eval_color and j >= 4:
                            score -= 5
        else:
            for i in range(8):
                for j in range(8):
                    occupant = board.location(i, j).occupant
                    if(occupant is not None):
                        num_pieces += 1
                        if occupant.color == self.eval_color and occupant.king:
                            score += 10
                        elif occupant.color != self.eval_color and occupant.king:
                            score -= 10
                        elif occupant.color == self.eval_color and j < 4:
                            score += 7
                        elif occupant.color != self.eval_color and j < 4:
                            score -= 5
                        elif occupant.color == self.eval_color and j >= 4:
                            score += 7
                        elif occupant.color != self.eval_color and j >= 4:
                            score -= 5
        return score / num_pieces

    def _all_kings(self, board):
        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None and occupant.king == False):
                    return False
        return True

    def _dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _pieces_loc(self, board):
        player_pieces = []
        adversary_pieces = []
        for i in range(8):
            for j in range(8):
                occupant = board.location(i, j).occupant
                if(occupant is not None):
                    if(occupant.color == self.eval_color):
                        player_pieces.append((i, j))
                    else:
                        adversary_pieces.append((i, j))
        return player_pieces, adversary_pieces

    def _sum_of_dist(self, board):
        player_pieces, adversary_pieces = self._pieces_loc(board)
        sum_of_dist = 0
        for pos in player_pieces:
            for adv in adversary_pieces:
                sum_of_dist += self._dist(pos[0], pos[1], adv[0], adv[1])
        if(len(player_pieces) >= len(adversary_pieces)):
            sum_of_dist *= -1
        return sum_of_dist

    def _farthest_piece(self, board):
        player_pieces, adversary_pieces = self._pieces_loc(board)
        farthest_dist = 0
        for pos in player_pieces:
            for adv in adversary_pieces:
                farthest_dist += max(farthest_dist, self._dist(pos[0], pos[1], adv[0], adv[1]))
        if(len(player_pieces) >= len(adversary_pieces)):
            farthest_dist *= -1
        return farthest_dist

    def _check_for_endgame(self, board):
        for x in range(8):
            for y in range(8):
                if board.location(x, y).color == BLACK and board.location(x, y).occupant != None and board.location(x, y).occupant.color == self.game.turn:
                    if board.legal_moves(x, y) != []:
                        return False
        return True
