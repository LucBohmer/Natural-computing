import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import checkers
import gamebot
import evobotPhases
import random
import json
from time import sleep
from threading import Thread
import multiprocessing as mp
from itertools import combinations
import numpy as np

BLUE = (0,   0, 255)
RED = (255,   0,   0)

class Tournament:

	def __init__(self, standard_size=24, population_size=32, start_from_file=False, phases=3):
		self.standard_size = standard_size
		self.population_size = population_size
		self.randoms_size = 4
		self.random_change_probability = 0.5
		self.n_pools = 4
		self.pool_winners = []
		self.pool_second_place = []
		self.generation = []
		self.new_generation = []
		self.mating_pool = []
		self.max_moves = 100
		self.mutation_probability = 0.1
		self.start_from_file = start_from_file
		self.games_vs_ai = 5
		self.phases = phases

	def run(self):
		if self.start_from_file:
			self.read_weights_from_file()
		else:
			self.create_bots()
		print("Games per player:", self.games_vs_ai)
		for i in range(0, 200):
			print("Generation:", i)
			if i %20 == 0 and i != 0:
				if self.games_vs_ai < 35:
					self.games_vs_ai +=5
				print("Games per player:", self.games_vs_ai)
				if self.random_change_probability > 0.2:
					self.random_change_probability -= 0.1
			#print("Tournament")
			self.tournament_round()
			#print("Mating")
			self.mate()
			self.add_randoms()

			with open ('weights.json', 'a') as fp:
				json.dump(self.new_generation, fp)
				fp.write('\n')
			self.generation = self.new_generation.copy()
			self.new_generation = []
			self.mating_pool = []
			
	def add_randoms(self):
		for i in range(self.randoms_size):
			self.new_generation.append(self.generate_weights(self.new_generation[i], self.random_change_probability))
			
	def mate(self):
		# Each pair of parents get two children
		for winner in self.pool_winners:
			self.new_generation.append(winner)

		random.shuffle(self.pool_winners)
		for i in range(self.n_pools):
			#print(self.pool_winners[i])
			self.mating_pool.append(self.pool_winners[i])
			#add parents in different order (manually for now)
		self.mating_pool.append(self.pool_winners[0])
		self.mating_pool.append(self.pool_winners[2])
		self.mating_pool.append(self.pool_winners[1])
		self.mating_pool.append(self.pool_winners[3])
		self.mating_pool.append(self.pool_winners[1])
		self.mating_pool.append(self.pool_winners[2])
		self.mating_pool.append(self.pool_winners[0])
		self.mating_pool.append(self.pool_winners[3])
		# With second place from other pools
		self.mating_pool.append(self.pool_winners[0])
		self.mating_pool.append(self.pool_second_place[1])
		self.mating_pool.append(self.pool_winners[1])
		self.mating_pool.append(self.pool_second_place[0])
		self.mating_pool.append(self.pool_winners[2])
		self.mating_pool.append(self.pool_second_place[3])
		self.mating_pool.append(self.pool_winners[3])
		self.mating_pool.append(self.pool_second_place[2])
		self.mating_pool.append(self.pool_winners[0])
		self.mating_pool.append(self.pool_second_place[2])
		self.mating_pool.append(self.pool_winners[1])
		self.mating_pool.append(self.pool_second_place[3])



		for i in range(0, self.standard_size, 2):		
			#child1
			child1 = self.mating_pool[i].copy()
			#child2
			child2 = self.mating_pool[i].copy()
			for phase in range(len(child1)):
				for feature in child1[phase]:
					if random.choice([0, 1]) == 1:
						child1[phase][feature] = round(0.9*self.mating_pool[i+1][phase][feature] + 0.1*self.mating_pool[i][phase][feature],3)
						child2[phase][feature] = round(0.1*self.mating_pool[i+1][phase][feature] + 0.9*self.mating_pool[i][phase][feature],3)
					else:
						child1[phase][feature] = round(0.1*self.mating_pool[i+1][phase][feature] + 0.9*self.mating_pool[i][phase][feature],3)
						child2[phase][feature] = round(0.9*self.mating_pool[i+1][phase][feature] + 0.1*self.mating_pool[i][phase][feature],3)
					if random.random() < self.mutation_probability:
						child1[phase][feature] + round(random.uniform(-1, 1), 2)
					if random.random() < self.mutation_probability:
						child2[phase][feature] + round(random.uniform(-1, 1), 2)

			self.new_generation.append(child1)
			self.new_generation.append(child2)


	def tournament_round(self):
		pools = []
		self.pool_winners = []
		generation_indexes = list(range(self.population_size))
		random.shuffle(generation_indexes)
		for i in range(0, self.population_size, int(self.population_size/self.n_pools)):
			pools.append(generation_indexes[i:i+int(self.population_size/self.n_pools)])
		#all_combinations = []
		# for pool in pools:
		# 	all_combinations.append(list(combinations(pool, 2)))
		processes = []
		manager = mp.Manager()
		pool_winners_shared = manager.list()
		pool_second_place_shared = manager.list()
		average = manager.Value('d', 0)
		best = manager.Value('d', 0)
		for pool in pools:
			try:
				t = mp.Process(target=self.play_pool, args=(pool,pool_winners_shared,pool_second_place_shared,average,best))
				processes.append(t)
			except Exception as e:
				print(e)
				exit()
		[t.start() for t in processes]
		for t in processes:
			t.join()
		self.pool_winners = pool_winners_shared
		self.pool_second_place = pool_second_place_shared
		average.value = (average.value/32)/(self.games_vs_ai*3)
		print("Generation average score:", round(average.value,3))
		print("Generation best score:", round(best.value,3))

	def play_pool(self, pool,pool_winners_shared,pool_second_place_shared,average, best):
		pool_score = {x : 0 for x in np.unique([item for item in pool])}
		for player in pool:
			player1 = self.generation[player]
			for i in range(self.games_vs_ai):
				winner = None
				tries = 0
				while tries < 3 and winner == None:
					tries += 1
					winner = self.play(player1)
				if winner == "Blue":
					#print("AI wins")
					pool_score[player] += 3
				elif winner == "Red":
					#print("MinMax wins")
					pass
				if winner is None:
					#print("Draw")
					pool_score[player] += 1

		pool_sorted = sorted(pool_score, key=pool_score.get)
		pool_winner = pool_sorted[-1]
		pool_second_place = pool_sorted[-2]
		# self.pool_winners.append(self.generation[pool_winner])
		# self.pool_second_place.append(self.generation[pool_second_place])
		pool_winners_shared.append(self.generation[pool_winner])
		pool_second_place_shared.append(self.generation[pool_second_place])
		#print(pool_score)
		#print("Best score:",)
		#print("Average:", sum(list(pool_score.values()))/len(pool_score))
		average.value += sum(list(pool_score.values()))
		if pool_score[pool_winner]/(self.games_vs_ai*3) > best.value:
			best.value = pool_score[pool_winner]/(self.games_vs_ai*3)
		

	def play(self, weights1):
		max_tries = 5
		i = 0
		while i <= max_tries:
			game = checkers.Game(loop_mode=True)
			#game.setup()
			if self.phases == 2:
				bot1 = evobotPhases.Bot(game, BLUE, mid_eval='piece_and_board',
							end_eval='sum_of_dist', weights0=weights1[0], weights1=weights1[1], phases=2)
			elif self.phases ==3:
				bot1 = evobotPhases.Bot(game, BLUE, mid_eval='piece_and_board',
							end_eval='sum_of_dist', weights0=weights1[0], weights1=weights1[1], weights2=weights1[2], phases=3)

			bot2 = gamebot.Bot(game, RED, mid_eval='piece_and_board_pov', method='minmax', depth=1, end_eval='sum_of_dist')
			
			while True:  # main game loop
				if game.turn == BLUE:
					bot1.step(game.board)
					#game.update()	
				else:
					bot2.step(game.board)
					#game.update()
				if game.endit or bot1._end_eval_time or bot2._end_eval_time:
					break
				# This is to prevent infinite loops, need to evaluate the actual board states for result
				if game.board.moves > self.max_moves:

					return None
			if game.winner == "Blue":
				return "Blue"
			else:
				return "Red"

		print("Exceeded max error rate")

	def create_bots(self):
		self.generation = []
		for i in range(self.population_size):
			self.generation.append(self.generate_weights())

	def generate_weights(self, weights=None, probability=1):
		if weights is None:
			weights = []
			for phase in range(self.phases):
				weights_phase = {'nr_enemy_pawns' : -1,'nr_enemy_kings' : -1,'nr_safe_pawns' : 1,'nr_safe_kings' : 1, 'dis_friendly_promotion' : -1,"nr_movable_friendly_pawns" : 1,"nr_movable_friendly_kings" : 1, 'num_defended_pieces' : 1,"nr_lower_pieces":1,"nr_middle_pawns":1,"nr_middle_kings":1,"nr_higher_pawns":1,"nr_diagonal_pawns":1,"nr_diagonal_kings":1}
				weights.append(weights_phase)
		for weights_phase in range(self.phases):
			for feature in weights[weights_phase]:
				if random.random() < probability:
					weights[weights_phase][feature] = round(random.uniform(-1, 1), 3)
		return weights

	def read_weights_from_file(self):
		with open('weights.json', 'r') as f:
			last_line = f.readlines()[-1]
			self.generation = json.loads(last_line)

	def read_weights_from_file_generation(self, generation_i):
		with open('weights.json', 'r') as f:
			line = f.readlines()[generation_i]
			self.generation = json.loads(line)

if __name__ == '__main__':
	tournament = Tournament(start_from_file=False, phases=3)
	tournament.run()