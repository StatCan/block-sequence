"""
Classes used to implement Genetic Algorithm that solves the travelling salesman problem.
"""

import logging
import numpy as np
import random

logger = logging.getLogger(__name__)

__all__ = ['Block', 'TourManager', 'Tour', 'Population', 'GenetricAlgorithm']

class Block:
    """Used to represent an individual block in the collection.

    This is the equivalent of a gene in the standard Genetic Algorithm parlance.
    """

    def __init__(self, label=None, x=None, y=None):
        self.x = x
        self.y = y
        self.label = label

        # logger.debug("Block %s initialized at (%s, %s)", self.label, self.x, self.y)

    def distance_to(self, block):
        """Calculate the euclidean distance between two blocks."""

        x_dist = abs(self.x - block.x)
        y_dist = abs(self.y - block.y)

        dist = np.hypot(x_dist, y_dist)
        # logger.debug("Distance between %s and %s: %s", self, block, dist)
        return dist

    def __repr__(self):
        return f"Block {self.label}"


class TourManager:
    """A collection of blocks to be put into a tour."""

    def __init__(self):
        self.destination_blocks = []

    def add_block(self, block):
        """Add a new block to a set of destinations."""
        self.destination_blocks.append(block)

    def get_block(self, index):
        """Retrieve a block from the list of destinations."""
        return self.destination_blocks[index]

    def number_of_blocks(self):
        """Same as len(tourmanager)."""
        return len(self)

    def __len__(self):
        return len(self.destination_blocks)

    def __repr__(self):
        return f"TourManager containing {len(self)} blocks"


class Tour:
    """A path between all the destination blocks.

    The tour behaves like a list, allowing blocks to be set and retrieved by position in the tour.
    This is the equivalent to the chromosome in Genetic Algorithm parlance.
    """

    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.tour = []
        self.fitness = 0.0
        self.distance = 0

        # initialize the tour will null blocks
        if tour is not None:
            self.tour = tour
        else:
            # logger.debug("Initializing tour with %s empty blocks", len(self.tourmanager))
            for i in range(self.tourmanager.number_of_blocks()):
                self.tour.append(None)
            # logger.debug("Initialized tour size: %s", len(self.tour))

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, item):
        return self.tour[item]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        block_string = "|"
        for i in range(self.tour_size()):
            block_string += str(self.get_block(i)) + "|"
        return block_string

    def get_block_order(self):
        block_list = []
        for i in range(self.tour_size()):
            block_list.append(self.get_block(i).label)
        return block_list

    def generate_individual(self):
        tm_size = self.tourmanager.number_of_blocks()
        for block_index in range(tm_size):
            block = self.tourmanager.get_block(block_index)
            self.set_block(block_index, block)

        # shuffle the tour in a random order as a starting order
        random.shuffle(self.tour)

    def get_block(self, tour_position="last"):
        if tour_position != "last":
            return self.tour[tour_position]
        else:
            return self.tour[-1]

    def set_block(self, tour_position, block):
        self.tour[tour_position] = block
        self.fitness = 0.0
        self.distance = 0

    def get_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.get_distance())
        return self.fitness

    def get_distance(self):
        """Calculate the distance between all the nodes in the tour."""

        # It is possible for a parent geo to only have one child, in which case the dsitance would be zero.
        # To defend against this, return 1.
        if self.tour_size() == 1:
            logger.debug("Tour contains only 1 block, so size is 1")
            return 1

        # calculate the distance between all the blocks in the tour
        if self.distance == 0:
            # logger.debug("Distance hasn't been calculated before. Iterating all blocks in the tour.")
            tour_distance = 0
            for block_index in range(self.tour_size()):
                from_block = self.get_block(block_index)
                destination_block = None

                if block_index + 1 < self.tour_size():
                    destination_block = self.get_block(block_index + 1)
                else:
                    destination_block = self.get_block("last")
                tour_distance += from_block.distance_to(destination_block)
            self.distance = tour_distance
        return self.distance

    def tour_size(self):
        return len(self.tour)

    def contains_block(self, block):
        return block in self.tour


class Population:
    """A collection of possible tours.

    Each tour is one way to navigate through the set of blocks.
    """

    def __init__(self, tourmanager, population_size, initialize):
        self.tours = []
        # initialize null tours to match the desired population size
        for i in range(population_size):
            self.tours.append(None)

        # logger.debug("Should initialize population: %s", initialize)
        if initialize:
            # logger.debug("Creating new tour of size %s", population_size)
            for i in range(population_size):
                new_tour = Tour(tourmanager)
                new_tour.generate_individual()
                # logger.debug("Saving new tour at index %s", i)
                self.save_tour(i, new_tour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, item):
        return self.tours[item]

    def __len__(self):
        return len(self.tours)

    def __repr__(self):
        return f"Population with {len(self)} tours"

    def save_tour(self, index, tour):
        self.tours[index] = tour

    def get_tour(self, index):
        return self.tours[index]

    def get_fittest(self):
        fittest = self.tours[0]
        for i in range(self.population_size()):
            if fittest.get_fitness() <= self.get_tour(i).get_fitness():
                fittest = self.get_tour(i)
        return fittest

    def population_size(self):
        return len(self.tours)


class GenetricAlgorithm:
    """Implementation of the Genetic Algorithm.

    This controls the way the population will evolve, mutating randomly by swapping two blocks in a tour to introduce
    variation among the populations.
    """

    def __init__(self, tourmanager):
        self.tourmanager = tourmanager
        self.mutation_rate = 0.015
        self.tournament_size = 5
        self.elitism = True
        logger.debug("Initilized GA with tour manager: %s", tourmanager)

    def evolve_population(self, pop):
        # logger.debug("Evolving the population of %s", pop)
        new_population = Population(self.tourmanager, pop.population_size(), False)
        elitism_offset = 0
        if self.elitism:
            new_population.save_tour(0, pop.get_fittest())
            elitism_offset = 1

        for i in range(elitism_offset, new_population.population_size()):
            parent1 = self.tournament_selection(pop)
            parent2 = self.tournament_selection(pop)
            child = self.crossover(parent1, parent2)
            new_population.save_tour(i, child)

        for i in range(elitism_offset, new_population.population_size()):
            self.mutate(new_population.get_tour(i))

        return new_population

    def crossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        start_pos = int(random.random() * parent1.tour_size())
        end_pos = int(random.random() * parent1.tour_size())

        for i in range(child.tour_size()):
            if start_pos < end_pos and start_pos < i < end_pos:
                child.set_block(i, parent1.get_block(i))
            elif start_pos > end_pos:
                if not (start_pos > i > end_pos):
                    child.set_block(i, parent1.get_block(i))

        for i in range(parent2.tour_size()):
            if not child.contains_block(parent2.get_block(i)):
                for ii in range(child.tour_size()):
                    if child.get_block(ii) is None:
                        child.set_block(ii, parent2.get_block(i))
                        break

        return child

    def mutate(self, tour):
        for tour_pos1 in range(tour.tour_size()):
            if random.random() < self.mutation_rate:
                tour_pos2 = int(tour.tour_size() * random.random())

                block1 = tour.get_block(tour_pos1)
                block2 = tour.get_block(tour_pos2)

                tour.set_block(tour_pos2, block1)
                tour.set_block(tour_pos1, block2)

    def tournament_selection(self, pop):
        tournament = Population(self.tourmanager, self.tournament_size, False)

        for i in range(self.tournament_size):
            random_id = int(random.random() * pop.population_size())
            tournament.save_tour(i, pop.get_tour(random_id))
        fittest = tournament.get_fittest()
        return fittest
