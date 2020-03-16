"""
Ordering of child geographies (block polygons) within a parent geography.
"""

from .evolution import *
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['BlockOrder']


class BlockOrder:
    """Calculate an optimal order for the child geographies within a parent geography.

    This is implemented as a Genetic Algorithm to find a best effort approximation to solve the Travelling Salesman
    Problem. The block list is initially loaded as a random tour and a distance calculated. The tour is then evolved,
    swapping the block order on each evolution to try and find the shortest possible route through all the blocks.

    Inspired by https://gist.github.com/turbofart/3428880
    """

    def __init__(self, block_data, cgeo_attr, max_evolution_count=150):
        """Initialize the object

        Parameters
        ----------
        block_data : pandas.DataFrame
            DataFrame of block information. Must include the block UID, and x/y coordinates or a representative point.

        cgeo_attr : String
            The name of the field in block_data containing the UID value

        max_evolution_count : Integer, 150
            The maximum number of evolutions to try before returning a result.
        """

        logger.debug("BlockSequence class initialization started")

        self.block_data = block_data
        self.cgeo_attr = cgeo_attr
        self.max_evolution_count = max_evolution_count

        self.bo_name = 'block_order'

        # Create a new tour manager to hold all the possible tours for this set of blocks
        self.tourmanager = TourManager()

        # Initialize a tour with the data provided
        self._initialize_tour()

        # Create the initial population from the tourmanager
        self.block_count = len(self.block_data)
        self.block_population = Population(self.tourmanager, self.block_count, True)

    def _initialize_tour(self):
        """Initialize the tour manager with the block data."""

        logger.debug("Adding each block to tour manager")
        for index, point in self.block_data.iterrows():
            block_rep = Block(point[self.cgeo_attr], point['rep_point_x'], point['rep_point_y'])
            # logger.debug("Adding block %s to tour manager", block_rep)
            self.tourmanager.add_block(block_rep)

    def get_optimal_order(self):
        """Determine the optimal block order.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame containing the block UID and an associated block order value for every block in the input.
        """

        logger.debug("get_optimal_order start")

        # Tours with only one or two blocks have no value in being evolved, so the random order
        # calculated earlier is used to save processing time.
        if self.block_count > 2:
            # For areas with very sall numbers of blocks, it doesn't make sense to use the maximum possible number of
            # evolutions to find the optimal order. The number of possible combinations is a factorial of the initial
            # block count, so that is used to determine the maximum number of evolutions up to generation_count.
            evolution_count = self.max_evolution_count
            block_count_factorial = np.math.factorial(self.block_count)
            if block_count_factorial < evolution_count:
                evolution_count = block_count_factorial

            logger.debug("Evolving the block population %s times to find an optimal solution", evolution_count)
            # Set up the genetic algorithm on the set of tours
            block_ga = GenetricAlgorithm(self.tourmanager)

            # Perform an initial evolution on the population to set things up
            self.block_population = block_ga.evolve_population(self.block_population)

            # Perform the maximum number of evolutions allowed to try and find the shortest path between blocks
            for i in range(evolution_count):
                self.block_population = block_ga.evolve_population(self.block_population)

        # Return the best path that was found
        logger.debug("Final distance: %s", self.block_population.get_fittest().get_distance())
        chosen_block_order = self.block_population.get_fittest().get_block_order()

        # Create a DataFrame to put the order on the block ID
        df = pd.DataFrame(list(zip(chosen_block_order, range(1, len(chosen_block_order) + 1))),
                          columns=[self.cgeo_attr, self.bo_name])
        return df
