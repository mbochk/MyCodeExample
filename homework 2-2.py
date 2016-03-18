# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:12:08 2016

@author: mbochk

See pdf report "Homework 2, Bochkarev Mikhail, problem 2.pdf" for info about the task and the programm.
"""
import argparse
import copy
import itertools
import math
import random

from collections import defaultdict

"""Global constant that defines dimensionality of ocean

Note, that ascii output works only with DIMENSIONS == 2.
"""
DIMENSIONS = 2


# ====== Direction ======


class Direction(list):
    """Vector-like class, based on list.

    Intendent to represent possible direction for move/eat/other action.
    """
    def __init__(self, array):
        if (len(array) != DIMENSIONS):
            raise ValueError("Direction is created from {}-element array only"
                             .format(DIMENSIONS))
        super(Direction, self).__init__(array)


class DirectionsFactory(object):
    """Factory for standart lists od Directions.
    """

    @staticmethod
    # i guess using wrap from functools is more correct
    def directionize(directions_list):
        return [Direction(direction) for direction in directions_list]

    @classmethod
    def directions_axiswise(cls):
        return cls.directions_square_sum(square_sum=1)

    @classmethod
    def directions_8(cls):
        return cls.directions_square_sum(square_sum=1) + \
            cls.directions_square_sum(square_sum=2)

    @classmethod
    def directions_square_sum(cls, square_sum=5):
        max_dev = int(math.ceil(float(square_sum) ** (1. / DIMENSIONS)))
        directions = [i for i in itertools.product(
                      range(-max_dev, max_dev + 1), repeat=DIMENSIONS)
                      if sum([j * j for j in i]) == square_sum]
        return cls.directionize(directions)


# ====== Position ======


class Position(list):
    """Basic list class serving as index in ocean space.
    """
    def __init__(self, array=[0, 0], *args, **kwargs):
        if (len(array) != DIMENSIONS):
            raise ValueError("Position is created from 2-element array only"
                             .format(DIMENSIONS))
        super(Position, self).__init__(array)

    def as_tuple(self):
        return tuple(self)

    def position_toward(self, direction):
        """Create another position, moved toward direction
        """
        another = copy.copy(self)
        another.move_toward(direction)
        return another

    def __add__(self, direction):
        return self.position_toward(direction)

    def move_toward(self, direction):
        assert(len(direction) == len(self))
        for i in xrange(len(self)):
            self[i] += direction[i]

    def swap_positions(self, another):
        assert(len(self) == len(another))
        for i in xrange(len(self)):
            self[i], another[i] = another[i], self[i]


class ModularPosition(Position):
    """Position with elements 'taken by modulo'
    """
    DEFAULT_MODULO = 7

    @staticmethod
    def _modulize(position):
        for i in xrange(len(position)):
            position[i] %= position._modulo

    def __init__(self, modulo=DEFAULT_MODULO, *args, **kwargs):
        super(ModularPosition, self).__init__(*args, **kwargs)
        if (not isinstance(modulo, int) or modulo <= 0):
            raise ValueError("PositionModular modulo is int >0")
        self._modulo = modulo
        self._modulize(self)

    def position_toward(self, direction):
        another = super(ModularPosition, self).position_toward(direction)
        self._modulize(another)
        return another


class OceanGridPosition(Position):
    """Position with bounded ocean grid.
    """
    def __init__(self, grid, *args, **kwargs):
        super(OceanGridPosition, self).__init__(*args, **kwargs)
        self._grid = grid

    def get(self):
        return self._grid(self)

    def is_legit(self):
        return self._grid.contains(self)

    def is_empty(self):
        return self._grid.empty_at(self)

    def swap_with_content(self, new_position):
        self._grid.swap_positions(self, new_position)
        self.swap_positions(new_position)

    def erase(self):
        self._grid.erase(self)

    def populate(self, kind):
        self._grid.populate_one(self, kind)

    def emplace(self, creature):
        self._grid.emplace(self, creature)

    def ocean(self):
        return self._grid


class ModularOceanGridPosition(ModularPosition, OceanGridPosition):
    """Position for 'toric' ocean.
    """
    def __init__(self, grid, *args, **kwargs):
        super(ModularOceanGridPosition, self).__init__(
            modulo=grid._size, grid=grid, *args, **kwargs)


# ====== Strategy ======


def check_legacy(position_predicate):
    """Extends predicate of two position to check correctness of the second.
    """
    def new_predicate(source_pos, next_pos):
        if next_pos is None or not next_pos.is_legit():
            return False
        return position_predicate(source_pos, next_pos)
    return staticmethod(new_predicate)


class DirectionChoosingStrategy(object):
    """Abstract class that makes decision to choose one of allowed directions.
    """
    def __init__(self, directions):
        self._allowed_directions = directions

    # should i provide predicates here for reusal, or in each concrete
    # strategy?

    @check_legacy
    def _possible_to_eat(source_pos, next_pos):
        return next_pos.get() is not None and \
            source_pos.get().can_eat(next_pos.get())

    @check_legacy
    def _possible_to_move(source_pos, next_pos):
        return next_pos.get() is None

    @check_legacy
    def _possible_to_move_or_eat(source_pos, next_pos):
        return next_pos.get() is None or \
            source_pos.get().can_eat(next_pos.get())

    def choose_target_position(self, source_pos):
        directions = self._filter_directions(source_pos)
        if len(directions) > 0:
            return random.choice(directions)

    def _filter_directions(self, source_pos):
        next_positions = [source_pos + direction for direction in
                          self._allowed_directions]
        return [position for position in next_positions if
                self._predicate(source_pos, position)]

    def __repr__(self):
        return "strategy {} with directions {}".format(
            self.__class__.__name__, self._allowed_directions)


class EatingStrategy(DirectionChoosingStrategy):
    def __init__(self, directions):
        super(EatingStrategy, self).__init__(directions)
        self._predicate = self._possible_to_eat


class MovingStrategy(DirectionChoosingStrategy):
    def __init__(self, directions):
        super(MovingStrategy, self).__init__(directions)
        # self._predicate = self._possible_to_move
        self._predicate = self._possible_to_move_or_eat


class ReproducingStrategy(DirectionChoosingStrategy):
    def __init__(self, directions):
        super(ReproducingStrategy, self).__init__(directions)
        self._predicate = self._possible_to_move


class CreatureStrategies(object):
    """Collective class for all strategies used in creatures.
    """
    @staticmethod
    def attribute_classes():
        return [EatingStrategy, MovingStrategy, ReproducingStrategy]

    # this looks bad and hard-written
    def __init__(self, eating_strategy, moving_strategy, reproducing_strategy):
        self.eating = eating_strategy
        self.moving = moving_strategy
        self.reproducing = reproducing_strategy

    @classmethod
    def compose(cls, *args, **kwargs):
        return cls(*map(lambda strat_cls, direction: strat_cls(direction),
                        cls.attribute_classes(), args))


# ====== Timers ======


class DiscreteTimer(object):
    """Auxulary class for event timing.
    """
    def __init__(self, time_limit):
        self._time_limit = time_limit
        self._time = 0
        self._disabled = False

    def __repr__(self):
        return self.__class__.__name__ + " {}/{}".format(
            self._time, self._time_limit)

    def refresh(self):
        self._time = 0

    def disable(self):
        self._disabled = True

    def activate(self):
        self._disables = False

    def add(self):
        if not self._disabled:
            self._time += 1

    def tic(self):
        if not self._disabled:
            self._time += 1
            if self._time >= self._time_limit:
                self.refresh()
                return True
        return False

    def randomized_reset(self):
        self._time = random.randint(0, self._time_limit / 3)


class CreatureTimers(object):
    """Collective class for timers needed in creatures
    """
    @staticmethod
    def attribute_classes():
        return [DiscreteTimer, DiscreteTimer]

    # this looks bad and hard-written
    def __init__(self, reproduction_timer, starvation_timer):
        self.reproduction = reproduction_timer
        self.starvation = starvation_timer

    @classmethod
    def compose(cls, *args, **kwargs):
        return cls(*map(lambda strat_cls, direction: strat_cls(direction),
                        cls.attribute_classes(), args))


# ====== Creature ======


class Creature(object):
    """Abstract creature class. Unifies both kind of behaviors.

    Attributes:
        _position (Position): position of creature in ocean
        _strats (CreatureStrategies): collection of DirectionChoosingStrategies
        _timers (CreatureTimers): collection of DiscreteTimers
        _prey_list (list): list of eatable class names
        is_alive (bool): flag of being not dead

        default_timelimits (list): 'initizlization list' for default
            CreatureTimers. Intended to be change while experimenting.

        default_directions (list): 'initizlization list' for default
            CreatureStrategies

        default_strats (CreatureStrategies): default strats collection
            (for strats we can have only one instance for all Creatures or
            individual classes)
    """
    default_timelimits = [20, 10]

    # CreatureStrategies = [.eating, .moving, .reproducing]

    default_directions = [DirectionsFactory.directions_axiswise(),
                          DirectionsFactory.directions_square_sum(),
                          DirectionsFactory.directions_axiswise()]

    default_strats = CreatureStrategies.compose(*default_directions)

    @classmethod
    def _kind(cls):
        return cls.__name__

    def __init__(self, position, strategies=None,
                 timers=None):
        if strategies is None:
            strategies = self.default_strats
        if timers is None:
            timers = CreatureTimers.compose(*self.default_timelimits)
        self._position = position
        self._strats = strategies
        self._timers = timers
        # makes reproduction events happen smoother
        self._timers.reproduction.randomized_reset()

        self._prey_list = []
        self.is_alive = True
        self._position.emplace(self)

    def __repr__(self):
        return self._kind() + " at {}".format(self._position)

    def _try_action(self, strategy, action, target_position=None):
        """Abstract method for trying action according to strategy.

        Returns: target_position if action succeed, None otherwise
        """
        if target_position is None:
            target_position = strategy.choose_target_position(self._position)
        if self._is_correct_target(target_position, strategy):
            action(target_position)
            return target_position
        return None

    def _is_correct_target(self, target_position, strategy):
        """Rechecks if taget_position is acceptable.
        """
        return strategy._predicate(self._position, target_position)

    def _try_move_or_eat(self):
        """Try method for _move_or_eat
        """
        self._try_action(self._strats.moving, self._move_or_eat)

    def _try_move_or_eat_deprecated(self):
        """Deprecated try method where _eat is prioritized over _move.

        It was implemented first (because more logical!), but makes
        ecosystem more unstable, resulting in higher chances for exctinction.
        (needs more ocean space to be consistnent)
        """
        if self._try_action(self._strats.eating, self._eat) is None:
            self._try_action(self._strats.moving, self._move)

    def _try_reproduce(self):
        """Try method for _reproduce
        """
        self._try_action(self._strats.reproducing, self._reproduce)

    def _move_or_eat(self, target_position):
        """Moves or eats self, according to target_position's content
        """
        if target_position.get() is None:
            self._move(target_position)
        else:
            self._eat(target_position)

    def _move(self, target_position):
        """Deligates _position to move self in ocean
        """
        self._position.swap_with_content(target_position)

    def _eat(self, target_position):
        """Asks another creature to die and then moves self
        """
        target_position.get()._die()
        self._timers.starvation.refresh()
        self._move(target_position)
        # self._timers.reproduction.add()

    def _reproduce(self, target_position):
        """Deligates target_position to populate it
        """
        target_position.populate(self._kind())

    def make_turn(self):
        self._try_move_or_eat()
        if self._timers.reproduction.tic():
            self._try_reproduce()
        if self._timers.starvation.tic():
            self._die()

    def _die(self):
        """Deligates _position to remove self from ocean.

        is_alive is used to prevent activity from self if self was eaten,
        but not acted this round.
        """
        self._position.erase()
        self.is_alive = False

    def can_eat(self, another):
        return another._kind() in self._prey_list


class DefaultCreaturesFactory(object):
    """Factory for ocean population. Does almost nothing since Creature is
    highly unified and defaulted.

    (class) Attributes:
        default_factories (dict): gives factory by class name.

    Idk if that the best way to do factory, but it seems very good way, if
    all Creatures are already standartized in behavior and all the work is
    can be done by constructors.

    Any comments about if it should be done in other way???
    Was the inintial design with default value in here better?
    """
    default_factories = {}

    @classmethod
    def add_factory(cls, creature_cls):
        """Pseudo-decorator used to populate default_factories.
        """
        cls.default_factories[creature_cls._kind()] = \
            cls.make_factory(creature_cls)
        return creature_cls

    @classmethod
    def make_factory(cls, creature_cls):
        """Make factory that creates Creature given position only.
        """
        def factory(position):
            return creature_cls(position=position)
        return factory

    @classmethod
    def make_factory_old(cls, creature_cls):
        """Used to made factory that created Creature given position only.
        """
        def factory(position):
            return creature_cls(position, cls.default_strats(),
                                cls.default_timers(creature_cls))
        return factory

    @staticmethod
    def default_strats():
        """Deprecated (moved to Creature class members)
        """
        default_directions = [DirectionsFactory.directions_axiswise(),
                              DirectionsFactory.directions_square_sum(),
                              DirectionsFactory.directions_axiswise()]
        return CreatureStrategies.compose(*default_directions)

    @classmethod
    def default_timers(cls, creature_cls):
        """Deprecated (moved to Creature class default constructor)
        """
        return CreatureTimers.compose(*creature_cls.default_timelimits)

    @classmethod
    def create(cls, creature_cls_str, position):
        return cls.default_factories[creature_cls_str](position)


@DefaultCreaturesFactory.add_factory
class Fishy(Creature):
    def __init__(self, *args, **kwargs):
        super(Fishy, self).__init__(*args, **kwargs)
        # make sure fishy lives forever and cannot eat
        self._timers.starvation.disable()
        self._strats.eating._allowed_directions = []


@DefaultCreaturesFactory.add_factory
class Sharky(Creature):
    def __init__(self, *args, **kwargs):
        super(Sharky, self).__init__(*args, **kwargs)
        # bad way to populate _prey_list?
        self._prey_list.append(Fishy._kind())


@DefaultCreaturesFactory.add_factory
class Ktulhu(Creature):
    def __init__(self, *args, **kwargs):
        super(Ktulhu, self).__init__(*args, **kwargs)
        # make sure Ktulhu lives forever and dont replicate
        self._timers.starvation.disable()
        self._timers.reproduction.disable()
        # bad way to populate _prey_list?
        self._prey_list.append(Fishy._kind())
        self._prey_list.append(Sharky._kind())


# ====== Ocean ======


class OceanGrid(object):
    """Basis of ocean that interacts with Positions.

    Attribures:
        _size (int): Grid size
        _grid (dict): Container for information about each position
    """
    def __init__(self, size):
        self._size = size
        self._grid = {}
        for pos in itertools.product(range(size), repeat=DIMENSIONS):
            self._grid[tuple(pos)] = None

    def __call__(self, position):
        return self._grid[tuple(position)]

    def contains(self, position):
        for x in position:
            if not 0 <= x < self._size:
                return False
        return True

    def empty_at(self, position):
        return self(position) is None

    def swap_positions(self, pos1, pos2):
        pos1, pos2 = pos1.as_tuple(), pos2.as_tuple()
        self._grid[pos1], self._grid[pos2] = self._grid[pos2], self._grid[pos1]
        return pos1, pos2

    def emplace(self, position, creature):
        if (self(position) is not None):
            raise ValueError("position {} already populated".format(position))
        self._grid[position.as_tuple()] = creature

    def erase(self, position):
        creature = self(position)
        self._grid[position.as_tuple()] = None
        return creature

    def populate_one(self, position, kind):
        raise NotImplementedError("Use Ocean class for population")


class Ocean(OceanGrid):
    """Class for handling creatures in ocean and making experiments.

    Args:
        position_cls (OceanGridPosition): class of positions used for
            communicating with creatures.
            (Default=ModularOceanGridPosition), effectivly making toric space
            from ocean

        creature_factory: Factory class for making creatures
    """
    def __init__(self, position_cls=ModularOceanGridPosition,
                 creature_factory=DefaultCreaturesFactory, *args, **kwargs):
        super(Ocean, self).__init__(*args, **kwargs)
        self._position_cls = position_cls
        self._factory = creature_factory
        self._inhabitants = []
        self._free_positions = set(self._grid)
        self._creature_counter = defaultdict(int)

    def _get_random_free_position(self):
        if len(self._free_positions) > 0:
            return self._position_cls(grid=self, array=random.choice(
                tuple(self._free_positions)))
        return None

    def _get_random_fair_kind(self):
        kinds = self._factory.default_factories.keys()
        # Ktulhu is unfair since its immortal
        kinds.remove(Ktulhu._kind())
        return random.choice(kinds)

    def swap_positions(self, pos1, pos2):
        pos1, pos2 = super(Ocean, self).swap_positions(pos1, pos2)
        if (pos1 in self._free_positions) != (pos2 in self._free_positions):
            self._free_positions.symmetric_difference_update(set([pos1, pos2]))

    def emplace(self, position, creature):
        super(Ocean, self).emplace(position, creature)
        self._inhabitants.append(creature)
        self._free_positions.remove(position.as_tuple())
        self._creature_counter[creature._kind()] += 1

    def erase(self, position):
        creature = super(Ocean, self).erase(position)
        if creature is not None:
            self._inhabitants.remove(creature)
            self._free_positions.add(position.as_tuple())
            self._creature_counter[creature._kind()] -= 1

    def populate_one(self, position=None, kind=None):
        if kind is None:
            kind = self._get_random_fair_kind()
        if position is None:
            position = self._get_random_free_position()
        if kind is not None and position is not None:
            self._factory.create(kind, position)
            # position will be emplaced on creation

    def populate_many(self, position=None, kind=None, count=None):
        if count is None:
            count = self._size
        while count > 0:
            self.populate_one(position=position, kind=kind)
            count -= 1

    def __contains__(self, element):
        if (isinstance(element, Creature)):
            return element in self._inhabitants
        else:
            raise ValueError("""Ocean.__contains__ can take only Creature as
                             argument""")

    def make_round(self):
        turn_order = self._inhabitants[:]
        for creature in turn_order:
            if creature.is_alive:
                creature.make_turn()

    def in_ascii(self):
        if DIMENSIONS != 2:
            raise ValueError("Ascii output work only for DIMENSION == 2")
        ocean_string = "." * self._size
        ocean_string += ("\n" + ocean_string) * (self._size - 1)
        ocean_string = list(ocean_string)
        for creature in self._inhabitants:
            position = creature._position
            char_position = position[1] + (self._size + 1) * position[0]
            ocean_string[char_position] = creature._kind()[0]
        return ''.join(ocean_string)

    def count_creatures(self):
        return self._creature_counter

    def volume(self):
        return self._size ** DIMENSIONS

# ====== Plotting ======


def make_path(population_sizes):
    import matplotlib.path as mpath

    Path = mpath.Path
    length = population_sizes.shape[1]
    population_sizes
    verts, codes = [], []
    for i in xrange(length):
        verts.append(tuple(population_sizes[:, i]))
        if i % 4 != 0:
            codes.append(Path.CURVE4)
        else:
            codes.append(Path.LINETO)

    while (len(codes) % 4 != 1):
        verts.append(tuple(population_sizes[:, -1]))
        codes.append(Path.CURVE4)
    codes[0] = Path.MOVETO
    codes[-1] = Path.LINETO
    verts.append(verts[-1])
    codes.append(codes[-1])
    return mpath.Path(verts, codes)


def plot_path(array, size, results):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    length = array.shape[1]

    if length < 4 * 4:
        return

    fig, ax = plt.subplots()
    plt.xlim([0, np.max(array[0, :])])
    plt.ylim([0, size ** DIMENSIONS])

    # plot path
    path = make_path(array)
    patch = mpatches.PathPatch(
        path, facecolor='w', lw=2, edgecolor='m', label="Fishy/Sharky")
    ax.add_patch(patch)

    # plot start
    plt.plot([array[0, 0]], [array[1, 0]], 'rx', ms=10, mew=4,
             label="Starting Point")

    # plot end
    plt.plot([array[0, -1]], [array[1, -1]], 'bx', ms=10, mew=4,
             label="Ending Point")

    # plot border
    max_value = size ** DIMENSIONS
    border = [[0, max_value], [max_value, 0]]
    plt.plot(border[0], border[1], 'k--', lw=10, label="Ocean Volume Limit")

    plt.title(results)
    plt.ylabel("Fishy")
    plt.xlabel("Sharky")
    plt.legend()

    ax.grid()
    plt.show()


def plot_normal(array, size, results):
    import numpy as np
    import matplotlib.pyplot as plt

    length = array.shape[1]
    x_values = range(length)
    max_value = size ** DIMENSIONS

    plt.figure()
    plt.ylim([0, np.max(array[1, :]) + np.max(array[0, :])])

    # plot border
    border = [[0, x_values[-1]], [max_value, max_value]]
    plt.plot(border[0], border[1], 'k--', lw=10, label="Ocean Volume Limit")

    # plot array
    plt.plot(x_values, array[0, :], label="Sharky", color='r')
    plt.plot(x_values, array[1, :], label="Fishy", color='b')

    plt.title(results)
    plt.ylabel("Population")
    plt.xlabel("Iterations")

    plt.legend()
    plt.show()


def draw_plots(population_sizes, ocean_size, results):
    import numpy as np

    population_sizes = np.array(population_sizes, dtype=float).T
    plot_normal(population_sizes, ocean_size, results)

    data_reduced = population_sizes[:, ::Sharky.default_timelimits[0]/3]
    data_reduced[:, -1] = population_sizes[:, -1]
    plot_path(data_reduced, ocean_size, results)


# ====== Experimenting ======


def print_experiment(f, ocean, turn):
    f.write("Turn {} \n".format(turn))
    f.write(ocean.in_ascii())
    f.write("\n")
    f.write(str(ocean.count_creatures()))
    f.write("\n\n")


def fancy_defaultdict_repr(dict_):
    string = str(dict_)
    return string[26:-1]


def ocean_experiment(size, iterations_number, ascii_output=None,
                     make_plots=False):
    Fishy.default_timelimits = [10, 20]
    Sharky.default_timelimits = [60, 15]
    fishy_initial_size = int(size ** 1.5)
    sharky_initial_count = size

    population_sizes = []
    ocean = Ocean(size=size)

    ocean.populate_many(kind=Fishy._kind(), count=fishy_initial_size)
    ocean.populate_many(kind=Sharky._kind(), count=sharky_initial_count)
    creature_counter = ocean.count_creatures()
    population_sizes.append(creature_counter.values())

    # ocean.populate_one(kind=Ktulhu._kind())

    for iterations in xrange(iterations_number):
        ocean.make_round()
        if ascii_output is not None:
            print_experiment(ascii_output, ocean, iterations)
        population_sizes.append(creature_counter.values())
        if iterations % 100 == 0:
            print iterations, "iterations done"
        if not all(creature_counter.values()):
            break
    iterations += 1

    result_string = "{} iterations with ocean_size of {}\n".format(
        iterations, size)
    result_string += "final result is {}".format(
        fancy_defaultdict_repr(ocean.count_creatures()))
    print result_string

    if make_plots:
        draw_plots(population_sizes, size, result_string)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--size', type=int, default=5, help="ocean size")
    parser.add_argument('--iter', type=int, default=100,
                        help="number of iterations")
    parser.add_argument('--ascii', nargs='?', default=None, const="output.txt",
                        help="filename for ascii output")
    parser.add_argument('--plots', nargs='?', default=False, const=True,
                        help="make plots with matplotlib and numpy")
    args = parser.parse_args()

    ascii_filename, make_plots = args.ascii, args.plots
    size, iterations = args.size, args.iter

    if ascii_filename is not None:
        with open(ascii_filename, 'w') as output_file:
            ocean_experiment(size=size, iterations_number=iterations,
                             ascii_output=output_file, make_plots=make_plots)
    else:
        ocean_experiment(size=size, iterations_number=iterations,
                         make_plots=make_plots)


# ====== Testing ======


def test_fishy_creation():
    ocean = Ocean(size=5)
    pos = ocean._get_random_free_position()
    new_fishy = DefaultCreaturesFactory.create(Fishy._kind(), pos)
    # print new_fishy
    return new_fishy


def test_fishy_behavior(fishy):
    for tries in range(10):
        fishy._try_move_or_eat()
        print fishy


def test_sharky_creation():
    ocean = Ocean(size=5)
    pos = ocean._get_random_free_position()
    sharky = DefaultCreaturesFactory.create(Sharky._kind(), pos)
    return sharky


def test_eatability():
    ocean = Ocean(size=10)
    pos1 = ModularOceanGridPosition(grid=ocean, array=[0, 0])
    pos2 = ModularOceanGridPosition(grid=ocean, array=[1, 2])
    shark = DefaultCreaturesFactory.create(Sharky._kind(), pos2)
    fish = DefaultCreaturesFactory.create(Fishy._kind(), pos1)

    print fish in ocean, shark in ocean
    print fish, shark
    step_count = 0
    encountering = False

    while(fish in ocean and shark in ocean and step_count < 50):
        ocean.make_round()
        print ocean.in_ascii()
        if (shark._position == fish._position):
            encountering = True
            print "Happy meal!"
        print fish, shark
        step_count += 1

    print fish in ocean, shark in ocean
    print step_count, encountering


# ====== Launching main() ======


if __name__ == "__main__":
    main()
