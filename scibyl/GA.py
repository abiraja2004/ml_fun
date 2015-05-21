import random


class GA(object):
    def __init__(self, match_string, population_size=20, generations=10000):
        self.optimal = match_string
        self.dna_size = len(self.optimal)
        self.population_size = population_size
        self.generations = generations

        self.population = []
        for i in range(self.population_size):
            organism = ""
            for c in range(self.dna_size):
                organism += self.random_char()
            self.population.append(organism)

    # objective function
    def fitness(self, organism):
        fitness = 0
        for c in range(self.dna_size):
            fitness += abs(ord(organism[c]) - ord(self.optimal[c]))
        return fitness
 
    def mutate(self, organism):
        organism_out = ""
        mutation_chance = 100
        for c in range(self.dna_size):
            if int(random.random()*mutation_chance) == 1:
                organism_out += self.random_char()
            else:
                organism_out += organism[c]
        return organism_out
 
    def crossover(self, parent1, parent2):
        pos = int(random.random()*self.dna_size)
        return parent2[:pos]+parent2[pos:], parent2[:pos]+parent1[pos:]

    def evolve(self):
        for generation in range(self.generations):
            print("Generation %s... Random sample: '%s'... Fitness: %s" % (generation, self.population[0],
                                                                           self.fitness(self.population[0])))
            weighted_population = []

            for individual in self.population:
                fitness_val = self.fitness(individual)

                # Generate the (individual,fitness) pair, taking in account whether or
                # not we will accidently divide by zero.
                if fitness_val == 0:
                    pair = (individual, 1.0)
                else:
                    pair = (individual, 1.0/fitness_val)

                weighted_population.append(pair)

            self.population = []

            for _ in range(int(self.population_size/2)):
                ind1 = self.weighted_choice(weighted_population)
                ind2 = self.weighted_choice(weighted_population)

                ind1, ind2 = self.crossover(ind1, ind2)

                self.population.append(self.mutate(ind1))
                self.population.append(self.mutate(ind2))

        fit_dict = {}
        for individual in self.population:
            fit_dict[individual] = self.fitness(individual)

        fittest = min(fit_dict, key=fit_dict.get)
        print("Fittest String: %s, Fitness: %s" % (fittest, fit_dict[fittest]))


    @staticmethod
    def weighted_choice(items):
        weight_total = sum((item[1] for item in items))
        n = random.uniform(0, weight_total)
        for item, weight in items:
            if n < weight:
                return item
            n = n - weight

    @staticmethod
    def random_char():
        return chr(int(random.randrange(32, 126, 1)))

def demo():
    GA
