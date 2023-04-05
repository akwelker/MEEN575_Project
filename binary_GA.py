# My implementation of a Binary GA optimizer
# -- Adam Welker

import numpy as np
import random
import os


# Terminal Printing colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# A Class that can perform Binary GA
class Binary_GA:

    MIN = None
    MAX = None
    NUM_BITS = None

    MUTATION_GAIN = 0.005

    VERBOSE = True

    def __init__(self, min, max, num_bits, verbose = False, mutation_gain = 0.005) -> None:
        
        self.MIN = min
        self.MAX = max
        self.NUM_BITS = num_bits

        self.PRECISION = (self.MAX - self.MIN) / (2**self.NUM_BITS - 1)

        self.VERBOSE = verbose

        self.MUTATION_GAIN = mutation_gain

        if self.VERBOSE:

            print(bcolors.OKGREEN + '--- GA Optimizer Initialized ---' + bcolors.ENDC)
            print(bcolors.OKBLUE + 'Min x bound: ' + bcolors.ENDC + str(self.MIN))
            print(bcolors.OKBLUE + 'Max x bound: ' + bcolors.ENDC + str(self.MAX))
            print(bcolors.OKBLUE + 'Precision: ' + bcolors.ENDC + str(self.PRECISION))
            print(bcolors.OKBLUE + 'Mutation Gain: ' + bcolors.ENDC + str(self.MUTATION_GAIN))



    # =============== Decoding methods ======================
    def binary_decode(self, binary_val):

        x = self.MIN

        for i in range(0,len(binary_val)):

            if binary_val[i] == '1':

                x += 2**(len(binary_val)-1 - i) * self.PRECISION 

        return x



    def get_real_x(self, genome):

        x_real  = []

        for gene in genome:

            val = self.binary_decode(gene)

            x_real.append(val)

        return x_real


    def get_obj_funct(self, func, genome):

        x_real = self.get_real_x(genome)

        return func(x_real)


    # =================== Selection =============================

    # Implements roulette wheel 
    def select_pairs(self, func, gene_pool):

        obj_metrics = []
        obj_functions =  []

        next_gen_parents = []

        # Extract objective function measures
        for i in gene_pool:

            objective = self.get_obj_funct(func,i)

            obj_metrics.append([i, objective])
            obj_functions.append(objective)
        
        # Get performance measure for routlette wheel
        f_low = min(obj_functions)
        f_hi = max(obj_functions)
        delta_f = 1.1*f_hi - 0.1*f_low

        total_F = 0

        for i in range(0, len(obj_metrics)):

            Fj = (-obj_metrics[i][1] + delta_f)/(max([1, delta_f - f_low]))

            assert Fj >= 0

            obj_metrics[i].append(Fj)

            total_F += Fj

        #Now sort the points according to F
        obj_metrics.sort(key = lambda x: x[2])

        sum_fj = 0
        #Now find Sj for each point
        for i in obj_metrics:

            sum_fj += i[2] # Add current fj to sum of fj

            i.append(sum_fj/total_F)

        # now roll the roulette wheel

        random.seed(obj_metrics[0][2]) # pick the first given objective value as a random seed

        for i in range(0, len(obj_metrics)):

            r = random.random()

            for j in range(0, len(obj_metrics)):

                if (r <= obj_metrics[j][3] and r > obj_metrics[j - 1][3]) or (r <= obj_metrics[j][3] and j == 0):
                    
                    next_gen_parents.append(obj_metrics[j][0])

        assert len(next_gen_parents) == len(gene_pool)

        return next_gen_parents


    # ======================= Cross Over =========================

    #Given a two genome parents,
    # will cross breed children. Uses single point methodology
    def cross_breed(self, genome_1, genome_2):


        child_1 = []
        child_2 = []

        for i in range(0, len(genome_1)):

            setpoint = int(random.random() * len(genome_1[0]))

            if setpoint == 0:

                setpoint = 1

            chromezone_1 = genome_1[i][0: setpoint] + genome_2[i][setpoint: len(genome_2[i])]
            chromezone_2 = genome_2[i][0: setpoint] + genome_1[i][setpoint: len(genome_1[i])]

            assert len(chromezone_1) == len(chromezone_2) # Can't be having any genetic deformities here

            child_1.append(chromezone_1)
            child_2.append(chromezone_2)

        return child_1, child_2

    # Given the full list of parents, will 
    def cross_over(self, parents):

        next_gen = []

        for i in range(0,len(parents),2):

            p1 = parents[i]
            p2 = parents[i + 1]

            for child in self.cross_breed(p1,p2): next_gen.append(child)

        return next_gen

    # =========================== Mutation ===================================

    def mutate_genome(self, genome):

        p = self.MUTATION_GAIN # tunable mutation parameter

        out_gene = [] # the genome we return

        for chromosome in genome:

            new_chrome = ''
            
            for i in range(0, len(chromosome)):

                r = random.random() # see if we mutate

                if r <= p: # if we do add the opposite binary value

                    gene = chromosome[i]

                    if gene == '0':

                        new_chrome += '1'

                    else:

                        new_chrome += '0'

                else: # if not, just add the next recurring gene to the chromosome

                    new_chrome += chromosome[i]

            
            out_gene.append(new_chrome)

        
        return out_gene


    #Given the population of points, will mutate all population
    def mutate_population(self, population):

        new_population = []

        for genome in population: 

            new_genome = self.mutate_genome(genome)

            new_population.append(new_genome)

        return new_population


    # Find minimum in a population
    def get_population_minimum(self, func, population):

        x_star = None
        y_star = np.inf

        for member in population:

            eval = self.get_obj_funct(func, member)

            if eval < y_star:

                x_star = self.get_real_x(member)
                y_star = eval

        return x_star, y_star


    def GA_optimization(self, func, max_itr, size, x_size):

        # Ensure even population size:
        assert size >= 4

        if size % 2 != 0:

            size += 1

        # Generate initial population
        population = []
        
        for i in range(0, size):

            x_vector = []

            for k in range(0, x_size):

                x_k = ''

                for j in range(0,self.NUM_BITS):

                    x_k += str(round(random.random()))

                x_vector.append(x_k)

            population.append(x_vector)


        # Simulate generational evolution
        k = 0
        while k < max_itr:

            parents = self.select_pairs(func,population)

            new_population = self.cross_over(parents)

            new_population = self.mutate_population(new_population)

            population = new_population

            k += 1

            if self.VERBOSE:

                percentage_completion = round(k/max_itr * 100, 1)
                os.system('cls')
                print(f'GEN: {k} | [' + bcolors.OKGREEN + f'{percentage_completion}%' + bcolors.ENDC +']')

        
        # return the minimum element
        final_population = []
        for i in population:

            final_population.append(self.get_real_x(i))

        result = list(self.get_population_minimum(func,population))

        result.append(final_population)

        return result
    

# Main Method Optimization
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from eggshell import egg_shell

    optimizer = Binary_GA(-1.5, 1.5, 32, verbose=True)

    optimizer.VERBOSE = False

    result = optimizer.GA_optimization(egg_shell,5000,100, 2)

    # Make a plot of the eggshell function
    x1_space = 1000
    x2_space = 1000

    x1_domain = np.linspace(-2.0,2.0,x1_space)
    x2_domain = np.linspace(-2.0,2.0,x2_space)

    egg_shell_domain = np.zeros((x1_space, x2_space))

    # Evaluate each function at all points
    for i in range(0,len(x1_domain)):

        for j in range(0,len(x2_domain)):

            egg_shell_domain[j][i] = egg_shell(np.array([x1_domain.item(i), x2_domain.item(j)]))


    min_location = np.unravel_index(np.argmin(egg_shell_domain), egg_shell_domain.shape)
    MIN = -1.5
    MAX = 1.5

    plt.figure(1, figsize=(6.5,6.5))
    plt.contour(x1_domain,x2_domain,egg_shell_domain, 50)
    plt.plot([MIN,MIN, MAX, MAX, MIN],[MIN,MAX, MAX, MIN, MIN],'k--')

    x_vals = []
    y_vals = []

    for point in result[2]:

        x_vals.append(point[0])
        y_vals.append(point[1])

    plt.plot(x_vals,y_vals,'ob')

    plt.plot(x1_domain[min_location[0]], x2_domain[min_location[1]], "*r")

    plt.legend(['X Bounds', 'X Population', 'X*'])
    plt.show()


    print(f'x* = {result[0]}')
    print(f'minimum is: {result[1]}')