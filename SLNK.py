import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def ind2gen(index,n):
    genotype = np.zeros(n)
    while n > 0:
        n = n - 1
        if index % 2 == 0:
            genotype[n] = 0
        else:
            genotype[n] = 1
        index = index // 2
    return genotype

def gen2ind(genotype, n):
    i = 0
    index = 0
    while i < n:
        index += genotype[i]*(2**(n-i-1))
        i += 1
    return int(index)

class NKLandscape:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.maxfit = 0.0
        self.minfit = 1000000
        self.interactions = np.random.rand(n*(2**(k+1))).reshape(n,2**(k+1))
        self.fit_table = np.zeros(2**n)
        self.best = 0
        for solution in range(2**n):
            genotype = ind2gen(solution,n)
            fit = self.calcFit(genotype)
            self.fit_table[solution] = fit
            if fit > self.maxfit:
                self.maxfit = fit
                self.best = genotype
            if fit < self.minfit:
                self.minfit = fit
        self.fit_table = (self.fit_table - self.minfit)/(self.maxfit-self.minfit)
        self.fit_table = self.fit_table**8

    def calcFit(self, genotype):
        fit = 0.0
        for gene in range(self.n):  # Calculate contribution of each gene in the current solution
            subgen = []
            for nbr in range(self.k + 1):  # Identify neighbors
                nbr_ind = (gene + nbr) % self.n
                subgen.append(genotype[nbr_ind])
            ind = gen2ind(subgen, self.k + 1)  # Calculate epistatic interactions with each neighbor
            fit += self.interactions[gene][ind]
        return fit

    def fitness(self, genotype):
        index = gen2ind(genotype,self.n)
        return self.fit_table[index]

class Population:
    def __init__(self, popsize, n, landscape, community=False):
        self.popsize = popsize          # population size
        self.ng = n                     # number of genes
        self.share_rate = 1.0           # recombination rate
        self.share_radius = popsize - 1 # how big your "neighborhood" is
        self.mut_rate = 0.0             # Percentage of the solution that is mutated during a copy/share
        self.landscape = landscape      # NK landscape
        self.genotypes = np.random.randint(2,size=popsize*n).reshape(popsize,n)
        self.shared = np.zeros(popsize,dtype=int)  # and who just shared
        self.community = community

    def set_community(self, ncomm, in_group, out_group):
        pmatrix = np.eye(ncomm) * in_group + np.where(np.eye(ncomm) == 1, 0, 1) * out_group
        csizes = [int(self.popsize / ncomm)] * ncomm
        graph = nx.stochastic_block_model(sizes=csizes, p=pmatrix)
        self.adj_matrix = nx.to_numpy_array(graph)

    def set_small_world(self, k = 4, p = 0.1):
        graph = nx.connected_watts_strogatz_graph(self.popsize, k, p)
        self.adj_matrix = nx.to_numpy_array(graph)

    def set_scale_free(self, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0):
        graph = nx.scale_free_graph(self.popsize, alpha, beta, gamma, delta_in, delta_out).to_undirected()
        self.adj_matrix = nx.to_numpy_array(graph)

    def set_pop(self, genotypes):
        self.genotypes = genotypes.copy()

    def avgfit(self):
        avg = 0
        for i in range(self.popsize):
            avg += self.landscape.fitness(self.genotypes[i])
        return avg / self.popsize

    def stats(self):
        avg = 0
        best = 0
        for i in range(self.popsize):
            fit = self.landscape.fitness(self.genotypes[i])
            self.landscape.visited(self.genotypes[i])
            avg += fit
            if fit > best:
                best = fit
        k=0
        unique = np.ones(self.popsize)
        for i in range(self.popsize):
            for j in range(i+1,self.popsize):
                self.dist_list[k] = np.mean(np.abs(self.genotypes[i] - self.genotypes[j]))
                if (self.dist_list[k]==0):
                    unique[i]=0.0
                    unique[j]=0.0
                k+=1
        return avg/self.popsize, np.mean(self.dist_list), np.mean(unique)

    def learn(self):
        for i in range(self.popsize):
            if self.shared[i]==0:
                original_fitness = self.landscape.fitness(self.genotypes[i])
                new_genotype = self.genotypes[i].copy()
                j = np.random.randint(self.ng)
                if new_genotype[j] == 0:
                    new_genotype[j] = 1
                else:
                    new_genotype[j] = 0
                new_fitness = self.landscape.fitness(new_genotype)
                if new_fitness > original_fitness:
                    self.genotypes[i] = new_genotype.copy()

    def share(self):
        self.shared = np.zeros(self.popsize,dtype=int)
        new_genotypes = self.genotypes.copy()
        for i in range(self.popsize):
            if self.community:
                neighbors = [j for j in range(self.popsize) if self.adj_matrix[i, j] >= 1]
                j = np.random.choice(neighbors)
            else:
                j = np.random.randint(i-self.share_radius,i+self.share_radius+1) % self.popsize
                while (j == i):
                    j = np.random.randint(i-self.share_radius,i+self.share_radius+1) % self.popsize
            if self.landscape.fitness(self.genotypes[j]) > self.landscape.fitness(self.genotypes[i]):
                self.shared[i] = 1
                for g in range(self.ng):
                    if np.random.rand() <= self.share_rate:
                        new_genotypes[i][g] = self.genotypes[j][g]
                        if np.random.rand() <= self.mut_rate:
                            new_genotypes[i][g] = np.random.randint(2)
        self.genotypes = new_genotypes.copy()
