import os
import time
import math
import numpy as np
from commgnas.search_algorithm.graphpastune import utils
from commgnas.parallel import ParallelOperater

class GraphPASTune(object):

    def __init__(self,
                 sharing_num,
                 mutation_num,
                 initial_num,
                 search_space,
                 search_epoch):

        self.sharing_num = sharing_num
        self.mutation_num = mutation_num
        self.initial_num = initial_num
        self.mutation_num_change_threshold = math.ceil(search_epoch/mutation_num)

        self.initial_gnn_architecture_embedding_list = []
        self.initial_gnn_architecture_list = []
        self.initial_gnn_architecture_performance = []

        self.search_space = search_space.space_getter()
        self.stack_gcn_architecture = search_space.stack_gcn_architecture

        self.random_initialize_population()

    def search(self,
               total_pop,
               sharing_population,
               sharing_performance,
               mutation_selection_probability,
               epoch):

        print(35*"=", "graphpastune search", 35*"=")
        print("sharing population:\n", sharing_population)
        print("sharing performance:\n", sharing_performance)
        print("[sharing population performance] Mean/Median/Best:\n",
              np.mean(sharing_performance),
              np.median(sharing_performance),
              np.max(sharing_performance))

        # select parents based on wheel strategy.
        parents = self.selection(sharing_population, sharing_performance)
        print("parents:\n", parents)

        # mutation based on mutation_select_probability
        children = self.mutation(parents,
                                 mutation_selection_probability,
                                 total_pop,
                                 epoch)

        print("children:\n", children)

        total_pop = total_pop + children
        return children, total_pop

    def random_initialize_population(self):

        print(35*"=", "population initializing based on random strategy", 35*"=")

        while len(self.initial_gnn_architecture_embedding_list) < self.initial_num:
            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding(self.search_space,
                                                                                          self.stack_gcn_architecture)
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space,
                                                                        self.stack_gcn_architecture)
            # gnn architecture genetic embedding based on number
            self.initial_gnn_architecture_embedding_list .append(gnn_architecture_embedding)
            self.initial_gnn_architecture_list.append(gnn_architecture)

    def selection(self,
                  population,
                  performance):
        print(35*"=", "select parents based on wheel strategy", 35*"=")

        fitness = np.array(performance)
        fitness_probility = fitness / sum(fitness)
        fitness_probility = fitness_probility.tolist()

        index_list = [index for index in range(len(fitness))]
        parents = []
        parent_index = np.random.choice(index_list, self.sharing_num, replace=False, p=fitness_probility)
        for index in parent_index:
            parents.append(population[index].copy())
        return parents

    def mutation(self,
                 parents,
                 mutation_selection_probability,
                 total_pop,
                 epoch):

        if (epoch % self.mutation_num_change_threshold) == 0:

            self.mutation_num = self.mutation_num - 1
            if self.mutation_num == 0:
                self.mutation_num = 1

        print("i:", epoch)
        print("self_mutation_num:", self.mutation_num)

        print(35 * "=", "mutation based on mutation_select_probability", 35 * "=")
        for index in range(len(parents)):

            # stopping until sampling the new gnn architecture which not in the total_pop
            while parents[index] in total_pop:
                # confirming mutation point in the gnn architecture genetic list based on information_entropy_probability
                position_to_mutate_list = np.random.choice([gene for gene in range(len(parents[index]))],
                                                           self.mutation_num,
                                                           replace=False,
                                                           p=mutation_selection_probability)

                for mutation_index in position_to_mutate_list:
                    mutation_space = self.search_space[self.stack_gcn_architecture[mutation_index]]
                    parents[index][mutation_index] = np.random.randint(0, len(mutation_space))

        children = parents

        return children

    def updating(self,
                 sharing_children,
                 sharing_children_val_performance_list,
                 sharing_population,
                 sharing_performance
                 ):

        print(35*"=", "updating", 35*"=")
        print("before sharing_performance:\n", sharing_performance)

        # calculating the average fitness based on top k gnn architecture in sharing population
        _, top_performance = utils.top_population_select(sharing_population,
                                                 sharing_performance,
                                                 top_k=self.sharing_num)
        avg_performance = np.mean(top_performance)

        index = 0
        for performance in sharing_children_val_performance_list:
            if performance > avg_performance:
                sharing_performance.append(performance)
                sharing_population.append(sharing_children[index])
                index += 1
            else:
                index += 1
        print("after sharing_performance:\n", sharing_performance)
        return sharing_population, sharing_performance

class Search(object):

    def __init__(self, data, search_parameter, gnn_parameter, search_space):

        self.data = data
        self.search_parameter = search_parameter

        # parallel estimation operator initialize
        self.parallel_estimation = ParallelOperater(data, gnn_parameter)

        self.search_space = search_space

    def search_operator(self):

        print(35 * "=", "graphpastune search start", 35 * "=")
        time_initial = time.time()
        searcher_list = []
        for index in range(int(self.search_parameter["parallel_num"])):
            searcher =  GraphPASTune(sharing_num=int(self.search_parameter["sharing_num"]),
                                      mutation_num=eval(self.search_parameter["mutation_num"])[index],
                                      initial_num=int(self.search_parameter["initial_num"]),
                                      search_space=self.search_space,
                                      search_epoch=int(self.search_parameter["search_epoch"]))
            searcher_list.append(searcher)

        # fitness calculate / fitness merge
        total_pop = []
        total_performance = []
        for searcher in searcher_list:
            gnn_architecture_list = searcher.initial_gnn_architecture_list

            # parallel estimation
            result = self.parallel_estimation.estimation(gnn_architecture_list)

            for performance in result:
                searcher.initial_gnn_architecture_performance += [performance]

            total_pop = total_pop + searcher.initial_gnn_architecture_embedding_list
            total_performance = total_performance + searcher.initial_gnn_architecture_performance

        time_initial = time.time() - time_initial

        # initial gnn architecture time cost record
        path = os.path.split(os.path.realpath(__file__))[0][:-38] + "logger/graphpastune_logger/"
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save_initial(path, self.data.data_name + "_initial_time.txt", time_initial)

        # sharing population select based on top strategy
        print(35 * "=", "sharing population select", 35 * "=")
        sharing_population, sharing_performance = utils.top_population_select(total_pop,
                                                                              total_performance,
                                                                              top_k=self.search_parameter["sharing_num"])
        # mutation select probability vector calculate
        sharing_population_temp = sharing_population.copy()
        mutation_selection_probability = utils.mutation_selection_probability(sharing_population_temp,
                                                                              self.search_space.stack_gcn_architecture)
        print(35 * "=", "mutation select probability vector:", 35 * "=")
        print(mutation_selection_probability)

        print(35 * "=", "multiple mutation search_algorithm start", 35 * "=")

        time_search_list = []
        epoch = []

        for i in range(int(self.search_parameter["search_epoch"])):
            
            time_search = time.time()
            # parent select and mutate based on mutation select probability and mutation intensity
            sharing_children_embedding = []
            for searcher in searcher_list:
                
                children, total_pop = searcher.search(total_pop,
                                                      sharing_population,
                                                      sharing_performance,
                                                      mutation_selection_probability,
                                                      i+1)
                sharing_children_embedding = sharing_children_embedding + children

            # sharing children decoding
            sharing_children_architecture = []
            sharing_children_val_performance_list = []

            for gnn_architecture_embedding in sharing_children_embedding:
                
                gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                            self.search_space.space_getter(),
                                                                            self.search_space.stack_gcn_architecture)
                sharing_children_architecture.append(gnn_architecture)

            # sharing children parallel estimation
            result = self.parallel_estimation.estimation(sharing_children_architecture)

            for performance in result:
                
                sharing_children_val_performance_list += [performance]

            # sharing population updating
            sharing_population, sharing_performance = searcher_list[0].updating(sharing_children_embedding,
                                                                             sharing_children_val_performance_list,
                                                                             sharing_population,
                                                                             sharing_performance)

            #  mutation select probability vector recalculate
            sharing_population_temp = sharing_population.copy()
            mutation_selection_probability = utils.mutation_selection_probability(sharing_population_temp, self.search_space.stack_gcn_architecture)
            time_search_list.append(time.time()-time_search)
            epoch.append(i+1)

            # model architecture and val performance record
            if not os.path.exists(path):
                os.makedirs(path)
            utils.experiment_graphpas_data_save(path,
                                                self.data.data_name + "_search_epoch_" + str(i+1) + ".txt",
                                                sharing_population,
                                                sharing_performance,
                                                self.search_space.space_getter(),
                                                self.search_space.stack_gcn_architecture)

        index = sharing_performance.index(max(sharing_performance))
        best_val_architecture = sharing_population[index]
        best_val_architecture = utils.gnn_architecture_embedding_decoder(best_val_architecture,
                                                                         self.search_space.space_getter(),
                                                                         self.search_space.stack_gcn_architecture)
        best_performance = max(sharing_performance)
        print("Best GNN Architecture:\n", best_val_architecture)
        print("Best VAL Performance:\n", best_performance)

        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save(path,
                                   self.data.data_name + "_search_time.txt",
                                   epoch,
                                   time_search_list)

if __name__=="__main__":
    pass