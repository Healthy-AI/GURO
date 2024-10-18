from tqdm import tqdm
from helper_functions import get_ranking_loss


class Simulation:
    '''
    Generic simulation class
    '''

    def __init__(self, algorithm, env, test_set=None, collect_model_info=False):
        self.algorithm = algorithm
        self.env = env
        self.test_set = test_set
        self.collect_model_info = collect_model_info

    def run(self, n_steps, eval_steps=10):
        '''
        Run simulation for n_steps
        '''
        losses = []
        test_losses = []
        model_info = []
        aux_results = {"comparisons": [], "aux": []}
        incorrect_count = 0
        pbar = tqdm(range(n_steps),)

        initial_estimate = self.algorithm.ordering()
        loss = self.env.get_ranking_loss(initial_estimate)
        losses.append(loss)

        for t in pbar:
            action, aux = self.algorithm.act()
            aux_results["comparisons"].append((action[0], action[1]))
            reward, correct_comp = self.env.step(action)
            if not correct_comp:
                incorrect_count += 1
            self.algorithm.update(*action, reward)
            aux_results["aux"].append(aux)
            if t % eval_steps == 0:
                estimated_order = self.algorithm.ordering()
                loss = self.env.get_ranking_loss(estimated_order)
                losses.append(loss)

                pbar.set_description("Current loss: " + str(loss))

                if self.test_set is not None:
                    test_loss = get_ranking_loss(
                        self.test_set, self.algorithm.theta_hat, self.env.theta)
                    test_losses.append(test_loss)

                if self.collect_model_info:
                    model_info.append(self.algorithm.get_model_info())

        aux_results["test_losses"] = test_losses
        aux_results["model_info"] = model_info

        aux_results["incorrect_count"] = incorrect_count
        return losses, aux_results
