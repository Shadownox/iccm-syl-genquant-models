""" Creates the mReasoner prediction cache.

"""

import sys
import logging

import numpy as np
import ccobra
import tqdm

import mreasoner

# Define encoded syllogism names
SYLLOGISMS_gen = []
for _prem1 in ['A', 'T', 'D', 'I', 'E', 'O']:
    for _prem2 in ['A', 'T', 'D', 'I', 'E', 'O']:
        for _fig in ['1', '2', '3', '4']:
            SYLLOGISMS_gen.append(_prem1 + _prem2 + _fig)

RESPONSES_gen = []
for _quant in ['A', 'T', 'D', 'I', 'E', 'O']:
    for _direction in ['ac', 'ca']:
        RESPONSES_gen.append(_quant + _direction)
RESPONSES_gen.append('NVC')

def syllog_to_premises(syllog):
    template_quant = {
        'A': 'All {} are {}',
        'T': 'Most {} are {}',
        'D': 'Most {} are not {}',
        'I': 'Some {} are {}',
        'E': 'No {} are {}',
        'O': 'Some {} are not {}'
    }

    template_fig = {
        '1': [['A', 'B'], ['B', 'C']],
        '2': [['B', 'A'], ['C', 'B']],
        '3': [['A', 'B'], ['C', 'B']],
        '4': [['B', 'A'], ['B', 'C']]
    }

    prem1 = template_quant[syllog[0]].format(*template_fig[syllog[-1]][0])
    prem2 = template_quant[syllog[1]].format(*template_fig[syllog[-1]][1])
    return [prem1, prem2]

def generate_cache(fit_its, n_samples):
    # Initialize mReasoner interface
    cloz = mreasoner.ClozureCL()
    mreas_path = mreasoner.source_path()
    mr = mreasoner.MReasoner(cloz.exec_path(), mreas_path)

    #SYLLOGISMS_only_gen = [x for x in SYLLOGISMS_gen if x not in ccobra.syllogistic.SYLLOGISMS]

    # Generate predictions
    cache = None
    with tqdm.tqdm(total=fit_its ** 3 * 6) as pbar:
        cache = np.zeros((fit_its, 6, fit_its, fit_its, 144, 13))
        for idx_epsilon, p_epsilon in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[0], fit_its)):
            #print('epsilon:', p_epsilon)
            for idx_lambda, p_lambda  in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[1], 6)):
                #print('   lambda:', p_lambda)
                for idx_omega, p_omega in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[2], fit_its)):
                    #print('      omega:', p_omega)
                    for idx_sigma, p_sigma in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[3], fit_its)):
                        #print('         sigma:', p_sigma)

                        param_dict = {
                            'epsilon': p_epsilon,
                            'lambda': p_lambda,
                            'omega': p_omega,
                            'sigma': p_sigma
                        }

                        # Generate mReasoner prediction matrix
                        pred_mat = np.zeros((144, 13))

                        failed_state = False
                        for syl_idx, syllog in enumerate(SYLLOGISMS_gen):
                            #print('                 ', syllog)
                            premises = syllog_to_premises(syllog)

                            for _ in range(n_samples):
                                #print('                     ', _)
                                # Obtain mReasoner prediction
                                predictions = mr.query(premises, param_dict=param_dict)
                                if not predictions:
                                    print(syllog, str({x: y for x, y in param_dict.items()}))
                                    failed_state=True
                                    pred_mat = np.zeros((144, 13))
                                    break
                                for pred in predictions:
                                    if pred == "Mac":
                                        pred = "Tac"
                                    if pred == "Mca":
                                        pred = "Tca"
                                    if pred == "Ma-c":
                                        pred = "Dac"
                                    if pred == "Mc-a":
                                        pred = "Dca"
                                    if pred in RESPONSES_gen:
                                        pred_mat[syl_idx, RESPONSES_gen.index(pred)] += 1 / len(predictions)
                            if failed_state:
                                print("Skipping all tasks for parameters")
                                break

                        cache[idx_epsilon, idx_lambda, idx_omega, idx_sigma] = pred_mat

                        # Update progress
                        pbar.update(1)

    mr.terminate()
    return cache



if __name__ == '__main__':

    if len(sys.argv) != 4:
        sys.argv.append('6')
        sys.argv.append('1')
        sys.argv.append('cache_gen.npy')


    if len(sys.argv) != 4:
        print('usage: python create_cache_genquant.py <fit_its> <n_samples> <out_file>')
        exit()

    logging.basicConfig(level=logging.INFO)

    fit_its = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    out_file = sys.argv[3]

    cache = generate_cache(fit_its, n_samples)
    np.save(out_file, cache)
