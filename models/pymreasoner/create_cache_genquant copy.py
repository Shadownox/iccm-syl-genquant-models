""" Creates the mReasoner prediction cache.

"""

import sys
import logging
from joblib import Parallel, delayed

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

    # Generate predictions
    cache = None
    with tqdm.tqdm(total=fit_its ** 3 * 7) as pbar:
        cache = np.zeros((fit_its, fit_its, fit_its, fit_its, 144, 13))
        for idx_epsilon, p_epsilon in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[0], fit_its)):
            print('epsilon:', p_epsilon)
            for idx_lambda, p_lambda  in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[1], 7)):
                print('   lambda:', p_lambda)
                for idx_omega, p_omega in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[2], fit_its)):
                    print('      omega:', p_omega)
                    for idx_sigma, p_sigma in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[3], fit_its)):
                        print('         sigma:', p_sigma)
                        #progress = (idx_epsilon * (fit_its ** 3) + idx_lambda * (fit_its ** 2) + idx_omega * (fit_its) + idx_sigma)
                        #print('Current progress: {:.2f}%'.format(progress * 100))

                        param_dict = {
                            'epsilon': p_epsilon,
                            'lambda': p_lambda,
                            'omega': p_omega,
                            'sigma': p_sigma
                        }

                        # Generate mReasoner prediction matrix
                        pred_mat = np.zeros((144, 13))
                        for syl_idx, syllog in enumerate(SYLLOGISMS_gen[52:60]):
                            print('                 ', syllog)
                            premises = syllog_to_premises(syllog)

                            for _ in range(n_samples):
                                print('                     ', _)
                                # Obtain mReasoner prediction
                                predictions = mr.query(premises, param_dict=param_dict)
                                if not predictions:
                                    print(syllog, str({x: y for x, y in param_dict.items()}))
                                for pred in predictions:
                                    if pred in RESPONSES_gen:
                                        pred_mat[syl_idx, RESPONSES_gen.index(pred)] += 1 / len(predictions)

                        cache[idx_epsilon, idx_lambda, idx_omega, idx_sigma] = pred_mat

                        # Update progress
                        pbar.update(1)

    mr.terminate()
    return cache

def generate_cache_multi(fit_its, n_samples):
    # Initialize mReasoner interface
    cloz = mreasoner.ClozureCL()
    mreas_path = mreasoner.source_path()
    mr = mreasoner.MReasoner(cloz.exec_path(), mreas_path)

    # Generate predictions
    cache = None
    cache = np.zeros((6, 7, 6, 6, 144, 13))
            
    epsilons = np.linspace(*mreasoner.PARAM_BOUNDS[0], fit_its)
    Parallel(n_jobs=2, backend='multiprocessing')(delayed(cache_loop)(idx_epsilon, p_epsilon, cache, mr) for idx_epsilon, p_epsilon in enumerate(epsilons))

    mr.terminate()
    return cache

def cache_loop(idx_epsilon, p_epsilon, cache, mr):
    print('epsilon:', p_epsilon)
    for idx_lambda, p_lambda  in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[1], 7)):
        print('   lambda:', p_lambda)
        for idx_omega, p_omega in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[2], fit_its)):
            print('      omega:', p_omega)
            for idx_sigma, p_sigma in enumerate(np.linspace(*mreasoner.PARAM_BOUNDS[3], fit_its)):
                print('         sigma:', p_sigma)
                #progress = (idx_epsilon * (fit_its ** 3) + idx_lambda * (fit_its ** 2) + idx_omega * (fit_its) + idx_sigma)
                #print('Current progress: {:.2f}%'.format(progress * 100))
                param_dict = {
                    'epsilon': p_epsilon,
                    'lambda': p_lambda,
                    'omega': p_omega,
                    'sigma': p_sigma
                }
                # Generate mReasoner prediction matrix
                pred_mat = np.zeros((144, 13))
                for syl_idx, syllog in enumerate(SYLLOGISMS_gen[28:36]):
                    print('                 ', syllog)
                    premises = syllog_to_premises(syllog)
                    for _ in range(n_samples):
                        print('                     ', _)
                        # Obtain mReasoner prediction
                        predictions = mr.query(premises, param_dict=param_dict)
                        if not predictions:
                            print(syllog, str({x: y for x, y in param_dict.items()}))
                        for pred in predictions:
                            if pred in RESPONSES_gen:
                                pred_mat[syl_idx, RESPONSES_gen.index(pred)] += 1 / len(predictions)
                cache[idx_epsilon, idx_lambda, idx_omega, idx_sigma] = pred_mat
    
    return cache

if __name__ == '__main__':

    if len(sys.argv) != 4:
        sys.argv.append('6')
        sys.argv.append('1')
        sys.argv.append('cache_gen_2.npy')


    if len(sys.argv) != 4:
        print('usage: python create_cache_genquant.py <fit_its> <n_samples> <out_file>')
        exit()

    logging.basicConfig(level=logging.INFO)

    fit_its = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    out_file = sys.argv[3]

    cache = generate_cache(fit_its, n_samples)
    np.save(out_file, cache)
