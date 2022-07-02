""" CCOBRA model wrapper for mReasoner.

"""

from http.client import responses
import sys
import collections
import threading
import copy

import ccobra
import mreasoner
import numpy as np

import time

import logging

logging.basicConfig(level=logging.INFO)

class CCobraMReasoner(ccobra.CCobraModel):
    """ mReasoner CCOBRA model implementation.

    """

    def __init__(self, name='mReasoner', n_samples=2, fit_its=5):
        """ Initializes the CCOBRA model by launching the interactive LISP subprocess.

        Parameters
        ----------
        name : str
            Name for the CCOBRA model.

        n_samples : int
            Number of samples to draw from mReasoner in order to mitigate the effects
            of its randomized inference processes.

        method : str
            Parameter optimization technique ('grid' or 'random').

        fit_its : int
            Number of iterations for the parameter optimization (grid or random). If set to 0,
            fitting is deactivated.

        num_threads : int
            Number of parallel threads to perform the parameter optimization with.

        """

        super(CCobraMReasoner, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Define encoded syllogism names
        self.SYLLOGISMS_gen = []
        for _prem1 in ['A', 'T', 'D', 'I', 'E', 'O']:
            for _prem2 in ['A', 'T', 'D', 'I', 'E', 'O']:
                for _fig in ['1', '2', '3', '4']:
                    self.SYLLOGISMS_gen.append(_prem1 + _prem2 + _fig)

        self.RESPONSES_gen = []
        for _quant in ['A', 'T', 'D', 'I', 'E', 'O']:
            for _direction in ['ac', 'ca']:
                self.RESPONSES_gen.append(_quant + _direction)
        self.RESPONSES_gen.append('NVC')

        # Launch mreasoner interface
        self.cloz = mreasoner.ClozureCL()
        self.mreas_path = mreasoner.source_path()
        self.mreasoner = mreasoner.MReasoner(self.cloz.exec_path(), self.mreas_path)

        self.params = copy.deepcopy(mreasoner.DEFAULT_PARAMS)

        # Store instance variables
        self.n_samples = n_samples
        self.fit_its = fit_its

        # Initialize auxiliary variables
        self.n_pre_train_dudes = 0
        self.pre_train_data = np.zeros((144, 13))
        self.person_train_data = np.zeros((144, 13))
        self.history = np.zeros((144, 13))

        self.start_time = None

    def __deepcopy__(self, memo):
        """ Custom deepcopy required because thread locks cannot be pickled. Deepcopy realized by
        creating a fresh instance of the mReasoner model and syncing parameters.

        Parameters
        ----------
        memo : dict
            Memo dictionary of objects already copied. Should be passed to nested deepcopy calls.

        Returns
        -------
        CCobraMReasoner
            Copied object instance.

        """

        # Create the new instance
        new = CCobraMReasoner(self.name, self.n_samples, self.fit_its)

        # Copy member variables
        new.n_pre_train_dudes = self.n_pre_train_dudes
        new.pre_train_data = self.pre_train_data
        new.person_train_data = self.person_train_data
        new.history = self.history
        new.params = self.params

        return new

    def end_participant(self, subj_id, model_log, **kwargs):
        """ When the prediction phase is finished, terminate the LISP subprocess.

        """

        # Parameterization output
        print('End Participant ({:.2f}s, {} its) id={} params={}'.format(
            time.time() - self.start_time,
            self.fit_its,
            subj_id,
            str(list(self.params.items())).replace(' ', ''),
        ))

        sys.stdout.flush()

        # Terminate the mReasoner instance
        self.mreasoner.terminate()

    def start_participant(self, **kwargs):
        """ Model setup method. Stores the time for use in end_participant().

        """

        self.start_time = time.time()

    def pre_train(self, dataset):
        """ Pre-trains the model by fitting mReasoner.

        Parameters
        ----------
        dataset : list(list(dict(str, object)))
            Training data.

        """

        # Check if fitting is deactivated
        if self.fit_its == 0 or self.evaluation_type == 'coverage':
            return

        # Extract the training data to fit mReasoner with
        self.n_pre_train_dudes = len(dataset)
        self.pre_train_data = np.zeros((144, 13))
        for subj_data in dataset:
            for task_data in subj_data:
                item = task_data['item']
                enc_task = ccobra.syllogistic_generalized.encode_task(item.task)
                enc_resp = ccobra.syllogistic_generalized.encode_response(task_data['response'], item.task)

                task_idx = self.SYLLOGISMS_gen.index(enc_task)
                resp_idx = self.RESPONSES_gen.index(enc_resp)
                self.pre_train_data[task_idx, resp_idx] += 1

        div_mask = (self.pre_train_data.sum(axis=1) != 0)
        self.pre_train_data[div_mask] /= self.pre_train_data[div_mask].sum(axis=1, keepdims=True)

        # Fit the model
        self.fit()

    def pre_train_person(self, dataset, **kwargs):
        """ Perform the person training of mReasoner.

        """

        # Check if fitting is deactivated
        if self.fit_its == 0:
            return

        # Extract the training data to fit mReasoner with
        self.person_train_data = np.zeros((144, 13))
        for task_data in dataset:
            item = task_data['item']
            enc_task = ccobra.syllogistic_generalized.encode_task(item.task)
            enc_resp = ccobra.syllogistic_generalized.encode_response(task_data['response'], item.task)

            task_idx = self.SYLLOGISMS_gen.index(enc_task)
            resp_idx = self.RESPONSES_gen.index(enc_resp)
            self.person_train_data[task_idx, resp_idx] += 1

        div_mask = (self.person_train_data.sum(axis=1) != 0)
        self.person_train_data[div_mask] /= self.person_train_data[div_mask].sum(axis=1, keepdims=True)

        # Fit the model
        self.fit()

    def fit(self):
        # Merge the training datasets
        history_copy = self.history.copy()
        div_mask = (history_copy.sum(axis=1) != 0)
        history_copy[div_mask] /= history_copy[div_mask].sum(axis=1, keepdims=True)

        train_data = self.pre_train_data + self.person_train_data + history_copy

        best_score = 0
        best_param_dicts = []

        for p_epsilon in np.linspace(*mreasoner.PARAM_BOUNDS[0], self.fit_its):
            print('epsilon:', p_epsilon)
            for p_lambda  in np.linspace(*mreasoner.PARAM_BOUNDS[1], self.fit_its):
                print('   lambda:', p_lambda)
                for p_omega in np.linspace(*mreasoner.PARAM_BOUNDS[2], self.fit_its):
                    print('      omega:', p_omega)
                    for p_sigma in np.linspace(*mreasoner.PARAM_BOUNDS[3], self.fit_its):
                        print('         sigma:', p_sigma)
                        param_dict = {
                            'epsilon': p_epsilon,
                            'lambda': p_lambda,
                            'omega': p_omega,
                            'sigma': p_sigma
                        }

                        # Generate mReasoner prediction matrix
                        pred_mat = np.zeros((144, 13))
                        for syl_idx, syllog in enumerate(self.SYLLOGISMS_gen):
                            premises = self.syllog_to_premises(syllog)

                            for _ in range(self.n_samples):
                                # Obtain mReasoner prediction
                                predictions = self.mreasoner.query(premises, param_dict=param_dict)
                                for pred in predictions:
                                    if pred in self.RESPONSES_gen:
                                        pred_mat[syl_idx, self.RESPONSES_gen.index(pred)] += 1 / len(predictions)

                        # Compare predictions with data
                        pred_mask = (pred_mat == pred_mat.max(axis=1, keepdims=True))
                        score = np.sum(np.mean(train_data * pred_mask, axis=1))

                        if score > best_score:
                            best_score = score
                            best_param_dicts = [param_dict]
                        elif score == best_score:
                            best_param_dicts.append(param_dict)

        # Randomly select ont of the best param dicts
        self.params = best_param_dicts[int(np.random.randint(0, len(best_param_dicts)))]
        self.best_param_dicts = best_param_dicts

    def syllog_to_premises(self, syllog):
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

    def format_response(self, responses):
        template_resp = {
            'Mac': 'Tac',
            'Mca': 'Tca',
            'Ma-c': 'Dac',
            'Mc-a': 'Dca',
        }

        responses_formatted = []
        for resp in responses:
            if resp in template_resp:
                resp = template_resp[resp]
        
            responses_formatted.append(resp)

        return responses_formatted

    def predict(self, item, **kwargs):
        """ Queries mReasoner for a prediction.

        Parameters
        ----------
        item : ccobra.Item
            Task item.

        Returns
        -------
        list(str)
            Syllogistic response prediction.

        """
        # Extract premises
        syllog = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        premises = self.syllog_to_premises(syllog.encoded_task)
        print(item.identifier, item.task_str)

        # Sample predictions from mReasoner
        pred_scores = np.zeros((13,))
        for _ in range(self.n_samples):
            predictions = self.mreasoner.query(premises, self.params)
            predictions = self.format_response(predictions)
            for pred in predictions:
                if pred not in self.RESPONSES_gen:
                    print('Invalid Response:', pred)
                else:
                    resp_idx = self.RESPONSES_gen.index(pred)
                    pred_scores[resp_idx] += 1 / len(predictions)

        # Determine best prediction
        cand_idxs = np.arange(len(pred_scores))[pred_scores == pred_scores.max()]
        pred_idx = np.random.choice(cand_idxs)
        return syllog.decode_response(self.RESPONSES_gen[pred_idx])

    def adapt(self, item, truth, **kwargs):
        """ Adapts mReasoner to the participant responses.

        """

        # Encode syllogistic information
        enc_task = ccobra.syllogistic_generalized.encode_task(item.task)
        enc_resp = ccobra.syllogistic_generalized.encode_response(truth, item.task)

        # Update history
        task_idx = self.SYLLOGISMS_gen.index(enc_task)
        resp_idx = self.RESPONSES_gen.index(enc_resp)
        self.history[task_idx, resp_idx] += 1

        # Perform training
        self.fit()
