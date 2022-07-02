import collections

import ccobra
import numpy as np


class UBCF(ccobra.CCobraModel):
    def __init__(self, name='UBCF', k=10, exp=2):
        super(UBCF, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Initialize member variables
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
        
        self.k = k
        self.exp = exp

    def pre_train(self, dataset, **kwargs):
        """ Determine most-frequent answers from the training data.

        """
        
        self.user_vectors = []

        for subject_data in dataset:
            subj_vector = np.zeros((144))
            # Iterate over tasks
            for task_data in subject_data:
                # Encode the task
                syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resp = syl.encode_response(task_data['response'])
                
                subj_vector[self.SYLLOGISMS_gen.index(enc_task)] = self.RESPONSES_gen.index(enc_resp)
            self.user_vectors.append(subj_vector)
        
    def pre_train_person(self, dataset, **kwargs):
        """ Stores all responses (but the current task will be omitted for evaluation)

        """
        profile = np.zeros((144))
        for task_data in dataset:
            syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(task_data['item'])
            enc_task = syl.encoded_task
            enc_resp = syl.encode_response(task_data['response'])
            
            profile[self.SYLLOGISMS_gen.index(enc_task)] = self.RESPONSES_gen.index(enc_resp)

        self.profile = profile

    def get_neighbors(self):
        sims = []
        for user in self.user_vectors:
            sim = (np.sum(user == self.profile) / len(user))**self.exp
            sims.append((sim, user))
        
        return sorted(sims, key=lambda x: x[0], reverse=True)[:self.k]

    def predict(self, item, **kwargs):
        """ Generate prediction based on the most-frequent answer.

        """
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        enc_task = syl.encoded_task
        task_idx = self.SYLLOGISMS_gen.index(enc_task)
        
        # temporarily disable the current task
        cur_task = self.profile[task_idx]
        self.profile[task_idx] = -1
        
        neighbors = self.get_neighbors()
        
        # restore profile
        self.profile[task_idx] = cur_task
        
        # Calculate prediction
        prediction_vec = np.zeros((13))
        for neighbor in neighbors:
            sim, vec = neighbor
            user_resp = int(vec[task_idx])

            prediction_vec[user_resp] += sim
        
        response_mask = prediction_vec == prediction_vec.max()
        predictions = np.array(self.RESPONSES_gen)[response_mask]
        pred = np.random.choice(predictions)
        return syl.decode_response(pred)


    def adapt(self, item, truth, **kwargs):
        pass
