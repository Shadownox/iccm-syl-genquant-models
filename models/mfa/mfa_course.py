""" Implementation of the most-frequent answer (MFA) model which predicts responses based on the
most-frequently selected choice from the available background (training) data.

"""

import collections

import ccobra
import numpy as np


class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(MFAModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

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

    def pre_train(self, dataset, **kwargs):
        """ Determine most-frequent answers from the training data.

        """
        
        self.response_mat = np.zeros((144, 13))
        for subject_data in dataset:
            # Iterate over tasks
            for task_data in subject_data:
                # Encode the task
                syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resp = syl.encode_response(task_data['response'])
                
                self.response_mat[self.SYLLOGISMS_gen.index(enc_task), self.RESPONSES_gen.index(enc_resp)] += 1    
        

        self.response_mat = self.response_mat == self.response_mat.max(axis=1, keepdims=True)
        self.RESPONSES_gen = np.array(self.RESPONSES_gen)

    def pre_train_person(self, dataset, **kwargs):
        """ The MFA model is not supposed to be person-trained.

        """

        pass

    def predict(self, item, **kwargs):
        """ Generate prediction based on the most-frequent answer.

        """
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        enc_task = syl.encoded_task
        
        response_mask = self.response_mat[self.SYLLOGISMS_gen.index(enc_task)]
        predictions = self.RESPONSES_gen[response_mask]
        pred = np.random.choice(predictions)
        return syl.decode_response(pred)


    def adapt(self, item, truth, **kwargs):
        pass
