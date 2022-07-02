import numpy as np

partials = 10

resulting_mat = None

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


for i in range(1, partials + 1):
    partial = np.load("partial/cache_gen_{}.npy".format(i))
    if resulting_mat is None:
        resulting_mat = np.zeros(partial.shape)
    
    resulting_mat += partial

for a in range(resulting_mat.shape[0]):
    for b in range(resulting_mat.shape[1]):
        for c in range(resulting_mat.shape[2]):
            for d in range(resulting_mat.shape[3]):
                res = resulting_mat[a, b, c, d]
                if res.sum() != 1440:
                    print("Not all values for: ", a, b, c, d)

np.save("genQuant_cache_6its.npy", resulting_mat)