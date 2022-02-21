import numpy as np
from scipy.stats import halfnorm
import plotly.express as px
import pandas as pd


rng = np.random.default_rng()

recovery_rate_vector = rng.normal(0, 2, 133)

def sparse_distribution(num_entries):
    dist = rng.binomial(6, 0.3, num_entries)
    return np.clip(dist, a_min=0, a_max=None)

def generate_sparse_matrix(size, density, distribution):

    all_indices = set(range(size * size))
    diagonal_indices = set([i*size + i for i in range(size)])
    indices_except_diagonal = all_indices.difference(diagonal_indices)

    num_of_non_zero_entries = int(density * (size * size))
    sample_from_dist = distribution(num_of_non_zero_entries)

    non_zero_entry_dist_uniform_part = rng.uniform(0, 1, size=len(indices_except_diagonal))
    non_zero_entry_dist_normal_part = rng.normal(0, 5, size=len(indices_except_diagonal)).clip(0, None)
    rng.shuffle(non_zero_entry_dist_normal_part)
    combined_non_zero_distribution = non_zero_entry_dist_uniform_part + non_zero_entry_dist_normal_part

    combined_non_zero_distribution = non_zero_entry_distribution / sum(non_zero_entry_distribution)
    non_zero_entry_distribution = sorted(list(non_zero_entry_distribution), reverse=True)

    non_zero_indices = rng.choice(list(indices_except_diagonal), num_of_non_zero_entries,
                                  p=non_zero_entry_distribution)

    matrix = np.zeros(size*size)
    matrix[non_zero_indices] = sample_from_dist

    return matrix.reshape(size, size)


flow_matrix = generate_sparse_matrix(133, 0.1, sparse_distribution)

def convert2infection_matrix(flow_matrix):
    with np.errstate(divide='ignore'):
        row_sums = flow_matrix.sum(axis=1, keepdims=True)
        infection_matrix = flow_matrix / row_sums

        #Assign zero to rows entries that are divided by zero.
        infection_matrix[(row_sums == 0).reshape(infection_matrix.shape[0])] = 0

    return infection_matrix


sample_inf_matrix = convert2infection_matrix(flow_matrix)

infection_rates = pd.read_pickle('../data/all_time_infection_rate_df.pickle')

random_idx = rng.choice(infection_rates.index, 1)

date, tw, _ = random_idx.item()

fig = px.imshow(infection_rates.loc[(date, tw)], title='real_'+date+'_'+str(tw))
fig2 = px.imshow(sample_inf_matrix, title='generated')

print(np.count_nonzero(infection_rates.loc[(date, tw)]))
print(np.count_nonzero(sample_inf_matrix))
fig.show()
fig2.show()
