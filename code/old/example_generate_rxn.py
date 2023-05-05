
from generate_rxn import generate_reactions


# generate_reactions( reaction_string, index_sizes, reaction_group_name, rate_fn, no_self = False )


# states for three regions
# 0  100  A
# 1  010  B
# 2  001  C
# 3  110  AB
# 4  101  AC
# 5  011  BC
# 6  111  ABC

regions = 'ABC'
num_regions = len(regions)
num_states = 2**num_regions - 1

# GeoSSE dispersal, e.g. A -> B or S[0] -> S[3]; BC -> ABC or S[5] -> S[6]
reaction_string_d      = 'S[i] -> S[j]'
index_sizes_d          = { 'i':num_states, 'j':num_states }
reaction_group_name_d  = 'Dispersal'

# GeoSSE extirpation, e.g. AB -> B or S[3] -> S[1]; ABC -> BC or S[6] -> S[5]
reaction_string_e      = 'S[i] -> S[j]'
index_sizes_e          = { 'i':num_states, 'j':num_states }
reaction_group_name_e  = 'Extirpation'

# GeoSSE extinction, e.g. B -> 0 or S[1] -> X;
reaction_string_x      = 'S[i] -> X'
index_sizes_x          = { 'i':num_states }
reaction_group_name_x  = 'Extinction'

# GeoSSE within-region speciation, e.g. AB -> AB + B or S[3] -> S[1]; ABC -> BC or S[6] -> S[5]
reaction_string_w      = 'S[i] -> S[i] + S[j]'
index_sizes_w          = { 'i':num_states, 'j':num_states }
reaction_group_name_w  = 'Within-region speciation'

# GeoSSE within-region speciation, e.g. AB -> AB + B or S[3] -> S[1]; ABC -> BC or S[6] -> S[5]
reaction_string_b      = 'S[i] -> S[j] + S[k]'
index_sizes_b          = { 'i':num_states, 'j':num_states, 'k':num_states }
reaction_group_name_b  = 'Between-region speciation'


# Define Dispersal rate_fn for generate_rxn
def d_rate_fn(idx, s):
    # idx is an 2-tuple of integers to index states
    # s is a state space object

    # convert integer states into bits (e.g. 2:001)    
    from_bits = s.int2vec[ idx[0] ]
    to_bits   = s.int2vec[ idx[1] ]
    n = len(from_bits)

    # check if state transition is valid dispersal event
    n_diff = 0
    disp_idx = -1
    for i in range(n):
        if from_bits[i] != to_bits[i]:
            n_diff += 1
            disp_idx = i
    
    # dispersal event has exactly 1 diff bit
    if n_diff != 1:
        return 0.0
    
    # disperal event has less from than to bits
    if sum(from_bits) >= sum(to_bits):
        return 0.0
    
    # create new destination state
    new_bits = to_bits
    new_bits[disp_idx] = 1
    new_int = s.vec2int[ tuple(new_bits) ]

    print( idx[0], '->', idx[1], '; gain', disp_idx, ';', new_bits, new_int )

    return 1.0 * sum(from_bits)


d_rate_fn( (1,3), ss )

zz = generate_reactions( reaction_string_d, index_sizes_d, reaction_group_name_d, d_rate_fn, state_space=ss)