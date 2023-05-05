#!/usr/bin/python3
import itertools
import re
import numpy as np
def generate_reactions(reaction_string, index_sizes, reaction_group_name, rate_fn, state_space = None, no_self = False):
    """
    generates all possible combinations of indices, replaces the keys in the reaction_string 
    with the corresponding index values, calculates the reaction rates using the provided rate 
    function, and creates an XML string representing the reaction group.

    Args:
            reaction_template (str): The reaction template string, with placeholders for indices in the format
                                     "[i]", where "i" represents the index position. For example: "A[i] + B[i] -> C[j]".
            index_sizes (dictnary of and keys "i","j" etc.): A list containing the sizes for each of the indices in the 
                                reaction template.
            reaction_group_name (str): The name to be assigned to the reaction group.
        rate_fn (function(idx)): a function that takes in a tuple of all the index values for a given rxn and outputs a rate.

    Returns:
        the generated XML string for the rxn group and a dictionary containing the reaction rates.
    """

    # Generate all possible combinations of indices
    pre_index_combinations = itertools.product(*[range(size) for size in index_sizes.values()])
    index_combinations = [cc for cc in pre_index_combinations]

    # Extract keys from the index_sizes dictionary
    keys = [k for k in index_sizes.keys()]

    # Initialize an empty dictionary to store reaction rates
    rxn_rates = {}        

    # Start creating the XML string for the reaction group
    xml_string = "<reactionGroup spec='ReactionGroup' reactionGroupName='{}'>\n".format(reaction_group_name)

    # Loop through all index combinations
    for combo_idx in range(len(index_combinations)):
        # Initialize the reaction name
        reaction_name = reaction_group_name
        new_rxn = reaction_string

        if not (no_self and len(index_combinations[combo_idx]) != len(set(index_combinations[combo_idx]))):

            # Replace keys with corresponding index values in reaction_string
            for idx in range(len(index_combinations[combo_idx])):
                reaction_name = reaction_name + "_" + str(index_combinations[combo_idx][idx])
                new_rxn = new_rxn.replace('[' + keys[idx] + ']', '[' + str(index_combinations[combo_idx][idx]) + ']')

            # Calculate the rate using the provided rate function
            rate = rate_fn(index_combinations[combo_idx], state_space)

            # only store valid positive-valued rates
            if rate > 0.0:
                
                # Store the reaction rate in the rxn_rates dictionary
                rxn_rates[reaction_name] = rate

                # Append the reaction to the XML string and substitute rxn name and rate
                xml_string += '  <reaction spec="Reaction" reactionName="{reaction_name}" rate="{rate}">\n'.format(reaction_name = reaction_name, rate = rate)
                xml_string += "\t\t" + new_rxn + "\n"
                xml_string += "  </reaction>\n"

    # Close the reaction group XML tag
    xml_string += '</reactionGroup>'

    # Return the generated XML string and reaction rates dictionary
    return xml_string, rxn_rates
