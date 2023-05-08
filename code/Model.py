#!/usr/local/bin/python3

# import models
from Models.GeosseModel import GeosseModel
from Models.SirmModel import SirmModel

# register models for retrieval
def make_model(mdl_args):
    model_type = mdl_args['model_type']
    if   model_type == 'geosse':      mdl = GeosseModel
    elif model_type == 'sirm':        mdl = SirmModel
    #elif model_type == 'nichesse':    return NicheModel
    #elif model_type == 'birthdeath':  return BirthDeathModel
    #elif model_type == 'chromosse':   return ChromosseModel
    #elif model_type == 'classe':      return ClasseModel
    #elif model_type == 'musse':       return MusseModel
    else:                             return None
    return mdl(**mdl_args)


#
# TODO: add Model base class
#
# class Model:
#     def __init__(self):
#         return
#     def set_model(self):
#         raise NotImplementedError
#     def clear_model(self):
#         self.is_model_set = False
#         self.states = None
#         self.rates = None
#         self.events = None
#         self.df_events = None
#         self.df_states = None
#     def make_settings(self):
#         raise NotImplementedError
#     def make_states(self):
#          raise NotImplementedError
#     def make_events(self):
#         raise NotImplementedError
#     def make_rates(self):
#         raise NotImplementedError
#     def make_start_state(self):
#         raise NotImplementedError
#     def make_start_sizes(self):
#         raise NotImplementedError
