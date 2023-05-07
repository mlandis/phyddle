#!/usr/local/bin/python3

# import models
from geosse_model import GeosseModel
from sirm_model import SirmModel

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