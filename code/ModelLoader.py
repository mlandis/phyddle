#import GeosseModel
#import SirmModel

# class ModelLoader:
#     def __init__(self, args):
#         self.args = args
#         self.load_model(args)
#         return
#     # register models for retrieval

def load_model(args):
    model_type = args['model_type']
    #show_model_variants = args['show_model_variants']
    print('Loading', model_type)
    if model_type == 'geosse':
        import GeosseModel
        mdl = GeosseModel.GeosseModel
    elif model_type == 'sirm':
        import SirmModel
        mdl = SirmModel.SirmModel
    elif model_type == 'cont_trait':
        import ContTraitModel
        mdl = ContTraitModel.ContTraitModel
    else:
        return None
    return mdl(args)
