#! python

########################
# LOAD PIPELINE CONFIG #
########################

from phyddle import Utilities
print(Utilities.make_pipeline_header('title'))
my_args = Utilities.load_config('config', arg_overwrite=True)
step = my_args['step']

################
# RUN PIPELINE #
################


# Step 1: simulate training data
if 'sim' in step:
    #print('### Simulating ###' % (format, format))
    print(Utilities.make_pipeline_header('sim'))
    from phyddle import ModelLoader
    from phyddle import Simulating
    my_mdl = ModelLoader.load(my_args)
    my_sim = Simulating.load(my_args, my_mdl)
    my_sim.run()

# Step 2: format training data into tensors
if 'fmt' in step:
    print(Utilities.make_pipeline_header('fmt'))
    from phyddle import Formatting
    my_fmt = Formatting.load(my_args)
    my_fmt.run()

# Step 3: train network with training data
if 'lrn' in step:
    print(Utilities.make_pipeline_header('lrn'))
    from phyddle import Learning
    my_lrn = Learning.load(my_args)
    my_lrn.run()

# Step 4: predict for new dataset
if 'prd' in step:
    print(Utilities.make_pipeline_header('prd'))
    from phyddle import Formatting
    my_fmt = Formatting.load(my_args)
    pred_prefix = f"{my_args['pred_dir']}/{my_args['proj']}/{my_args['pred_prefix']}"
    my_fmt.encode_one(tmp_fn=pred_prefix, idx=-1, save_phyenc_csv=True)

    from phyddle import Predicting
    my_prd = Predicting.load(my_args)
    my_prd.run()

# Step 5: plot results
if 'plt' in step:
    print(Utilities.make_pipeline_header('plt'))
    from phyddle import Plotting
    my_plt = Plotting.load(my_args)
    my_plt.run()

