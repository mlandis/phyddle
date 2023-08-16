#!/usr/bin/env python3

def run():

    ########################
    # LOAD PIPELINE CONFIG #
    ########################

    import phyddle.utilities as util
    my_args = util.load_config('config', arg_overwrite=True)
    step = my_args['step']

    logger = util.Logger(my_args)
    logger.save_log('run')

    ################
    # RUN PIPELINE #
    ################

    # Step 1: simulate training data
    if 'S' in step:
        import phyddle.simulate as sim
        my_sim = sim.load(my_args)
        my_sim.run()

    # Step 2: format training data into tensors
    if 'F' in step:
        import phyddle.format as fmt
        my_fmt = fmt.load(my_args)
        my_fmt.run()

    # Step 3: train network with training data
    if 'T' in step:
        import phyddle.train as trn
        my_trn = trn.load(my_args)
        my_trn.run()

    # Step 4: estimates for new dataset
    if 'E' in step:
        import phyddle.format as fmt
        import phyddle.estimate as est
        est_prefix = f"{my_args['est_dir']}/{my_args['proj']}/{my_args['est_prefix']}"
        my_fmt = fmt.load(my_args)
        my_fmt.encode_one(tmp_fn=est_prefix, idx=-1, save_phyenc_csv=True)
        my_est = est.load(my_args)
        my_est.run()

    # Step 5: plot results
    if 'P' in step:
        import phyddle.plot as plt
        my_plt = plt.load(my_args)
        my_plt.run()

