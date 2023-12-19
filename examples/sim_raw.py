#!/usr/bin/env python
# Copyright Friedemann Zenke 2020

import argparse
import json
import numpy as np
import sys
import os
import time
import logging

import wandb
import torch
import stork
import stork.datasets
import stork.utils
from stork.models import RecurrentSpikingModel
from stork.connections import Connection, ConvConnection
from stork.datasets import RawHeidelbergDigits, RawSpeechCommands, DatasetView
from stork.generators import StandardGenerator
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup, AdaptLearnLIFGroup, TsodyksMarkramSTP, MaxPool1d


def get_model(args, device, dtype, logger, wandb):

    neuron_group = LIFGroup
    if args.adaptive:
        neuron_group = AdaptLearnLIFGroup


    model = RecurrentSpikingModel(args.batch_size,
                                  args.nb_time_steps,
                                  args.nb_inputs,
                                  device,
                                  dtype)

    input_shape = (1,args.nb_inputs)
    input_group = model.add_group(stork.nodes.InputWarpGroup(input_shape, args.nb_input_steps))

    # Define regularizer list
    regs = []

    if args.weightL1Strength:
        w_regs = [ stork.regularizers.WeightL1Regularizer(args.weightL1Strength) ] 
    else:
        w_regs = []


    # scl_act_reg = 1.0/(args.nb_hidden_blocks+1.0) # TODO should try this compensate for higher reg loss when adding layers
    scl_act_reg = 1.0/(2*args.nb_hidden_blocks+1.0)
    if args.lowerBoundL2Strength:
        reg1 = stork.regularizers.LowerBoundL2(scl_act_reg*args.lowerBoundL2Strength,
                                                threshold=args.lowerBoundL2Threshold, dims=2 ) # sum over time average over position (new)
        regs.append(reg1)

    if args.upperBoundL1Strength:
        reg = stork.regularizers.UpperBoundL1(scl_act_reg*args.upperBoundL1Strength,
                                                 threshold=args.upperBoundL1Threshold, dims=(1,2) ) # sum over time and average over feature and position
        regs.append(reg)

    if args.upperBoundL2Strength:
        reg2 = stork.regularizers.UpperBoundL2(scl_act_reg*args.upperBoundL2Strength,
                                                 threshold=args.upperBoundL2Threshold, dims=(1,2) )
        regs.append(reg2)


    if args.perNeuronUpperBound is not None:
        reg = stork.regularizers.PerNeuronUpperBoundL2Regularizer(0.1, threshold=args.perNeuronUpperBound)
        regs.append(reg)


    # Collect neuronal params in dictionary
    act = stork.activations.SuperSpike 
    if args.stochastic:
        act = stork.activations.StochasticSuperSpike
    if args.nonmonotonic:
        act = stork.activations.NonmonotonicSuperSpike
    neuronal_params = dict(tau_mem=args.tau_mem*args.tau_scale,
                                       tau_syn=args.tau_syn*args.tau_scale,
                                       diff_reset=args.diffreset,
                                       stateful=args.stateful,
                                       dropout_p=args.dropout, 
                                       regularizers=regs,
                                       activation=act)


    # add cell groups
    upstream_group = input_group
    model.plot_groups = []

    # Implement spike encoder
    fanout_group =  model.add_group(stork.nodes.FanOutGroup(upstream_group, args.encoder_fanout, dim=0)) 
    # upstream_group = analog_digital_conversion_group = model.add_group(stork.nodes.AdaptiveLIFGroup(fanout_group.shape, adapt_a=0.5, tau_ada=100e-3*args.tau_scale, tau_mem=args.tau_mem, tau_syn=args.tau_syn, dropout_p=args.dropout, stateful=args.stateful,)) 
    upstream_group = analog_digital_conversion_group = model.add_group(stork.nodes.AdaptiveLIFGroup(fanout_group.shape, adapt_a=0.5, tau_ada=100e-3*args.tau_scale, **neuronal_params, name="encoder")) 
    id_con = model.add_connection(stork.connections.IdentityConnection(fanout_group, analog_digital_conversion_group, bias=True, tie_weights=[1], weight_scale=1.0))
    scl = args.encoder_gain*args.time_step/(args.tau_syn*args.tau_scale)
    id_con.init_parameters(0.0,scl)
    shp = id_con.weights.shape
    id_con.weights.data = torch.reshape(torch.linspace(-scl, scl, analog_digital_conversion_group.shape[0], requires_grad=True), shp)
    print(id_con.get_weights())
    model.plot_groups.append(upstream_group)
    analog_digital_conversion_group.regularizers.extend(regs)

    input_dim = args.nb_inputs # 40 for the raw datasets
    nb_hidden_channels = args.encoder_fanout
    block_output = [ analog_digital_conversion_group ] 
    for l in range(args.nb_hidden_blocks):
        logger.info("Adding hidden block %i"%l)
        nb_hidden_channels = int(args.channel_fanout*nb_hidden_channels)

        neurons1 = model.add_group(neuron_group((nb_hidden_channels,input_dim), name="Conv%i"%l, **neuronal_params))
        if args.stp: neurons1 = model.add_group(TsodyksMarkramSTP(neurons1))
        con = model.add_connection(ConvConnection(upstream_group, neurons1, regularizers=w_regs, bias=args.bias, kernel_size=5, stride=1, padding=2 ))  
        con.init_parameters(args.wmean,args.wscale*args.basescale)

        if args.recurrent:
            con = model.add_connection(ConvConnection(neurons1, neurons1, regularizers=w_regs, bias=args.bias, kernel_size=5, stride=1, padding=2))
            con.init_parameters(args.wmean,args.wscale*args.basescale*args.recscale)

        neurons2 = model.add_group(neuron_group((nb_hidden_channels,input_dim), name="Conv%i"%l, **neuronal_params))
        if args.stp: neurons2 = model.add_group(TsodyksMarkramSTP(neurons2))
        con = model.add_connection(ConvConnection(neurons1, neurons2, regularizers=w_regs, bias=args.bias, kernel_size=5, stride=1, padding=2 ))  
        con.init_parameters(args.wmean,args.wscale*args.basescale)

        if args.recurrent:
            con = model.add_connection(ConvConnection(neurons2, neurons2, regularizers=w_regs, bias=args.bias, kernel_size=5, stride=1, padding=2))
            con.init_parameters(args.wmean,args.wscale*args.basescale*args.recscale)

        model.plot_groups.extend([neurons1,neurons2])
        input_dim = input_dim//2
        upstream_group=model.add_group(MaxPool1d(neurons2))
        block_output.append(neurons2)


    # Readout layer 
    readout_group = model.add_group(ReadoutGroup(args.nb_classes,
                                                 tau_mem=args.tau_readout*args.tau_scale,
                                                 tau_syn=args.tau_syn*args.tau_scale,
                                                 stateful=args.stateful,
                                                 initial_state=-1e-3))
    # con.init_parameters(0.0,args.wscale*args.readoutscale)
    readout_connection_scale = 1e-3
    wandb.config.update({"readout_connection_scale" : readout_connection_scale})
    # Add readout connections from each block
    if args.skip:
        for blk in block_output:
            con = model.add_connection(Connection(blk, readout_group, flatten_input=True))
            con.init_parameters(0.0,readout_connection_scale)
    else:
        con = model.add_connection(Connection(upstream_group, readout_group, flatten_input=True))
        con.init_parameters(0.0,readout_connection_scale)



    generator = StandardGenerator(nb_workers=args.nb_workers, persistent_workers=False)

    if args.loss_type=="MaxOverTime":
        loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()
    elif args.loss_type=="MaxOverTimeBinary":
        loss_stack = stork.loss_stacks.MaxOverTimeBinaryCrossEntropy()
    elif args.loss_type=="LastStep":
        loss_stack = stork.loss_stacks.LastStepCrossEntropy()
    elif args.loss_type=="SumOverTime":
        loss_stack = stork.loss_stacks.SumOverTimeCrossEntropy()
    else:
        logger.warning("Unknown loss type, defaulting to MaxOverTimeCrossEntropy")
        loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()

    # Select optimizer
    optstr = args.optimizer.upper()
    if optstr=="ADAM": 
        opt=torch.optim.Adam
    elif optstr=="SMORMS4":
        opt=stork.optimizers.SMORMS4
    elif optstr=="SMORMS5":
        opt=stork.optimizers.SMORMS5
    elif optstr=="ADAMAX":
        opt=torch.optim.Adamax
    else:
        opt=torch.optim.Adam
        logger.warning("Unknown optimizer defaulting to Adam")


    model.configure(input=input_group,
                    output=readout_group,
                    loss_stack=loss_stack,
                    generator=generator,
                    optimizer=opt,
                    optimizer_kwargs=dict(lr=args.lr),
                    time_step=args.time_step,
                    wandb=wandb)

    # Load network parameters from statefile if given
    if args.load:
        logger.debug("Loading network state from file")
        model.load_state_dict(torch.load(args.load))

    return model


def speaker_performance(y, y_pred, ids):
    num_ids = np.unique(ids).size
    speaker_acc = np.empty(num_ids)
    for i in range(num_ids):
        idx = (ids==i)
        speaker_acc[i] = (y[idx]==y_pred[idx]).mean()
    return speaker_acc


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("netraw")
    
    parser = argparse.ArgumentParser("Perform single classification experiment")
    parser.add_argument("--nb_inputs", type=int, default=40,
                        help="Number of input nodes")
    parser.add_argument("--dataset", type=str,
                        help="Name of dataset", default="hd")
    parser.add_argument("--standardize", action='store_true', default=False,
                        help="Standardize inputs")
    parser.add_argument("--stochastic", action='store_true', default=False,
                        help="Use stochastic neuron")
    parser.add_argument("--nonmonotonic", action='store_true', default=False,
                        help="Use nonmonotonic neuron")
    parser.add_argument("--optimizer", type=str,
                        help="Name of optimizer", default="SMORMS4")
    parser.add_argument("--training_data", type=str,
                        help="Path to training data", default=None)
    parser.add_argument("--validation_split", type=float,
                        help="Validation split ratio", default=0.0)
    parser.add_argument("--nb_input_steps", type=int, default=80,
                        help="Number of input time steps")
    parser.add_argument("--encoder_fanout", type=int, default=5,
                        help="Encoder fan out")
    parser.add_argument("--encoder_gain", type=float, default=0.5,
                        help="Encoder gain")
    parser.add_argument("--testing_data", type=str,
                        help="Path to testing data", default=None)
    parser.add_argument("--nb_classes", type=int, default=20,
                        help="Number of output classes")
    parser.add_argument("--channel_fanout", type=float, default=2,
                        help="Number of nodes in hidden layer")
    parser.add_argument("--nb_hidden_blocks", type=int, default=3,
                        help="Number of hidden layers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for network initialization")
    parser.add_argument("--recurrent", action='store_true', default=False,
                        help="Allow recurrent synapses")
    parser.add_argument("--skip", action='store_true', default=False,
                        help="Enable skip connections")
    parser.add_argument("--adaptive", action='store_true', default=False,
                        help="Use adaptive neuron model")
    parser.add_argument("--stateful", action='store_true', default=False,
                        help="Retain neuronal state across mini batches")
    parser.add_argument("--detach", action='store_true', default=False,
                        help="Stop gradients from flowing through recurrent connections")
    parser.add_argument("--tau_readout", type=float, default=20e-3,
                        help="Membrane time constant of readout neurons")
    parser.add_argument("--beta", type=float, default=100.0,
                        help="Steepness parameter of the surrogate gradient")
    parser.add_argument("--tau_mem", type=float, default=20e-3,
                        help="Membrane time constant of LIF neurons")
    parser.add_argument("--tau_syn", type=float, default=10e-3,
                        help="Synaptic time constant of LIF neurons")
    parser.add_argument("--tau_scale", type=float, default=1.0,
                        help="Scale neuronal time constants by this factor")
    parser.add_argument("--time_step", type=float, default=2e-3,
                        help="Integration time step in ms")
    parser.add_argument("--time_scale", type=float, default=1.0,
                        help="Rescale generator time by this factor ")
    parser.add_argument("--unit_scale", type=float, default=1.0,
                        help="Rescale units by this factor ")
    parser.add_argument("--duration", type=float, default=0.8,
                        help="Maximum duration to consider")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of digits in a single batch")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--nb_priming_epochs", type=int, default=0,
                        help="Number of priming epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=0,
                        help="Number of epochs to fine tune")
    parser.add_argument("--fine_step", type=float, default=1.0e-3,
                        help="Time step used for fine tuning")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--wscale", type=float, default=1.0,
                        help="Weight scale")
    parser.add_argument("--recscale", type=float, default=0.05,
                        help="Additional scaling factor for recurrent weights applied on top of wscale")
    parser.add_argument("--wmean", type=float, default=0.0,
                        help="Mean weight")
    parser.add_argument("--stp", action='store_true',
                        help="Use STP at FF exc synapses")
    parser.add_argument("--permute_units", action='store_true',
                        help="Permute units in dataset")
    parser.add_argument("--weightL1Strength", type=float, default=0.0,
                        help="Weight L1 regulizer strength")
    parser.add_argument("--upperBoundL1Strength", type=float, default=0.0,
                        help="Upper bound L1 regulizer strength")
    parser.add_argument("--upperBoundL1Threshold", type=float, default=5.0,
                        help="Upper bound L1 regulizer threshold")
    parser.add_argument("--upperBoundL2Strength", type=float, default=0.0,
                        help="Upper bound L2 regulizer strength")
    parser.add_argument("--upperBoundL2Threshold", type=float, default=5.0,
                        help="Upper bound L2 regulizer threshold")
    parser.add_argument("--lowerBoundL2Threshold", type=float, default=1e-3,
                        help="Lower bound L2 regulizer threshold")
    parser.add_argument("--lowerBoundL2Strength", type=float, default=100.0,
                        help="Lower bound L2 regulizer strength")
    parser.add_argument("--perNeuronUpperBound", type=float, default=None,
                        help="Per neuron upper bound on spike count")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Probability of dropping a hidden layer spike")
    parser.add_argument("--p_drop", type=float, default=0.0,
                        help="Probability of dropping an input spike")
    parser.add_argument("--p_insert", type=float, default=0.0,
                        help="Probability of inserting an input spike")
    parser.add_argument("--sigma_t", type=float, default=0.0,
                        help="Time jitter amplitude added to spikes in bins")
    parser.add_argument("--sigma_u", type=int, default=0.0,
                        help="Unit jitter amplitude added to spikes")
    parser.add_argument("--sigma_u_uniform", type=int, default=0.0,
                        help="Unit jitter amplitude added to spikes which is coherent across channels")
    parser.add_argument("--coalesced", action='store_true',
                        help="Coalesce input spikes")
    parser.add_argument("--gpu", action='store_true',
                        help="Run simulation on gpu")
    parser.add_argument("--bias", action='store_true',
                        help="Use bias term in forward connections except readout")
    parser.add_argument("--nb_threads", type=int, default=None,
                        help="Number of threads")
    parser.add_argument("--nb_workers", type=int, default=1,
                        help="Number of independent dataloader worker threads")
    parser.add_argument("--diffreset", action='store_true', default=False,
                        help="Enable differentiating reset term")
    parser.add_argument("--loss_type", type=str, default="MaxOverTime",
                        help="Select loss type")
    parser.add_argument("--wandbentity", type=str, default="fzenke",
                        help="Entity name for wandb")
    parser.add_argument("--wandbproject", type=str, default="default",
                        help="Project name for wandb")
    parser.add_argument("--prefix", type=str, default="raw",
                        help="Output file prefix")
    parser.add_argument("--timestr", type=str, default=None,
                        help="Timestr for filename")
    parser.add_argument("--gitdescribe", type=str, default="default",
                        help="Git describe string for logging")
    parser.add_argument("--dir", type=str, default="out",
                        help="Path to where to write results as json")
    parser.add_argument("--cachedir", type=str, default="/tungstenfs/scratch/gzenke/datasets/cache/",
                        help="Path to cache directory")
    parser.add_argument("--save", action='store_true',
                        help="Save results as json")
    parser.add_argument("--plot", action='store_true', default=False, 
                        help="If set, some plots are saved.")
    parser.add_argument("--load", type=str, default=None,
                        help="Filepath to statefile to load network state")
    parser.add_argument("--speaker_acc", action='store_true',
                        help="Compute per-speaker accuracy on test set")
    parser.add_argument("--verbose", action='store_true', default=False)

    args = parser.parse_args()



    sim_start_time = time.time()
    # store datetime string
    if args.timestr is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        basepath = stork.utils.get_basepath(args.dir, args.prefix)
    else:
        timestr = args.timestr
        basepath = "%s/%s-%s" % (args.dir, args.prefix, timestr)

    # store datetime string
    args.basepath = basepath
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))


    # create error file handler and set level to error
    log_filename = "%s.log"%basepath
    file_handler = logging.FileHandler(log_filename, "w", encoding=None, delay="true")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        logger.debug("Set loglevel: DEBUG")
    else:
        logger.setLevel(logging.INFO)

    logger.info("Basepath {}".format(basepath))


    if args.plot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt


    dtype = torch.float
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available(): 
            logger.warning("Cuda is available, but not used.")



    if args.nb_threads is not None:
        logger.debug("PyTorch set to use %i threads." % args.nb_threads)
        torch.set_num_threads(args.nb_threads)

    logger.info("Building model")
    args.nb_time_steps = int(args.duration/args.time_step)
    logger.debug("Simulation with %i time steps"%args.nb_time_steps)
    nb_input_steps = args.nb_input_steps 
    logger.debug("Simulation with %i input time steps"%nb_input_steps)


    # Compute basescale for weights
    nubar = args.nubar = 5.0 # assumed mean input firing rate
    xi = args.xi = 3 # sigma target at threshold
    kernel = stork.utils.get_lif_kernel(args.tau_mem*args.tau_scale, args.tau_syn, args.time_step)
    epsilon_hat = (kernel**2).sum()*args.time_step
    logger.debug("epsilon_hat=%e"%epsilon_hat)
    args.basescale = 1.0/np.sqrt(epsilon_hat*nubar)/xi
    args.std_weight = args.wscale*args.basescale # store value used for init
    logger.debug("Setting basescale for weights to %e"%args.basescale)

    kernel = stork.utils.get_lif_kernel(args.tau_readout, args.tau_syn, args.time_step)
    epsilon_hat = (kernel**2).sum()*args.time_step
    args.readoutscale = 1.0/np.sqrt(epsilon_hat*nubar)


    if args.seed is not None:
        logger.debug("Seeding numpy with %i"%args.seed)
        np.random.seed(args.seed)
        tseed = np.random.randint(int(1e9))
        logger.debug("Seeding torch with %i"%tseed)
        torch.random.manual_seed(tseed)

    wandb.init(project=args.wandbproject, entity=args.wandbentity, config=vars(args))
    model = get_model(args, device, dtype, logger, wandb)

    if args.load:
        logger.debug("Loading network state from file")
        model.load_state_dict(torch.load(args.load))

    permutation = None
    if args.permute_units:
        permutation = np.arange(int(args.nb_inputs/args.unit_scale))
        np.random.shuffle(permutation)
    
    gen_kwargs = dict( nb_steps=nb_input_steps,
                       time_step=10e-3,
                       standardize=args.standardize,
                      )


    if args.training_data:
        logger.info("Load data")
        if args.dataset=="hd":

            def read_filelist(filename):
                with open(filename) as f:
                    content = f.readlines()
                # Remove whitespace characters like `\n` at the end of each line
                content = [x.strip() for x in content]
                return content

            dirpath=args.training_data
            train_dataset = RawHeidelbergDigits(dirpath, subset=read_filelist("%s/train_filenames.txt"%dirpath), cache_fname="%s/rawhd-std%i-cache-training.pkl.gz"%(args.cachedir, args.standardize), **gen_kwargs)
            logger.debug("Opened HD dataset with %i data"%len(train_dataset))
            test_dataset = RawHeidelbergDigits(dirpath, subset=read_filelist("%s/test_filenames.txt"%dirpath), cache_fname="%s/rawhd-std%i-cache-test.pkl.gz"%(args.cachedir, args.standardize), **gen_kwargs)
            logger.debug("Opened HD dataset with %i data"%len(test_dataset))

            if args.validation_split:
                logger.info("Splitting off validation data")
                mother_dataset=train_dataset
                elements = np.arange(len(mother_dataset))
                np.random.shuffle(elements)
                split = int(args.validation_split*len(mother_dataset))
                train_dataset = DatasetView(mother_dataset, elements[:split])
                valid_dataset = DatasetView(mother_dataset, elements[split:])

        elif args.dataset=="sc":
            args.validation_split = 0.9 # Since this is fixed for this dataset
            logger.debug("Loading training data")
            train_dataset = RawSpeechCommands(args.training_data, subset="training", cache_fname="%s/rawsc-cache-training.pkl.gz"%(args.cachedir), **gen_kwargs)
            logger.debug("Loading validation data")
            valid_dataset = RawSpeechCommands(args.training_data, subset="validation", cache_fname="%s/rawsc-cache-validation.pkl.gz"%(args.cachedir), **gen_kwargs)
            logger.debug("Loading test data")
            test_dataset = RawSpeechCommands(args.training_data, subset="testing", cache_fname="%s/rawsc-cache-testing.pkl.gz"%(args.cachedir), **gen_kwargs)

        logger.info("Loaded %i training data"%len(train_dataset))
        if args.validation_split: logger.info("Loaded %i validation data"%len(valid_dataset))
        logger.info("Loaded %i test data"%len(test_dataset))


        # model.summary()

        if args.plot:
            filepath = "%s-init.png"%(basepath)
            data, labels = model.get_example_batch(test_dataset,shuffle=False)
            model.predict(data)
            fig = plt.figure(figsize=(7,10), dpi=150)
            stork.plotting.plot_activity_snapshot(model,
                                                  data=data, labels=labels,
                                                  nb_samples=4,
                                                  point_alpha=0.3,
                                                  plot_groups=model.plot_groups,
                                                  input_heatmap=True)
            plt.title(os.path.basename(filepath), fontsize=8)
            plt.savefig(filepath,dpi=150)
            wandb.log({"snapshot-init": wandb.Image(filepath)})

        if args.nb_priming_epochs: 
            logger.info("Priming model on train data")
            history = model.prime(train_dataset,
                                     nb_epochs=args.nb_priming_epochs,
                                     verbose=args.verbose)

        if args.nb_epochs: 
            logger.info("Fitting model to train data")
            if args.validation_split:
                history = model.fit_validate(train_dataset, valid_dataset,
                                             nb_epochs=args.nb_epochs,
                                             verbose=args.verbose)

                # Finegrain time
                if args.fine_tune_epochs:
                    logger.info("Fine tuning model with %.2e time step for %i"%(args.fine_step, args.fine_tune_epochs))
                    logger.debug("before rescaling: nb_time_steps=%i"%model.nb_time_steps)
                    model.time_rescale(args.fine_step, batch_size=100)
                    logger.debug("rescaled: nb_time_steps=%i"%model.nb_time_steps)
                    id_con = model.connections[0]
                    id_con.weight_scale = args.fine_step/args.time_step
                    history = model.fit_validate(train_dataset, valid_dataset,
                                                 nb_epochs=args.fine_tune_epochs,
                                                 verbose=args.verbose)
            else:
                history = model.fit(train_dataset,
                                             nb_epochs=args.nb_epochs,
                                             verbose=args.verbose)

            results["train_loss"] = history["loss"].tolist()
            results["train_acc"]  = history["acc"].tolist()
            logger.info("Train loss {:.3f}".format(history["loss"][-1]))
            logger.info("Train acc {:.3f}".format(history["acc"][-1]))

            if args.validation_split:
                results["valid_loss"] = history["val_loss"].tolist()
                results["valid_acc"]  = history["val_acc"].tolist()
                logger.info("Valid loss {:.3f}".format(history["val_loss"][-1]))
                logger.info("Valid acc {:.3f}".format(history["val_acc"][-1]))

    #logger.info("Load testing data")
    #test_dataset = HDF5Dataset(args.testing_data, **gen_kwargs)
    #logger.info("Loaded %i test data"%len(test_dataset))

    # test_dataset.p_drop = 0.0
    # test_dataset.p_insert = 0.0
    
    logger.info("Evaluation model performance on test data")
    scores = model.evaluate(test_dataset).tolist()
    results["test_loss"] = scores[0]
    results["test_acc"] = scores[2]
    logger.info("Test loss {:.3f}".format(scores[0]))
    logger.info("Test acc {:.3f}".format(scores[2])) 
    wandb.run.summary["test_loss"] = results["test_loss"]
    wandb.run.summary["test_acc"]  = results["test_acc"]

    if args.speaker_acc:
        logger.info("Estimating per-speaker accuracy...")
        speaker_ids = test_dataset.h5file['extra']['speaker']
        y_pred = model.get_predictions(test_dataset)
        results["speaker_performance"] = speaker_performance(test_dataset.labels.cpu().numpy(),
                                                       y_pred, np.array(speaker_ids)).tolist()

    # basepath = "%s/%s-%s"%(args.dir, timestr, args.prefix)
    
    
    # Store wall clock time duration
    sim_end_time = time.time()
    results["wall_time"] = sim_end_time - sim_start_time

    logger.debug("Saving results")
    filepath = "%s.json"%(basepath)
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)


    if args.plot:
        filepath = "%s-final.png"%(basepath)
        data, labels = model.get_example_batch(test_dataset,shuffle=False)
        model.predict(data)
        fig = plt.figure(figsize=(7,10), dpi=150)
        stork.plotting.plot_activity_snapshot(model,
                                              data=data, labels=labels,
                                              nb_samples=4,
                                              point_alpha=0.3,
                                              plot_groups=model.plot_groups,
                                              input_heatmap=True)
        plt.title(os.path.basename(filepath), fontsize=8)
        plt.savefig(filepath,dpi=150)
        wandb.log({"snapshot-final": wandb.Image(filepath)})

    if args.save:
        logger.debug("Saving model")
        filepath = "%s.state"%(basepath)
        torch.save(model.state_dict(), filepath)
