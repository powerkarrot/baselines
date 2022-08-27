import time
import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.common import explained_variance, set_global_seeds
from baselines.common.models import get_network_builder
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner

def constfn(val):
    def f(_):
        return val
    return f

def learn(
        *, network, env, total_timesteps, eval_env=None, seed=None,
        nsteps=2048, ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5,
        gamma=0.99, lam=0.95, log_interval=10, nminibatches=4, noptepochs=4,
        cliprange=0.2, save_interval=0, load_path=None, model_fn=None,
        tb_logger=None, evaluator=None, model_fname=None, after_epoch_cb=None,
        schedule_gamma=False, schedule_gamma_after=4000,
        schedule_gamma_value=0.999, n_models=3, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        network = policy_network_fn(ob_space.shape)

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    models = []
    runners = []
    for i in range(n_models):
        model = model_fn(
            ac_space=ac_space, policy_network=network, ent_coef=ent_coef,
            vf_coef=vf_coef, max_grad_norm=max_grad_norm
        )

        if load_path is not None:
            # new_model = model.load(load_path)
            # new_model.step = model.step
            # new_model.value = model.value
            # new_model.train = model.train
            # new_model.save = model.save
            # new_model.load = model.load

            #model = new_model
            model.load_chk(load_path, model)

        models.append(model)
        j = 'm' + str(i + 1)
        env.models[j] = model

        # Instantiate the runner object
        runner = Runner(
            env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
            tb_logger=tb_logger, after_epoch_cb=after_epoch_cb,
            schedule_gamma=schedule_gamma,
            schedule_gamma_after=schedule_gamma_after,
            schedule_gamma_value=schedule_gamma_value, model_id=j
        )
        runners.append(runner)

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch.
        obs_ = []
        returns_ = []
        masks_ = []
        actions_ = []
        values_ = []
        neglogpacs_ = []
        states_ = []
        mblossvals_ = []
        lossvals_ = []
        for i in range(n_models):
            obs, returns, masks, actions, values, neglogpacs, states, _ =\
                    runners[i].run(update)
            obs_.append(obs)
            returns_.append(returns)
            masks_.append(masks)
            actions_.append(actions)
            values_.append(values)
            neglogpacs_.append(neglogpacs)
            states_.append(states)

            mblossvals = get_mblossvals(
                nbatch, noptepochs, nbatch_train, lrnow, cliprangenow,
                models[i], obs, returns, masks, actions, values, neglogpacs,
                states
            )
            # For each minibatch calculate the loss and append it.
            mblossvals_.append(mblossvals)

            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            lossvals_.append(lossvals)

        # End timer
        tnow = time.perf_counter()

        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            for i in range(n_models):
                j = str(i + 1)
                log(
                    'm' + j, logger, tb_logger, values_[i], returns_[i], update, nsteps,
                    nbatch, fps, lossvals_[i], models[i], tnow, tfirststart
                )

                if update % 1000 == 0 or update == 1:
                    fname = model_fname + '-' + str(update // 1000) + '-m' + j
                    models[i].save(fname)

                if update % 1000 == 0 and evaluator:
                    evaluator(models[i], update)

    return models


def log(
        model_identifier, logger, tb_logger, values, returns, update, nsteps,
        nbatch, fps, lossvals, model, tnow, tfirststart):
    # Calculates if value function is a good predicator of the returns (ev > 1)
    # or if it's just worse than predicting nothing (ev =< 0)
    ev = explained_variance(values, returns)

    misc_key = 'misc_' + model_identifier
    fps_key = 'fps_' + model_identifier
    loss_key = 'loss_' + model_identifier

    logger.logkv(misc_key + "/serial_timesteps", update*nsteps)
    logger.logkv(misc_key + "/nupdates", update)
    logger.logkv(misc_key + "/total_timesteps", update*nbatch)
    logger.logkv(fps_key, fps)
    logger.logkv(misc_key + "/explained_variance", float(ev))
    logger.logkv(misc_key + "/time_elapsed", tnow - tfirststart)
    for (lossval, lossname) in zip(lossvals, model.loss_names):
        logger.logkv(loss_key + '/' + lossname, lossval)

    logger.dumpkvs()

    misc_key = 'ppo_misc_' + model_identifier
    fps_key = 'ppo_fps_' + model_identifier
    loss_key = 'ppo_loss_' + model_identifier

    # Using my own logger here. Got rid of a few of those KPIs tho
    tb_logger.log_kv(fps_key, fps, update)
    tb_logger.log_kv(misc_key + "/explained_variance", float(ev), update)
    tb_logger.log_kv(misc_key + "/time_elapsed", tnow - tfirststart, update)
    for (lossval, lossname) in zip(lossvals, model.loss_names):
        tb_logger.log_kv(loss_key + '/' + lossname, lossval, update)


def get_mblossvals(
        nbatch, noptepochs, nbatch_train, lrnow, cliprangenow, model,
        obs, returns, masks, actions, values, neglogpacs, states):
    mblossvals = []
    if states is None: # nonrecurrent version
        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (
                    tf.constant(arr[mbinds])
                    for arr in (obs, returns, masks, actions, values, neglogpacs)
                )
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))
    else: # recurrent version
        raise ValueError('Not Support Yet')
    return mblossvals


# Avoid division error when calculate the mean (in our case if epinfo is empty
# returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
