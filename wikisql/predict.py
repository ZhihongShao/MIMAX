import traceback

import torch
from Exceptions import *

def predict(args, dataset, posterior_net, logger, prior_net=None):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataset.config.batch_size = args.eval_batch_size
    # Note that DistributedSampler samples randomly

    # multi-gpu evaluate
    if args.n_gpu > 1:
        if prior_net is not None:
            prior_net = torch.nn.DataParallel(prior_net)
        posterior_net = torch.nn.DataParallel(posterior_net)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_steps = 0
    metrics = {}
    if prior_net is not None:
        prior_net.eval()
    posterior_net.eval()

    while True:
        try:
            batch = dataset.get_next_batch()
        except EndOfShardError:
            try:
                dataset.prepare_next_shard(
                    posterior_net,
                    prior_net
                )
                continue
            except EndOfEpochError:
                break
        guids = batch[0]
        dataset.set_current_batch_guids(guids)
        # sample from the posterior network
        posterior_inps = dataset.get_posterior_network_inputs(with_sample=False)
        try:
            with torch.no_grad():
                instances = [dataset.guid2instance[guid] for guid in dataset.guids]
                samples = dataset.infer_program(instances, posterior_net=posterior_net, infer=True, execute=True, compute_log_prob=False, augmented=False)
                batch_metrics = dataset.update_programs(posterior_net, prior_net, program_samples=samples, compute_log_prob=True)

            eval_steps += len(guids)
            for key, val in batch_metrics.items():
                metrics[key] = metrics.get(key, 0) + val * len(guids)
        except Exception as err:
            logger.info("ERROR FOUND DURING EVALUATION")
            logger.info(traceback.format_exc())

    result = {key: val / eval_steps for key, val in metrics.items()}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result, dataset.get_best_program_results(criterion='gen_prob'), dataset.get_guid2programs()
