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
                    posterior_net if prior_net is None else prior_net
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
                res = posterior_net.forward(**posterior_inps)
                dataset.update_programs(posterior_net if prior_net is None else prior_net, start_logits=res['start_logits'], end_logits=res['end_logits'], switch_logits=res['switch_logits'], compute_log_prob=False)

        except Exception as err:
            logger.info("ERROR FOUND DURING EVALUATION")
            logger.info(traceback.format_exc())

    results, all_predictions, all_nbest_json = dataset.get_best_program_results(logger, args.n_best_size, do_lower_case=True, verbose_logging=True, n_paragraphs=[int(n) for n in args.n_paragraphs.split(',')])

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, all_predictions, all_nbest_json
