import os
import logging
import json
import random
import numpy as np

import configs
import WikiSQLLibrary
import MemoisedDataset
import NeRdDataset
from GenerativeModels import Prior, Posterior
import mimax_train
import train
import predict

import torch

args = configs.get_configs()

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    library = WikiSQLLibrary.WikiSQLLibrary(1000, 1000, args.prior_vocab_dir, args.post_vocab_dir)
    if args.use_prior:
        prior_config = Prior.PriorConfig()
        prior_config.pretrain_model_name_or_path = args.pretrained_model_dir_for_prior
        prior_model = Prior.Prior(library, config=prior_config)
    else:
        prior_model = None
    posterior_config = Posterior.PosteriorConfig()
    posterior_config.pretrain_model_name_or_path = args.pretrained_model_dir_for_posterior
    posterior_model = Posterior.Posterior(library, config=posterior_config)

    if args.should_continue:
        map_location = {'cuda:0': 'cpu'}
        posterior_global_step, args.restore_posterior_path = posterior_model.restore(args.ckpt_dir, restore_best=args.continue_from_best_ckpt, map_location=map_location, restore_ema_model=(args.do_predict and args.use_ema_model_for_eval))
        args.global_step = posterior_global_step
        logger.info("Posterior Network continues from checkpoint {}".format(args.restore_posterior_path))
        if args.use_prior:
            prior_global_step, args.restore_prior_path = prior_model.restore(args.ckpt_dir, restore_best=args.continue_from_best_ckpt, map_location=map_location, restore_ema_model=(args.do_predict and args.use_ema_model_for_eval))
            assert prior_global_step == posterior_global_step
            logger.info("Prior Network continues from checkpoint {}".format(args.restore_prior_path))
    else:
        args.global_step = -1
        args.restore_prior_path, args.restore_posterior_path = None, None
        logger.info("Training from scratch")

    guid2programs_file = None

    if args.device != torch.device('cpu'):
        if args.use_prior:
            prior_model.move_to_device(args.device)
        posterior_model.move_to_device(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    if args.do_pretrain:
        dataset_config = NeRdDataset.NeRdDatasetConfig()
        dataset_config.alpha = args.alpha
        dataset_config.gamma = args.gamma
        dataset_config.subsample = args.subsample
        dataset_config.extreme_case = args.extreme_case
        dataset_config.set_loss_type(args.loss_type)
        dataset = NeRdDataset.NeRdDataset(args.predict_train_file, args.predict_train_table_file, args.predict_train_db_file, library, NeRdDataset.NeRdDataset.TRAIN_MODE, args.local_rank, num_shards=args.num_shards, guid2programs_file=guid2programs_file, config=dataset_config)
    else:
        dataset_config = MemoisedDataset.MemoisedDatasetConfig()
        dataset_config.set_loss_type(args.loss_type)
        dataset_config.subsample = args.subsample
        dataset_config.extreme_case = args.extreme_case
        dataset = MemoisedDataset.MemoisedDataset(
            args.predict_train_file if args.do_train else args.predict_test_file,
            args.predict_train_table_file if args.do_train else args.predict_test_table_file,
            args.predict_train_db_file if args.do_train else args.predict_test_db_file,
            library,
            MemoisedDataset.MemoisedDataset.TRAIN_MODE if args.do_train else MemoisedDataset.MemoisedDataset.EVAL_MODE,
            args.local_rank,
            num_shards=args.num_shards,
            guid2programs_file=guid2programs_file,
            config=dataset_config
        )

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train and args.loss_type != 'mimax':
        train.train(args, library, dataset, prior_model, posterior_model, logger)
    elif args.do_train and args.loss_type == 'mimax':
        mimax_train.train(args, library, dataset, prior_model, posterior_model, logger)
    else:
        res, best_programs, guid2programs = predict.predict(args, dataset, posterior_model, logger, prior_net=prior_model)
        infer_dir = args.restore_posterior_path
        json.dump(best_programs, open(os.path.join(infer_dir, "test_program.json"), "w"))
        json.dump(guid2programs, open(os.path.join(infer_dir, "test_guid2programs"), "w"))

if __name__ == '__main__':
    main()
