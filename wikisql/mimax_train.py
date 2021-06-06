import os
import json
import random
import numpy as np
from tqdm import trange
import copy
import shutil

import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

import predict
import MemoisedDataset
from Exceptions import *

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, dirname, logger, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted(os.listdir(dirname), key=lambda x: int(x))
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        checkpoint = os.path.join(dirname, checkpoint)
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def _get_optimizer_and_scheduler(args, total_steps, model, learning_rate, model_path=None):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    # Check if saved optimizer or scheduler states exist
    if (
        model_path is not None
        and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    return optimizer, scheduler

def train(args, library, dataset, prior_net, posterior_net, logger):
    if args.local_rank in [-1, 0]:
        tb_prefix = 'train'
        tb_writer = SummaryWriter(args.summary_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    dataset.config.batch_size = args.train_batch_size

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(dataset) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(dataset) // args.gradient_accumulation_steps * args.num_train_epochs

    # Take care of distributed/parallel training
    prior_net = prior_net.module if hasattr(prior_net, "module") else prior_net
    posterior_net = posterior_net.module if hasattr(posterior_net, "module") else posterior_net

    prior_optimizer, prior_scheduler = _get_optimizer_and_scheduler(args, t_total, prior_net, args.prior_learning_rate, model_path=args.restore_prior_path)
    if args.plm_adaptive_lr:
        bert_optimizer = torch.optim.Adam(posterior_net.encoder.parameters(), lr=args.plm_learning_rate)
        other_optimizer = torch.optim.Adam(posterior_net.program_decoder.parameters(), lr=args.learning_rate)
    else:
        posterior_optimizer, posterior_scheduler = _get_optimizer_and_scheduler(args, t_total, posterior_net, args.learning_rate, model_path=args.restore_posterior_path)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        prior_net, prior_optimizer = amp.initialize(prior_net, prior_optimizer, opt_level=args.fp16_opt_level)
        posterior_net, posterior_optimizer = amp.initialize(posterior_net, posterior_optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        prior_net = torch.nn.DataParallel(prior_net)
        posterior_net = torch.nn.DataParallel(posterior_net)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        prior_net = torch.nn.parallel.DistributedDataParallel(
            prior_net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        posterior_net = torch.nn.parallel.DistributedDataParallel(
            posterior_net, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.should_continue:
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = args.global_step
            epochs_trained = global_step // (len(dataset) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(dataset) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss = {'prior': 0.0, 'posterior': 0.0}
    logging_loss = {'prior': 0.0, 'posterior': 0.0}
    tr_loss_bag = {'prior': {}, 'posterior': {}}
    logging_loss_bag = {'prior': {}, 'posterior': {}}
    best_eval_exe_acc, best_eval_logic_form_acc = 0.0, 0.0
    num_train, logging_num_train = 0, 0

    prior_net.zero_grad()
    posterior_net.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for epoch in train_iterator:
        step = 0
        while True:
            try:
                batch = dataset.get_next_batch()
                step += 1
            except EndOfShardError:
                try:
                    dataset.prepare_next_shard(
                        posterior_net,
                        prior_net,
                        dynamic_preparation=True
                    )
                    continue
                except EndOfEpochError:
                    dataset.increment_epoch()
                    break

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            guids = batch[0]
            dataset.set_current_batch_guids(guids)
            for model_type, model in zip(['prior', 'posterior'], [prior_net, posterior_net]):
                is_prior = (model_type == 'prior')
                if is_prior:
                    if args.mi_steps > -100 and global_step > args.mi_steps:
                        continue
                    prior_update_metrics = dataset.update_programs(posterior_net, None, compute_log_prob=True, subsample=args.subsample)
                    for key, value in prior_update_metrics.items():
                        if 'hit' in key:
                            tr_loss_bag['posterior'][key] = value + tr_loss_bag['posterior'].get(key, 0.0)
                    dataset.sample_program(for_posterior=False, use_other_net_for_selection=True)
                    inps = dataset.get_prior_network_inputs(with_sample=True)
                else:
                    if args.mi_steps > -100 and global_step > args.mi_steps:
                        update_metrics = dataset.update_programs(posterior_net, None, compute_log_prob=True, subsample=args.subsample)
                        update_model_key = 'posterior'
                        dataset.sample_program(for_posterior=True, use_other_net_for_selection=False)
                    else:
                        update_metrics = dataset.update_programs(None, prior_net, compute_log_prob=True, subsample=False)
                        update_model_key = 'prior'
                        dataset.sample_program(for_posterior=True, use_other_net_for_selection=True)
                    for key, value in update_metrics.items():
                        if 'hit' in key:
                            tr_loss_bag[update_model_key][key] = value + tr_loss_bag[update_model_key].get(key, 0.0)
                    inps = dataset.get_posterior_network_inputs(with_sample=True)

                if not is_prior:
                    num_train += inps['tensor_inps']['input_ids'].size(0)
                model.train()
                if model_type == 'posterior':
                    inps['tensor_inps'].update(inps['program_inps'])
                if global_step == 0 and args.local_rank in [-1, 0]:
                    logger.info(model_type)
                    for _i in range(5):
                        logger.info("instance #{}".format(_i))
                        for key, value in (inps['tensor_inps'].items() if model_type != 'prior' else inps.items()):
                            if value is not None and isinstance(value, torch.Tensor) and len(value) > _i:
                                if len(value.size()) == 2:
                                    value = value[_i].cpu().numpy().tolist()
                                elif len(value.size()) == 3:
                                    value = torch.sum(value, 2)[_i].long().cpu().numpy().tolist()
                                else:
                                    continue
                                logger.info("{}: {}".format(key, value))
                                if key in ['input_ids', 'decoder_input_ids']:
                                    key = key[:-3] + "tokens"
                                    tokenizer = library.prior_tokenizer if model_type == 'prior' else library.post_tokenizer
                                    logger.info("{}: {}".format(key, tokenizer.convert_ids_to_tokens(value)))
                if model_type == 'posterior':
                    res = model(**inps['tensor_inps'])
                else:
                    res = model(**inps)
                loss = None
                for key, value in res.items():
                    if args.n_gpu > 1:
                        value = value.mean()  # mean() to average on multi-gpu parallel training
                    if 'loss' in key:
                        loss = value if loss is None else loss + value
                    if 'loss' in key or key.startswith('rec_'):
                        tr_loss_bag[model_type][key] = value.item() + tr_loss_bag[model_type].get(key, 0.0)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss[model_type] += loss.item()

            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(prior_optimizer), args.max_grad_clip)
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(posterior_optimizer), args.max_grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(prior_net.parameters(), args.max_grad_clip)
                    # torch.nn.utils.clip_grad_norm_(posterior_net.parameters(), args.max_grad_clip)
                prior_optimizer.step()
                prior_scheduler.step()  # Update learning rate schedule
                prior_net.zero_grad()
                if args.plm_adaptive_lr:
                    bert_optimizer.step()
                    other_optimizer.step()
                else:
                    posterior_optimizer.step()
                    posterior_scheduler.step()
                posterior_net.zero_grad()
                global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar("{}/num_instance_trained".format(tb_prefix), (num_train - logging_num_train) / args.logging_steps, global_step)
                logger.info("step: {}".format(global_step))
                logger.info("\tnum_instance_trained: {}".format((num_train - logging_num_train) / args.logging_steps))
                tb_writer.add_scalar("{}/prior/lr".format(tb_prefix), prior_scheduler.get_lr()[0], global_step)
                logger.info("\tprior/lr: {:.5f}".format(prior_scheduler.get_lr()[0]))
                tb_writer.add_scalar("{}/posterior/lr".format(tb_prefix), args.learning_rate if args.plm_adaptive_lr else posterior_scheduler.get_lr()[0], global_step)
                logger.info("\tposterior/lr: {:.5f}".format(args.learning_rate if args.plm_adaptive_lr else posterior_scheduler.get_lr()[0]))
                for key in ['prior', 'posterior']:
                    tb_writer.add_scalar("{}/{}/loss".format(tb_prefix, key), (tr_loss[key] - logging_loss[key]) / args.logging_steps, global_step)
                    logger.info("\t{}/loss: {:.5f}".format(key, (tr_loss[key] - logging_loss[key]) / args.logging_steps))
                    for sub_key, value in tr_loss_bag[key].items():
                        tb_writer.add_scalar("{}/{}/{}".format(tb_prefix, key, sub_key), (value - logging_loss_bag[key].get(sub_key, 0.0)) / args.logging_steps, global_step)
                        logger.info("\t{}/{}: {:.5f}".format(key, sub_key, (value - logging_loss_bag[key].get(sub_key, 0.0)) / args.logging_steps))
                logging_loss = copy.deepcopy(tr_loss)
                logging_loss_bag = copy.deepcopy(tr_loss_bag)
                logging_num_train = num_train

            if args.local_rank != -1 and args.save_steps > 0 and global_step % args.save_steps == 0:
                dataset.synchronize()
                if args.local_rank == 1:
                    dataset_path = os.path.join(args.dataset_ckpt_dir, str(global_step))
                    os.makedirs(dataset_path, exist_ok=True)
                    dataset.save(dataset_path, args.guid2programs_file)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.local_rank not in [-1, 0]:
                    torch.distributed.barrier()
                    continue
                save_to_best_dir = True
                eval_programs = None
                # Log metrics
                if (
                    # args.local_rank == -1 and args.evaluate_during_training
                    args.evaluate_during_training
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    logger.info("Evaluating ...")
                    eval_dataset = MemoisedDataset.MemoisedDataset(args.predict_dev_file, args.predict_dev_table_file, args.predict_dev_db_file, library, MemoisedDataset.MemoisedDataset.EVAL_MODE, args.local_rank)
                    posterior_net_cpy = copy.deepcopy(posterior_net.module if hasattr(posterior_net, 'module') else posterior_net)
                    posterior_net_cpy.to(args.device)
                    prior_net_cpy = copy.deepcopy(prior_net.module if hasattr(prior_net, 'module') else prior_net)
                    prior_net_cpy.to(args.device)
                    eval_results, eval_programs, eval_guid2programs = predict.predict(args, eval_dataset, posterior_net_cpy, logger, prior_net_cpy)
                    test_results, test_programs, test_guid2programs = None, None, None
                    del posterior_net_cpy
                    del prior_net_cpy
                    for prefix, _results in zip(['eval', 'test'], [eval_results, test_results]):
                        if _results is None:
                            continue
                        for key, value in _results.items():
                            tb_writer.add_scalar("{}_{}".format(prefix, key), value, global_step)
                            logger.info("\t{}_{}: {:.5f}".format(prefix, key, value))
                    _results = eval_results
                    if [_results['exe_acc'], _results['logic_form_acc']] < [best_eval_exe_acc, best_eval_logic_form_acc]:
                        save_to_best_dir = False
                    else:
                        best_eval_exe_acc, best_eval_logic_form_acc = _results['exe_acc'], _results['logic_form_acc']

                prior_net_to_save = (
                    prior_net.module if hasattr(prior_net, "module") else prior_net
                )  # Take care of distributed/parallel training
                posterior_net_to_save = (
                    posterior_net.module if hasattr(posterior_net, "module") else posterior_net
                )
                prior_path = prior_net_to_save.save(args.ckpt_dir, save_to_best_dir=save_to_best_dir, global_step=global_step)
                posterior_path = posterior_net_to_save.save(args.ckpt_dir, save_to_best_dir=save_to_best_dir, global_step=global_step)
                if eval_programs is not None:
                    json.dump(eval_programs, open(os.path.join(posterior_path, "eval_program.json"), "w"))
                    json.dump(eval_guid2programs, open(os.path.join(posterior_path, "eval_guid2programs.json"), "w"))
                    if test_programs is not None:
                        json.dump(test_programs, open(os.path.join(posterior_path, "test_program.json"), "w"))
                        json.dump(test_guid2programs, open(os.path.join(posterior_path, "test_guid2programs.json"), "w"))
                logger.info("Saving prior network checkpoint to %s", prior_path)
                logger.info("Saving posterior network checkpoint to %s", posterior_path)

                _rotate_checkpoints(args, os.path.dirname(prior_path), logger)
                _rotate_checkpoints(args, os.path.dirname(posterior_path), logger)

                for optimizer, scheduler, path in zip([prior_optimizer, (None if args.plm_adaptive_lr else posterior_optimizer)], [prior_scheduler, (None if args.plm_adaptive_lr else posterior_scheduler)], [prior_path, posterior_path]):
                    if optimizer is not None:
                        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", path)

                if args.local_rank == 0:
                    torch.distributed.barrier()

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step
