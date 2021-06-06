import os
import argparse

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--do_train", action='store_true')

    parser.add_argument("--use_prior", action='store_true')
    parser.add_argument("--alpha", type=float, default=0.5, help="The initial threshold for hard-em-thres")
    parser.add_argument("--gamma", type=float, default=0.5, help="Threshold decay rate")

    parser.add_argument("--pretrained_model_type_for_prior", type=str, default="bart-base")
    parser.add_argument("--pretrained_model_type_for_posterior", type=str, default="bert-base-uncased")
    parser.add_argument("--pretrained_model_dir", type=str, help="the directory where you cache PLMs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--summary_dir", type=str, default="runs")
    parser.add_argument("--online_ckpt_dir", type=str, default="")
    parser.add_argument("--online_summary_dir", type=str, default="")
    parser.add_argument("--input_train_file", type=str, default="")
    parser.add_argument("--predict_train_file", type=str, default="")
    parser.add_argument("--input_dev_file", type=str, default="")
    parser.add_argument("--predict_dev_file", type=str, default="")
    parser.add_argument("--input_test_file", type=str, default="")
    parser.add_argument("--predict_test_file", type=str, default="")
    parser.add_argument("--guid2programs_file", type=str, default="guid2programs.json")
    parser.add_argument("--vocab_file", type=str, default="vocab.txt")

    parser.add_argument("--do_lower_case", action='store_true')

    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--should_continue", action='store_true')
    parser.add_argument("--continue_from_best_ckpt", action='store_true')

    parser.add_argument("--loss_type", type=str, default="", help="hard-em-thres, hard-em, mml, mimax")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0, help="1000 for pretrain with batch_size 32")
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--max_grad_clip", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--update_per_batch", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--mi_steps", type=int, default=-200)
    parser.add_argument("--save_total_limit", type=int, default=-1)
    parser.add_argument("--evaluate_during_training", action='store_true')

    parser.add_argument("--n_best_size", default=3, type=int, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--max_n_answers', type=int, default=20)
    parser.add_argument('--n_paragraphs', type=str, default='40')
    parser.add_argument('--tau', type=int, default=20000)
    parser.add_argument('--verbose', action="store_true", default=False)

    args, _ = parser.parse_known_args()
    if args.do_train:
        assert args.loss_type in ['mimax', 'hard-em-thres', 'mml', 'first-only']
    args.do_predict = not (args.do_train or args.do_pretrain)
    args.dataset_ckpt_dir = os.path.join(args.ckpt_dir, "dataset")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.dataset_ckpt_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)
    args.pretrained_model_dir_for_posterior = os.path.join(args.pretrained_model_dir, args.pretrained_model_type_for_posterior)
    args.pretrained_model_dir_for_prior = os.path.join(args.pretrained_model_dir, args.pretrained_model_type_for_prior)
    args.post_vocab_dir = args.pretrained_model_dir_for_posterior
    args.prior_vocab_dir = args.pretrained_model_dir_for_prior
    args.predict_train_file = ",".join(os.path.join(args.data_dir, filename) for filename in args.predict_train_file.split(","))
    args.predict_dev_file = os.path.join(args.data_dir, args.predict_dev_file)
    if args.predict_test_file:
        args.predict_test_file = os.path.join(args.data_dir, args.predict_test_file)
    return args
