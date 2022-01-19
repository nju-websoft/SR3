import argparse
from others.logging import *
import random
from pytorch_transformers import BertTokenizer
from model_graph.generator import AbsSummarizer
from model_graph.split_generator import splitAbsSummarizer
from model_graph.copyAfterGen import SentCopySummarizer
from model_graph.classsifyGenerator import classsifyGenerator
from model_graph.encoder import Bert
import signal
from model_graph.tokenizer_graph import BertData
import model_graph.gen_optimizer as gen_optimizer
import model_graph.optimizer as optimizer
from model_graph.loss import abs_loss,abs_loss_gen,abs_loss_gen_withclassfiy,abs_loss_withclassfiy#,abs_loss_copysent
from model_graph.data import *
import model_graph.data_multidoc as multidoc
from model_graph.trainer import *
from model_graph.predictor import build_predictor
from model_graph.extract import LexicalMap
from model_graph.extract import read_file
import torch
import glob
model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv']

def train_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()

def validate_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    print(cp_files)

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    cp_idx=0
    procs = []
    for i in range(nb_gpu):
        device_id = i
        if cp_idx>=len(cp_files):
            break
        test_from = cp_files[cp_idx]
        step = int(cp_files[cp_idx].split('.')[-2].split('_')[-1])

        procs.append(mp.Process(target=runval, args=(args,
                                                  device_id, error_queue,step,test_from,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
        cp_idx+=1
    for p in procs:
        p.join()

def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))

def runval(args, device_id, error_queue,step,test_from):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        test(args,step,"val", device_id,test_from)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def train(args, device_id):

    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.gpu == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    encoder = Bert(args.encoder_name, args.temp_dir, finetune=True)
    vocabs = dict()
    tokenizer = BertData(args)
    vocabs['tokens'] = tokenizer.tokenizer.vocab
    if args.dataset == "wikihow":
        vocabs['relation'] = Vocab(
            ['q_senario', 'senario_q', 'samedoc'] + [CLS, rCLS, SEL, TL])
    else:
        if args.comfirm_connect:
            vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity'] + [CLS, rCLS, SEL, TL])
        else:
            vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity','not_connect'] + [CLS, rCLS, SEL, TL])
    lexical_mapping = LexicalMap()

    if args.dataset == "wikihow":
        # if args.world_size==1:
        #     train_data = torch.load(args.data_path+'train_graph.bert')
        #     def train_iter_fct():
        #         return DataLoader(args, vocabs, tokenizer, train_data, args.train_batch_size, device, for_train=True)
        # else:
        if args.multidoc:
            def train_iter_fct():
                return multidoc.DataloaderWikihow(args, vocabs, tokenizer, multidoc.load_train_dataset(args), args.train_batch_size,
                                         device, for_train=True)
        else:
            def train_iter_fct():
                return DataloaderWikihow(args,vocabs, tokenizer, load_train_dataset(args), args.train_batch_size, device,for_train=True)
    else:
        train_data = torch.load(args.data_path + 'train_graph.bert')
        # train_data = read_file(args.data_path+'train_graph.json',args,tokenizer,lexical_mapping)
        if args.multidoc:
            def train_iter_fct():
                return multidoc.DataLoader(args, vocabs, tokenizer, train_data, args.train_batch_size, device, for_train=True)
        else:
            def train_iter_fct():
                return DataLoader(args,vocabs, tokenizer, train_data, args.train_batch_size, device,for_train=True)

    symbols = {'BOS': tokenizer.tokenizer.vocab['[unused1]'], 'EOS': tokenizer.tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.tokenizer.vocab['[unused3]']}

    if args.multidoc:
        model = classsifyGenerator(args, encoder, device, checkpoint, vocabs=vocabs)
    elif args.train_copy:
        if args.gen_from != '':
            logger.info('Loading checkpoint from %s' % args.gen_from)
            gencheckpoint = torch.load(args.gen_from,
                                    map_location=lambda storage, loc: storage)
            # opt = vars(checkpoint['opt'])
            # for k in opt.keys():
            #     if (k in model_flags):
            #         setattr(args, k, opt[k])
        else:
            gencheckpoint = None
        model = SentCopySummarizer(args, encoder, device, checkpoint, gencheckpoint,vocabs=vocabs)
    elif args.split_gen:
        model = splitAbsSummarizer(args, encoder, device, checkpoint, vocabs=vocabs)
    else:
        model = AbsSummarizer(args, encoder,device, checkpoint,vocabs=vocabs)
        # model = splitAbsSummarizer(args, encoder, device, checkpoint, vocabs=vocabs)

    if (args.sep_optim):
        # optim_bert = model_builder.build_optim_bert(args, model, None)
        # # optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        # optim_dec_exsit = model_builder.build_optim_dec_exsit(args, model, None)
        # optim_dec_nonexsit = model_builder.build_optim_dec_nonexsit(args, model, None)
        if checkpoint is None:
            optim_bert = gen_optimizer.build_optim_bert(args, model, checkpoint)
            optim_dec = gen_optimizer.build_optim_dec(args, model, checkpoint)
            optim = [optim_bert, optim_dec]
        else:
            if not args.train_copy:
                optim_bert = gen_optimizer.build_optim_bert(args, model, checkpoint)
                optim_dec = gen_optimizer.build_optim_dec(args, model, checkpoint)
                optim = [optim_bert, optim_dec]
            else:
                optim_dec = gen_optimizer.build_optim_dec(args, model, checkpoint)
                optim = [optim_dec]
    else:
        optim = [optimizer.build_optim(args, model, checkpoint)]

    logger.info(model)
    # logger.info(model.named_parameters())
    # for n,p in list(model.named_parameters()):
    #     print(n)
    # if args.train_copy:
    #     train_loss = abs_loss_copysent(symbols, model.vocab_size+max_node_size, device, train=True,
    #                           label_smoothing=args.label_smoothing)
    if args.copy_decoder and args.copy_sent and args.split_qm:
        if args.classify:
            train_loss = abs_loss_gen_withclassfiy(symbols, model.vocab_size + args.max_node_size, model.vocab_size, device,
                                      train=True,
                                      label_smoothing=args.label_smoothing)
        else:
            train_loss = abs_loss_gen(symbols, model.vocab_size+args.max_node_size,model.vocab_size, device, train=True,
                                          label_smoothing=args.label_smoothing)
    else:
        if args.classify:
            train_loss = abs_loss_withclassfiy(symbols, model.vocab_size, device, train=True,
                              label_smoothing=args.label_smoothing)
        else:
            train_loss = abs_loss(symbols, model.vocab_size, device, train=True,
                              label_smoothing=args.label_smoothing)
    trainer = build_trainer(args, device_id, model, optim, train_loss)

    if(args.do_eval):
        if args.dataset == "wikihow":
            test_data = torch.load(args.data_path + 'val_graph.bert')
        else:
            test_data = torch.load(args.data_path + 'val_graph.bert')
            # test_data = read_file(args.data_path + 'val_graph.json', args, tokenizer, lexical_mapping)
        if args.multidoc:
            def test_iter_fct():
                return multidoc.DataLoader(args, vocabs, tokenizer, test_data, args.test_batch_size, device, for_train=False)
        else:
            def test_iter_fct():
                return  DataLoader(args,vocabs, tokenizer, test_data, args.test_batch_size, device,for_train=False)

        predictor = build_predictor(args, tokenizer.tokenizer, symbols, model, logger)

        trainer.train(train_iter_fct, args.train_steps,valid_iter_fct=test_iter_fct,valid_steps=args.valid_steps,predictor=predictor)
    else:
        trainer.train(train_iter_fct, args.train_steps)
    if(args.do_test):
        checkpoint = torch.load(args.model_path+'/model_step_0.pt', map_location=lambda storage, loc: storage)
        if args.multidoc:
            model = classsifyGenerator(args, encoder, device, checkpoint, vocabs=vocabs)
        elif args.train_copy:
            model = SentCopySummarizer(args, encoder, device, checkpoint, None, vocabs=vocabs)
        elif args.split_gen:
            model = splitAbsSummarizer(args, encoder, device, checkpoint, vocabs=vocabs)
        else:
            model = AbsSummarizer(args, encoder, device, checkpoint, vocabs=vocabs)
        model.eval()
        if args.dataset == "wikihow":
            test_data = torch.load(args.data_path + 'test_graph.bert')
        else:
            test_data = torch.load(args.data_path + 'test_graph.bert')
            # test_data = read_file(args.data_path + 'test_graph.json', args, tokenizer, lexical_mapping)
        # test_iter = DataLoader(args, vocabs, tokenizer, test_data, args.test_batch_size, device, for_train=False)
        if args.classify:
            test_iter= multidoc.DataLoader(args, vocabs, tokenizer, test_data, args.test_batch_size, device, for_train=False)
        else:
            test_iter=  DataLoader(args,vocabs, tokenizer, test_data, args.test_batch_size, device,for_train=False)

        predictor = build_predictor(args, tokenizer.tokenizer, symbols, model, logger)
        predictor.translate(test_iter, 0)

def validateAll(args):
    import glob
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    for i, cp in enumerate(cp_files):
        step = int(cp.split('.')[-2].split('_')[-1])
        if (step < args.test_start_from):
            continue
        args.test_from = cp
        test(args, step,"val")

def test(args, step,dataset='test',device=None,test_from=None):
    vocabs = dict()
    tokenizer = BertData(args)
    vocabs['tokens'] = tokenizer.tokenizer.vocab
    if args.dataset == "wikihow":
        vocabs['relation'] = Vocab(
            ['q_senario', 'senario_q', 'samedoc'] + [CLS, rCLS, SEL, TL])
    else:
        if args.comfirm_connect:
            vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity'] + [CLS, rCLS, SEL, TL])
        else:
            vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity','not_connect'] + [CLS, rCLS, SEL, TL])
    lexical_mapping = LexicalMap()
    if device==None:
        device = "cpu" if args.gpu == '-1' else "cuda"
    if test_from==None:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    # tokenizer = BertData(args)
    if args.dataset == "wikihow":
        test_data = torch.load(args.data_path + dataset+'_graph.bert')
    else:
        test_data = torch.load(args.data_path + dataset + '_graph.bert')
        # test_data = read_file(args.data_path + dataset+'_graph.json', args, tokenizer, lexical_mapping)
    # test_iter = DataLoader(args,vocabs, tokenizer, test_data, args.test_batch_size,device, for_train=False)
    if args.multidoc:
        test_iter = multidoc.DataLoader(args, vocabs, tokenizer, test_data, args.test_batch_size, device,
                                        for_train=False)
    else:
        test_iter = DataLoader(args, vocabs, tokenizer, test_data, args.test_batch_size, device, for_train=False)

    # datas=[]
    # for data in test_iter:
    #     datas.append(data)
    # print(datas)

    # encoder = Bert(args.encoder_name, args.temp_dir, finetune=True)
    # model = AbsSummarizer(args, encoder, device, None,vocabs=vocabs)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)
    symbols = {'BOS': tokenizer.tokenizer.vocab['[unused1]'], 'EOS': tokenizer.tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.tokenizer.vocab['[unused3]']}

    encoder = Bert(args.encoder_name, args.temp_dir, finetune=True)
    if args.multidoc:
        model = classsifyGenerator(args, encoder, device, checkpoint, vocabs=vocabs)
    elif args.split_gen:
        model=splitAbsSummarizer(args, encoder, device, checkpoint,vocabs=vocabs)
    else:
        model = AbsSummarizer(args, encoder, device, checkpoint,vocabs=vocabs)

    # model = AbsSummarizer(args, device, checkpoint)
    model.eval()


    predictor = build_predictor(args, tokenizer.tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_args(args):
    args.temp_dir = "/home/ychen/PreSumm/temp"
    args.data_path ="graph_data/"
    args.model_path = "graph_data/model/"
    args.log_file = "graph_data/logs/log/"
    args.result_path = "graph_data/results/"
    args.rnn_hidden_size = 256
    args.rnn_num_layers = 2
    args.rel_dim = 100
    args.embed_dim = 768
    args.ff_embed_dim = 1024
    args.num_heads = 8
    args.snt_layers = 1
    args.graph_layers = 4
    args.inference_layers = 3
    args.copy_decoder = True
    args.split_qm = True
    args.copy_word = True
    args.encode_q = True
    args.comfirm_connect = False
    args.valid_step = 1
    args.warmup_steps_bert = 1
    args.gpu = '2'
    args.test_batch_size = 8
    args.mode = "test"
    args.test_from = "graph_data/model/model_step_10000.pt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder_name", default='bert-base-chinese', type=str)
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'dev'])
    parser.add_argument("-test_start_from", default=0, type=int)

    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-data_path", default='gen_data/pre_data/')
    parser.add_argument("-model_path", default='gen_data/models/')
    parser.add_argument("-result_path", default='gen_data/results/')
    parser.add_argument('-log_file', default='gen_data/logs/log')
    parser.add_argument('-temp_dir', default='../temp/')

    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument('-gpu', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-world_size', default='1', type=str)

    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-train_batch_size", default=20000, type=int)
    parser.add_argument("-test_batch_size", default=20000, type=int)
    parser.add_argument("-summary_size", default=10, type=int)
    parser.add_argument("-block_trigram", default=True, type=bool)
    parser.add_argument("-lr", default=1e-5, type=float)
    parser.add_argument("-warmup_steps", default=2000, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-save_checkpoint_steps", default=10000, type=int)
    parser.add_argument("-train_steps", default=10000, type=int)
    parser.add_argument("-do_eval", default=True, type=str2bool)
    parser.add_argument("-do_test", default=True, type=str2bool)
    parser.add_argument("-valid_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=5, type=int)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)

    #gen parameter
    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=200, type=int)
    parser.add_argument("-warmup_steps_bert", default=2000, type=int)
    parser.add_argument("-warmup_steps_dec", default=2000, type=int)

    parser.add_argument("-copy_decoder", default=False, type=bool)
    parser.add_argument("-split_qm", default=False, type=bool)
    parser.add_argument("-use_ext", default=False, type=str2bool)
    parser.add_argument("-copy_ext", default=False, type=str2bool)


    # relation encoder
    parser.add_argument('-rel_dim', type=int)
    parser.add_argument('-rnn_hidden_size', type=int)
    parser.add_argument('-rnn_num_layers', type=int)

    # core architecture
    parser.add_argument('-ff_embed_dim', type=int)
    parser.add_argument('-num_heads', type=int)
    parser.add_argument('-snt_layers', type=int)
    parser.add_argument('-graph_layers', type=int)
    parser.add_argument('-inference_layers', type=int)

    parser.add_argument('-copy_word', type=str2bool,default=False)
    parser.add_argument('-copy_sent', type=str2bool, default=False)
    parser.add_argument('-encode_q', type=str2bool, default=False)
    parser.add_argument('-split_gen', type=str2bool, default=False)
    parser.add_argument('-sent_attn', type=str2bool, default=False)
    parser.add_argument('-train_copy', type=str2bool, default=False)
    parser.add_argument("-gen_from", default='')
    parser.add_argument("-comfirm_connect", default=True, type=str2bool)
    parser.add_argument("-use_cls", default=False, type=str2bool)
    parser.add_argument('-max_node_size', type=int, default=5)
    parser.add_argument('-min_copy_rouge', type=float, default=0.3)
    parser.add_argument('-use_rouge_f', type=str2bool, default=True)
    parser.add_argument("-dataset", default='geo', type=str, choices=['geo', 'wikihow'])
    parser.add_argument("-max_string_len", default=300, type=int)
    parser.add_argument("-temp_eval_dir", default="", type=str)
    parser.add_argument("-avg_sent", default=False, type=str2bool)
    parser.add_argument("-graph_transformer", default=False, type=str2bool)
    parser.add_argument("-classify", default=False, type=str2bool)
    parser.add_argument("-multidoc", default=False, type=str2bool)
    parser.add_argument("-max_doc_size", default=1, type=int)

    args = parser.parse_args()
    # set_args(args)
    args.gpu_ranks = [int(i) for i in range(len(args.gpu.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # args.world_size = len(args.gpu_ranks)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    init_logger(args.log_file)
    device = "cpu" if args.gpu == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    # args.mode="test"
    # args.use_ext=True
    # args.data_path="gen_data/pre_ext_data/"
    if args.mode=="train":
        if args.world_size>1:
            train_multi(args)
        else:
            train(args, device_id)
    elif args.mode=="test":
        cp=args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test(args,step,'test')
    elif args.mode=="dev":
        if args.test_from =="":
            if args.world_size>1:
                validate_multi(args)
            else:
                validateAll(args)
        else:
            cp=args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test(args,step,'val')