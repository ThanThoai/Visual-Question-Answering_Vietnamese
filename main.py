from src.core.cfg import Config
from src.app import App
import argparse
import yaml


def parse_args():
   parser = argparse.ArgumentParser(description='VQA Vietnamese')

   parser.add_argument('--RUN', dest='RUN_MODE',
                     choices=['TRAIN', 'VAL', 'TEST'],
                     help='{TRAIN, VAL, TEST}',
                     type=str, required=True)

   parser.add_argument('--MODEL', dest='MODEL',
                     choices=[
                        'mcan_small',
                        'mcan_large'
                        ]
                     ,
                     help='{'
                        'mcan_small,'
                        'mcan_large,'
                        '}'
                     ,
                     type=str, required=True)
   
   parser.add_argument('--EMDED', dest='EMDEDING_METHOD',
                       choices = [
                          'BERT_BASE',
                          'BERT_LARGE'
                          ]
                        ,
                        help='{'
                              'Bert model'
                              '}'
                        ,
                        type=str, required=True)


   parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                     choices=['TRAIN', 'TRAIN+VAL'],
                     type=str)

   parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                     choices=['True', 'False'],
                     help='True: evaluate the val split when an epoch finished,'
                           'False: do not evaluate on local',
                     type=str)

   parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                     choices=['True', 'False'],
                     help='True: save the prediction vectors,'
                           'False: do not save the prediction vectors',
                     type=str)

   parser.add_argument('--BS', dest='BATCH_SIZE',
                     help='batch size in training',
                     type=int, default=32)

   parser.add_argument('--GPU', dest='GPU',
                     help="gpu choose, eg.'0, 1, 2, ...'",
                     type=str, default='0')

   parser.add_argument('--SEED', dest='SEED',
                     help='fix random seed',
                     type=int, default=42)

   parser.add_argument('--VERSION', dest='VERSION',
                     help='version control',
                     type=str)

   parser.add_argument('--RESUME', dest='RESUME',
                     choices=['True', 'False'],
                     help='True: use checkpoint to resume training,'
                        'False: start training with random init',
                     type=str)

   parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                     help='checkpoint version',
                     type=str)

   parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                     help='checkpoint epoch',
                     type=int)

   parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                     help='load checkpoint path, we '
                        'recommend that you use '
                        'CKPT_VERSION and CKPT_EPOCH '
                        'instead, it will override'
                        'CKPT_VERSION and CKPT_EPOCH',
                     type=str)

   parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                     help='split batch to reduce gpu memory usage',
                     type=int)

   parser.add_argument('--NW', dest='NUM_WORKERS',
                     help='multithreaded loading to accelerate IO',
                     type=int)

   parser.add_argument('--PINM', dest='PIN_MEM',
                     choices=['True', 'False'],
                     help='True: use pin memory, False: not use pin memory',
                     type=str)

   parser.add_argument('--VERB', dest='VERBOSE',
                     choices=['True', 'False'],
                     help='True: verbose print, False: simple print',
                     type=str)


   args = parser.parse_args()
   return args



if __name__ == '__main__':
   args = parse_args()

   cfg_file = "configs/{}.yml".format(args.DATASET, args.MODEL)
   with open(cfg_file, 'r') as f:
      yaml_dict = yaml.load(f)

   __C = Config(yaml_dict['MODEL_USE']).load()
   args = __C.str_to_bool(args)
   args_dict = __C.parse_to_dict(args)

   args_dict = {**yaml_dict, **args_dict}
   __C.add_args(args_dict)
   __C.proc()

   print('Hyper Parameters:')
   print(__C)

   app = App(__C)
   app.run(__C.RUN_MODE)