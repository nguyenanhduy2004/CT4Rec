from tf_compat import tf
from model import Model

class Args:
    pass

args = Args()
args.maxlen = 10
args.neg_sample_n = 5
args.hidden_units = 16
args.num_blocks = 1
args.num_heads = 1
args.dropout_rate = 0.1
args.l2_emb = 0.0
args.rd_alpha = 0.0
args.con_alpha = 0.0
args.rd_reduce = 'mean'
args.neg_test = 10
args.user_reg_type = 'kl'
args.lr = 0.001

print('TensorFlow version:', tf.__version__)
model = Model(usernum=20, itemnum=100, args=args)
print('Model graph built successfully')
