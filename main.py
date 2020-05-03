import argparse
import time
import os
import matplotlib.pyplot as plt
import pickle as pkl

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='LSTM text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training [default: 16]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true', default=False,
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./LSTM_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=64,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='number of lstm hidden dimension [default: 128]')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM [default: False]')
parser.add_argument('-g', '--gpu', default='0', type=str)
parser.add_argument('--sum_dir', default='./save/sum')
parser.add_argument('--draw',action='store_true')
parser.add_argument('--draw_dir',default='./save/pics')
parser.add_argument('--save_dir',default='./save/default')
parser.add_argument('--pickle',action='store_true')
args = parser.parse_args()
torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if not os.path.exists(args.sum_dir):
	os.makedirs(args.sum_dir)
if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)
use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
###############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']

training_data = DataLoader(
	data['train']['src'],
	data['train']['label'],
	args.max_len,
	batch_size=args.batch_size,
	cuda=use_cuda)

validation_data = DataLoader(
	data['valid']['src'],
	data['valid']['label'],
	args.max_len,
	batch_size=args.batch_size,
	shuffle=False,
	cuda=use_cuda)

# ##############################################################################
# Build model
# ##############################################################################
import model

rnn = model.LSTM_Text(args)
if use_cuda:
	rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []


def repackage_hidden(h):
	if isinstance(h, Variable):
		if use_cuda:
			return Variable(h.data).cuda()
		return Variable(h.data)
	else:
		return tuple(repackage_hidden(v) for v in h)


def evaluate():
	rnn.eval()
	corrects = eval_loss = 0
	_size = validation_data.sents_size
	hidden = rnn.init_hidden()
	for data, label in tqdm(validation_data, mininterval=0.2,
	                        desc='Evaluate Processing', leave=False):
		hidden = repackage_hidden(hidden)
		pred, hidden = rnn(data, hidden)
		loss = criterion(pred, label.long())

		eval_loss += loss.item()
		corrects += (torch.max(pred, 1)[1].view(label.long().size()).data == label.long().data).sum()

	return eval_loss / _size, corrects, corrects * 100.0 / _size, _size


def train():
	rnn.train()
	total_loss = 0
	hidden = rnn.init_hidden()
	for data, label in tqdm(training_data, mininterval=1,
	                        desc='Train Processing', leave=False):
		optimizer.zero_grad()
		hidden = repackage_hidden(hidden)
		target, hidden = rnn(data, hidden)
		loss = criterion(target, label.long())

		loss.backward()
		optimizer.step()

		total_loss += loss.item()
	return total_loss / training_data.sents_size


# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

train_ac=[]
train_ls=[]
valid_ls=[]

try:
	print('-' * 90)
	start_time = time.time()
	for epoch in range(1, args.epochs + 1):
		epoch_start_time = time.time()
		loss = train()
		train_loss.append(loss * 1000.)
		train_ls.append(loss)

		print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time,
		                                                                      loss))

		loss, corrects, acc, size = evaluate()
		train_ac.append(acc.item())
		valid_ls.append(loss)
		valid_loss.append(loss)
		accuracy.append(acc)

		epoch_start_time = time.time()
		print('-' * 90)
		print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch,
		                                                                                             time.time() - epoch_start_time,
		                                                                                             loss, acc,
		                                                                                             corrects, size))
		print('-' * 90)
	end_time = time.time()
	if args.draw:
		plt.figure()
		ax1 = plt.subplot()
		plt.title("loss and accuracy of training and validating")
		ax1.set_xlabel("epochs")
		x = range(len(train_ac))
		ax1.set_ylabel("loss")
		ax2 = ax1.twinx()
		ax2.set_ylabel("accuracy")

		kwargs = {
			"marker": None,
			"lw": 2,
		}
		l1, = ax1.plot(x, train_ls, color="tab:blue", label="train loss", **kwargs)
		l3, = ax1.plot(x, valid_ls, color="tab:green", label="test loss", **kwargs)
		l4, = ax2.plot(x, train_ac, color="tab:red", label="test accuracy", **kwargs)

		plt.legend(handles=[l1, l3, l4], loc="center right")
		sv = plt.gcf()
		sv.savefig(os.path.join(args.save_dir, "lossAndAcc" + str(args.lr) +".png"), format="png", dpi=300)

	if args.pickle:
		with open (os.path.join(args.save_dir,"train_ac.pkl"),'wb') as f:
			pkl.dump(train_ac,file=f)
		with open (os.path.join(args.save_dir,"train_ls.pkl"),'wb') as f:
			pkl.dump(train_ls,file=f)
		with open (os.path.join(args.save_dir,"val_ls.pkl"),'wb') as f:
			pkl.dump(valid_ls,file=f)

	with open(args.sum_dir + '/sum.txt', 'a+') as f:
		print(args.lr, args.dropout, args.embed_dim, args.hidden_size, args.lstm_layers, args.bidirectional,
		      "%.2f" % (end_time - start_time), '%.3f' % accuracy[-1], file=f)
		print(args.lr, args.dropout, args.embed_dim, args.hidden_size, args.lstm_layers, args.bidirectional,
		      "%.2f" % (end_time - start_time), '%.3f' % accuracy[-1])
except KeyboardInterrupt:
	print("-" * 90)
	print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time) / 60.0))
