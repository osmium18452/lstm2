# time = 1000
# m, s = divmod(time, 60)
# h, m = divmod(m, 60)
# print("%02d:%02d:%02d" % (h, m, s))
# exit(0)

with open("train.sh", "w+") as f:
	for lr in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001):
		for dropout in (0.5, 0.7, 0.9, 1.):
			for embdim in (32, 64, 96, 128):
				for hidden_size in (64, 96, 128, 156):
					for layers in (2, 3, 4, 5):
						print("python ./main.py --lr", lr, "--epochs", 100, '--dropout', dropout, '--embed-dim', embdim,
						      '--hidden-size', hidden_size, '--lstm-layers', layers, '--cuda-able', '--save_dir',
						      './sum/1', file=f)
	for lr in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001):
		for dropout in (0.5, 0.7, 0.9, 1.):
			for embdim in (32, 64, 96, 128):
				for hidden_size in (64, 96, 128, 156):
					for layers in (2, 3, 4, 5):
						print("python ./main.py --lr", lr, "--epochs", 100, '--dropout', dropout, '--embed-dim', embdim,
						      '--hidden-size', hidden_size, '--lstm-layers', layers, '--cuda-able', '--bidirectional',
						      '--sum_dir', './save/1', file=f)