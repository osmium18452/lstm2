# time = 1000
# m, s = divmod(time, 60)
# h, m = divmod(m, 60)
# print("%02d:%02d:%02d" % (h, m, s))
# exit(0)

# with open("train.sh", "w+") as f:
# 	for lr in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001):
# 		for dropout in (0.5, 0.7, 1.):
# 			for embdim in (32, 64, 96, 128):
# 				for hidden_size in (64, 96, 128, 156):
# 					for layers in (2, 3, 4, 5):
# 						print("python ./main.py --lr", lr, "--epochs", 100, '--dropout', dropout, '--embed-dim', embdim,
# 						      '--hidden-size', hidden_size, '--lstm-layers', layers, '--cuda-able', '--sum_dir',
# 						      './sum/2', file=f)
# 	for lr in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001):
# 		for dropout in (0.5, 0.7, 1.):
# 			for embdim in (32, 64, 96, 128):
# 				for hidden_size in (64, 96, 128, 256):
# 					for layers in (2, 3, 4, 5):
# 						print("python ./main.py --lr", lr, "--epochs", 100, '--dropout', dropout, '--embed-dim', embdim,
# 						      '--hidden-size', hidden_size, '--lstm-layers', layers, '--cuda-able', '--bidirectional',
# 						      '--sum_dir', './save/2', file=f)
with open("train.sh", "w+") as f:
	for lr in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001):
		print("python ./main.py --pickle --lr", lr, '--sum_dir', './save/0606/lr', "--cuda-able ", "--save_dir",
		      "./save/0606/lr/" + str(lr), file=f)
	# exit()
	lr = 0.0005
	for dropout in (0.5, 0.7, 0.9, 1.):
		print("python ./main.py --pickle --lr", lr, "--dropout", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/dropout/" + str(dropout), "--sum_dir", "./save/0606/dropout/", file=f)
		print("python ./main.py --pickle --lr", lr, "--dropout", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/dropout-b/" + str(dropout), "--sum_dir", "./save/0606/dropout/", "--bidirectional", file=f)

	for dropout in (32, 64, 96, 128):
		print("python ./main.py --pickle --lr", lr, "--embed-dim", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/embdim/" + str(dropout), "--sum_dir", "./save/0606/embdim/", file=f)
		print("python ./main.py --pickle --lr", lr, "--embed-dim", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/embdim-b/" + str(dropout), "--sum_dir", "./save/0606/embdim/", "--bidirectional", file=f)
	# exit()

	for dropout in (64, 96, 128, 256):
		print("python ./main.py --pickle --lr", lr, "--hidden-size", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/hidden_size/" + str(dropout), "--sum_dir", "./save/0606/hidden_size/", file=f)
		print("python ./main.py --pickle --lr", lr, "--hidden-size", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/hidden_size-b/" + str(dropout), "--sum_dir", "./save/0606/hidden_size/", "--bidirectional",
		      file=f)

	for dropout in (2, 3, 4, 5, 6):
		print("python ./main.py --pickle --lr", lr, "--lstm-layers", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/layers/" + str(dropout), "--sum_dir", "./save/0606/layers/", file=f)
		print("python ./main.py --pickle --lr", lr, "--lstm-layers", dropout, "--cuda-able", "--save_dir",
		      "./save/0606/layers-b/" + str(dropout), "--sum_dir", "./save/0606/layers/", "--bidirectional", file=f)
