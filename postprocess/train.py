import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

a=[]
for i in ('0.5','0.7','0.9','1.0'):
	with open("../save/0505/dropout/"+i+"/train_ac.pkl","rb") as f:
		a.append(pkl.load(f))

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

for y in a:

