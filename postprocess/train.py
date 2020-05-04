import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

a=[]
list=('0.1','0.01','0.001','0.0001','0.02','0.002','0.0002','0.05','0.005','0.0005')
# list=('32','64','96','128')
# list=('64','96','128','256')
# list=('2','3','4','5','6')
for i in list:
	#---
	with open("../save/0606/lr/"+i+"/train_ac.pkl","rb") as f:
		a.append(pkl.load(f))

plt.figure()
x=range(len(a[0]))
# plt.xticks(x,lr)
plt.xlabel("epochs")
plt.ylabel("Final Accuracy (%)")
# ---
plt.title("Final Accuracy with Different Learning Rate")
kwargs = {
			"marker": None,
			"lw":2
		}

for i,y in enumerate(a):
	# ---
	plt.plot(x, y, label="lr="+list[i], **kwargs)

plt.legend()
sv = plt.gcf()
# ---
sv.savefig("../save/pic/lr.png", format="png", dpi=100)
plt.show()
