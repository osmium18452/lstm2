import matplotlib.pyplot as plt


a=[]
lr=[]
with open("../sum.txt", "r") as f:
	for line in f.readlines():
		wordlist=line.strip().split()
		a.append(float(wordlist[-1]))
		lr.append(wordlist[0])

a.reverse()
lr.reverse()
plt.figure()
x=range(len(a))
# plt.axis('scaled')
plt.xticks(x,lr)
plt.xlabel("Learning Rate")
plt.ylabel("Final Accuracy (%)")
plt.title("Final Accuracy")
kwargs = {
			"marker": "o",
			"lw":2
		}
plt.plot(x, a, label="SGD train accuracy", **kwargs)
# plt.legend()
sv = plt.gcf()
sv.savefig("./save/accuracy.png", format="png", dpi=100)
plt.show()