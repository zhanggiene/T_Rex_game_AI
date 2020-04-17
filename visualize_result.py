import matplotlib.pyplot as plt
import pickle

with open("reward_history950", "rb") as fp:
    ys= pickle.load(fp)
max=max(ys)
for i in range(len(ys)):
    if ys[i]==max:
        ys[i]=ys[i+1]


xs = [x for x in range(len(ys))]

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title('episode and reward')
ax.set_xlabel('episode')
ax.set_ylabel('total reward')

plt.plot(xs, ys)
plt.show()
