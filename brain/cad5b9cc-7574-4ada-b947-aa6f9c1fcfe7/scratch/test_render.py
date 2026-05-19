import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Call clf (detaches axes)
fig.clf()

# Recreate axes and attach to figure
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Draw on attached ax
ax.cla()
ax.plot([0, 1], [0, 1], color='red', linewidth=5)
ax.text(0.5, 0.5, "Hello", color='white', transform=fig.transFigure)

# Save
plt.savefig("test_fig2.png")
print("Saved fig.")
