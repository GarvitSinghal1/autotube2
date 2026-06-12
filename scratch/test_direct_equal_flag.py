import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.modules.renderer_short import _add_flag_to_axes

fig, ax = plt.subplots(figsize=(10.8, 19.2), dpi=100)
ax.set_xlim(0, 100)
ax.set_ylim(0, 138.27)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

val = _add_flag_to_axes(ax, "India", 50.0, 69.13, box_alignment=(0.5, 0.5))
print(f"Direct equal-aspect flag returned: {val}")
fig.savefig("tmp/direct_equal_flag.png", facecolor="#000000")
print("Saved to tmp/direct_equal_flag.png")
