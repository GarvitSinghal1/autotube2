import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.modules.renderer_short import _add_flag_to_axes

fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
val = _add_flag_to_axes(ax, "India", 0.5, 0.5, box_alignment=(0.5, 0.5))
print(f"Direct flag returned: {val}")
fig.savefig("tmp/direct_flag_test.png")
print("Saved direct flag test to tmp/direct_flag_test.png")
