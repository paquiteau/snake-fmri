"""
Custom Phantom and ROI
======================

In this example we are going to show the different way to generate phantom data, using
the BIG phantom.
This phantom is more realistic than the Shepp-Logan Phantom, but is a 2D phantom.
"""

import matplotlib.pyplot as plt
from snkf.handlers.phantom._big import generate_phantom, raster_phantom

# %%
# Rasterisation vs Generation
# ---------------------------
# The phantom can be rasterize at any given dimension.
SHAPE = 128

rastered = raster_phantom(SHAPE)

generated = generate_phantom(SHAPE, anti_aliasing=False)
generated_aliased = generate_phantom(SHAPE, anti_aliasing=True)

# All the rest is plotting stuff.
fig = plt.figure()
gs = fig.add_gridspec(2, 3)
im_axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

cut_axs = fig.add_subplot(gs[1, :])

for ax, ph, name in zip(
    im_axs,
    [rastered, generated, generated_aliased],
    ["rastered", "generated", "anti-aliased"],
):
    ax.imshow(ph, cmap="gray")
    ax.axis("off")
    ax.set_title(name)

    cut_axs.plot(ph[SHAPE // 2], label=name)

inset_ax = cut_axs.inset_axes([-0.1, 1 - 0.4, 0.35, 0.35])
inset_ax.set_box_aspect(1)
inset_ax.imshow(ph, cmap="gray")
inset_ax.axis("off")
inset_ax.axhline(SHAPE // 2, ls="--", c="r")

cut_axs.legend(loc=8)
cut_axs.set_title("Cross section")
cut_axs.set_xlabel("readout dimension")


# %%
# Showing the ROI
# ---------------
#
# The BIG phantom comes with a tailored region of interest defined in the gray matter
# of the occipital cortex.

roi = raster_phantom(
    SHAPE,
    "big_roi",
    weighting="label",
)

fig, ax = plt.subplots()

ax.imshow(generated_aliased, cmap="gray")
ax.imshow(roi, alpha=1.0 * (roi > 0), cmap="jet", interpolation="none")

fig.show()
