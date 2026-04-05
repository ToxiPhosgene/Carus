import matplotlib.pyplot as plt
import numpy as np
import h5py

import library

# # Generate distributions
# gecis = ['jrgeco']
# mouse_ids = ['6841']
# recording_ids = [0, 1, 2]
sizelims = [1, 500]
durlims = [1, 50]

ks = list(range(1, 15))
xpt = 3

rows = 3
cols = 2
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)

colors = plt.cm.winter(np.linspace(0,1,len(ks)))

# original_arrays = []
# filename = "../data/deepIP/jrgeco-deepIP/8191-0.mat"
# with h5py.File(filename, 'r') as f:
#     data = f["data"][()][0]
#     l = [f[dat] for dat in data]
#     for recording_id in recording_ids:
#         array = l[recording_id]["probs"][()].T
#         original_arrays.append(array)

sizes_per_k = [[] for _ in range(len(ks))]
durations_per_k = [[] for _ in range(len(ks))]
for original_array in original_arrays:
    removed_array = library.get_removed_array(original_array)
    sizes_per_k_, durations_per_k_ = library.get_avalanches_per_k(removed_array, ks, bootstrap=False)
    sizes_per_k = [sizes_per_k[kid] + sizes_per_k_[kid] for kid in range(len(ks))]
    durations_per_k = [durations_per_k[kid] + durations_per_k_[kid] for kid in range(len(ks))]

slopes = []
high_slopes = []
size_slopes = []
duration_slopes = []
for kid, (sizes_k, durations_k) in enumerate(zip(sizes_per_k, durations_per_k)):
    xs, ys = library.get_scaling_xs_ys(durations_k, sizes_k)
    s_xs, s_ys = library.get_hist(sizes_k, binnum=15)
    d_xs, d_ys = library.get_hist(durations_k)
    slope_, xs_fit, ys_fit, err = library.get_scaling_exponent(xs, ys, xpt=xpt)
    high_slope_, _, _, _ = library.get_scaling_exponent(xs, ys, high=True)
    size_slope_, _, _, _ = library.get_pl_exponent(s_xs, s_ys, sizelims)
    duration_slope_, _, _, _ = library.get_pl_exponent(d_xs, d_ys, durlims)
    slopes.append(slope_)
    high_slopes.append(high_slope_)
    size_slopes.append(size_slope_)
    duration_slopes.append(duration_slope_)

for kkid, kid in enumerate(ks):
    s_xs, s_ys = library.get_hist(sizes_per_k[kkid], binnum=20)
    _, fit_xs, fit_ys, _ = library.get_pl_exponent(s_xs, s_ys, sizelims)
    axes[0, 0].plot(s_xs, s_ys, color=colors[kkid])
    axes[0, 0].plot(fit_xs, fit_ys, color="black", linestyle=":")

    d_xs, d_ys = library.get_hist(durations_per_k[kkid], binnum=15)
    _, fit_xs, fit_ys, _ = library.get_pl_exponent(d_xs, d_ys, durlims)
    axes[1, 0].plot(d_xs, np.array(d_ys), color=colors[kkid])
    axes[1, 0].plot(fit_xs, np.array(fit_ys), color="black", linestyle=":")

    xs, ys = library.get_scaling_xs_ys(durations_per_k[kkid], sizes_per_k[kkid])
    slope_, xs_fit, ys_fit, err = library.get_scaling_exponent(xs, ys)
    axes[2, 0].plot(xs, ys, color=colors[kkid])
    axes[2, 0].scatter(xs, ys, color=colors[kkid], s=1.5)
    axes[2, 0].plot(xs_fit, ys_fit, color="black", linestyle=":")

axes[2, 1].plot(ks, slopes)
axes[0, 1].plot(ks, np.abs(size_slopes))
axes[1, 1].plot(ks, np.abs(duration_slopes))
axes[2, 1].plot(ks, slopes)
axes[2, 1].plot(ks, high_slopes)

axes[0, 1].set_xlabel("k")
axes[0, 1].set_ylabel("exponent")
axes[1, 1].set_xlabel("k")
axes[1, 1].set_ylabel("exponent")

axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Sizes')
axes[0, 0].set_ylabel('PDF')

axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].set_xlabel('Duration')
axes[1, 0].set_ylabel('PDF')

axes[2, 0].set_xscale('log')
axes[2, 0].set_yscale('log')
axes[2, 0].set_xlabel('Duration')
axes[2, 0].set_ylabel('Mean size')
xs_ref = [1, 5]
if len(ys) > 0:
    ys_ref = ys[0]*np.power(xs_ref[0], -2.0)*np.power(xs_ref, 2.0)
axes[2, 0].plot(xs_ref, ys_ref, color="black", linewidth=1.5)

axes[2, 1].set_xlabel('k')
axes[2, 1].set_ylabel('$\gamma$')
axes[2, 1].set_ylim([0.9, 2.5])
axes[2, 1].axhline(y=1.0, color="black", linestyle=":")
axes[2, 1].axhline(y=2.0, color="red", linestyle=":")

plt.tight_layout()
plt.savefig("distributions.pdf", format="pdf", dpi=500)
