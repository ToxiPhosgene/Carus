# from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import copy
# import h5py
import hashlib
import itertools
import matplotlib
import matplotlib.pyplot as plt
# import mat73
import numpy as np
import os
import pandas as pd
import pickle
import scipy.optimize
import scipy.stats
import scipy.special
import scipy.io
import sys
import powerlaw

np.random.seed(0)

#########
# Mouse stuff
#########


# def get_data(gecis, mouse_ids, recording_idss, allen_locomotion=False, data_root=""):
#     assert(all(geci in ["allen", "allen-movie", "jrgeco", "gcamp7s", "tiago"]) for geci in gecis)
#
#     shuffle = "shuffle" in sys.argv
#
#     arrays = []
#     locomotions = []
#     bounds_traces = []
#
#     for (geci, mouse_id, recording_ids) in zip(gecis, mouse_ids, recording_idss):
#         if geci == "allen" or geci == "allen-movie":
#             boc = BrainObservatoryCache(manifest_file=data_root+'downloads/manifest.json')
#
#             if geci == "allen":
#                 stim_code = "drifting_gratings"
#                 session_code = "three_session_A"
#             elif geci == "allen-movie":
#                 stim_code = "natural_movie_three"
#                 session_code = "three_session_A"
#
#             exp_containers = boc.get_experiment_containers(cre_lines=["Cux2-CreERT2"], targeted_structures=["VISp"], imaging_depths=[275])
#             exp_container_ids = [container['id'] for container in exp_containers]
#
#             experiments = boc.get_ophys_experiments(experiment_container_ids=exp_container_ids, stimuli=[stim_code], session_types=[session_code])
#
#             for recording_id in recording_ids:
#                 eid = experiments[recording_id]['id']
#
#                 trace = boc.get_ophys_experiment_events(ophys_experiment_id=eid)
#
#                 data = boc.get_ophys_experiment_data(eid)
#                 stim_table = data.get_stimulus_table(stim_code)
#
#                 starts = list(stim_table.start)
#                 ends = list(stim_table.end)
#                 assert(len(starts) == len(ends))
#
#                 bounds = []
#                 low = starts[0]
#                 for k in range(len(starts)-1):
#                     if starts[k+1] > ends[k] + 1000:
#                         bounds.append((low, ends[k]))
#                         low = starts[k+1]
#                 bounds.append((low, ends[-1]))
#
#                 data_set = boc.get_ophys_experiment_data(eid)
#                 dxcm, dxtime = data_set.get_running_speed()
#
#                 bounded_locomotion = np.concatenate([dxcm[bound[0]:bound[1]] for bound in bounds], axis=0)
#
#                 bounds_trace = np.zeros_like(bounded_locomotion)
#                 for (st, nd) in zip(stim_table.start, stim_table.end):
#                     bounds_trace[st:nd] = 1.0
#
#                 if shuffle and geci == "allen":
#                     new_trace = np.ones_like(trace) * np.inf
#                     if geci == "allen":
#                         starts = stim_table["start"]
#                         ends = stim_table["end"]
#                         temporal_frequencies = stim_table["temporal_frequency"]
#                         orientations = stim_table["orientation"]
#                         blank_sweep = stim_table["blank_sweep"]
#                         stimsets = [[temporal_frequencies[i], orientations[i], blank_sweep[i]] if blank_sweep[i] == 0.0 else [-1, -1, blank_sweep[i]] for i in range(len(orientations))]
#                         unique_stimuli = np.unique(stimsets, axis=0)
#                         unique_stimuli_presentations = []
#                         for unique_stim in unique_stimuli:
#                             unique_stim_indices = np.nonzero([temporal_frequencies[i] == unique_stim[0] and orientations[i] == unique_stim[1] if blank_sweep[i] == 0.0 else blank_sweep[i] == 1.0 for i in range(len(stim_table.start))])[0]
#                             unique_stimuli_presentations.append([list(np.array(starts)[unique_stim_indices]), list(np.array(ends)[unique_stim_indices])])
#                         for i in range(len(stim_table.start)):
#                             start = stim_table.start[i]
#                             end = stim_table.end[i]
#                             frequency = stim_table.temporal_frequency[i]
#                             orientation = stim_table.orientation[i]
#                             blank_sweep = stim_table.blank_sweep[i]
#                             assert(any([start >= bound[0] and start < bound[1] for bound in bounds]))
#                             unique_stim_id = np.nonzero([(blank_sweep == 1.0 and ustim[2] == 1.0) or (frequency == ustim[0] and orientation == ustim[1]) for ustim in unique_stimuli])[0]
#                             assert(len(unique_stim_id)==1)
#                             other_unique_starts = unique_stimuli_presentations[unique_stim_id[0]][0]
#                             other_unique_ends = unique_stimuli_presentations[unique_stim_id[0]][1]
#                             for cell in range(len(trace)):
#                                 old_presentation_idx = np.random.choice(range(len(other_unique_starts)))
#                                 old_start = other_unique_starts[old_presentation_idx]
#                                 new_trace[cell, start:start+59+30+1] = trace[cell, old_start:old_start+59+30+1]
#
#                         trace = new_trace
#                     else:
#                         print("Error, can't trial shuffle this stimulus.")
#                         sys.exit(1)
#
#                 bounded_trace = np.concatenate([trace[:, bound[0]:bound[1]] for bound in bounds], axis=1)
#
#                 assert(len(bounded_trace) == len(trace))
#
#                 if shuffle:
#                     if geci == "allen-movie":
#                         for aid, arr in enumerate(bounded_trace):
#                             piv = np.random.choice(list(range(len(arr))))
#                             arr = np.concatenate([arr[piv:], arr[:piv]])
#                             bounded_trace[aid] = arr
#                     if geci == "allen":
#                         nidxs = np.nonzero([any([val==np.inf for val in bounded_trace[:, i]]) for i in range(len(bounded_trace[0]))])[0]
#                         non_nidxs = [i for i in range(len(bounded_trace[0])) if i not in nidxs]
#                         bounded_trace = bounded_trace.T[non_nidxs].T
#
#                 arrays.append(bounded_trace)
#                 locomotions.append(bounded_locomotion)
#                 bounds_traces.append(bounds_trace)
#
#         elif geci == "jrgeco":
#             filename = data_root+"downloads/Data-From-Patrick/spikes/jrgeco-march2021/"
#             if mouse_id == "3963":
#                 filename += "0013963_spikes_0.7npsub_denoised_210325.mat"
#             elif mouse_id == "6841":
#                 filename += "5316841_spikes_0.7npsub_denoised_210325.mat"
#             elif mouse_id == "8185":
#                 filename += "5358185_spikes_0.7npsub_denoised_210325.mat"
#             elif mouse_id == "8191":
#                 filename += "5358191_spikes_0.7npsub_denoised_210325.mat"
#             elif mouse_id == "8193":
#                 filename += "5358193_spikes_0.7npsub_denoised_210325.mat"
#             elif mouse_id == "3976":
#                 filename += "denoised_popvecs_0013976.mat"
#             elif mouse_id == "3979":
#                 filename += "denoised_popvecs_0013979.mat"
#             elif mouse_id == "3985":
#                 filename += "denoised_popvecs_0013985.mat"
#
#             with h5py.File(filename, 'r') as f:
#                 if mouse_id in ["3976", "3979", "3985"]:
#                     data = f["popvecs"][()][0]
#                 else:
#                     data = f["data"][()][0]
#                 l = [f[dat] for dat in data]
#                 for recording_id in recording_ids:
#                     if mouse_id in ["3976", "3979", "3985"]:
#                         array = l[recording_id]["popvec"][()].T
#                     else:
#                         array = l[recording_id]["probs"][()].T
#                     if shuffle:
#                         for aid, arr in enumerate(array):
#                             piv = np.random.choice(list(range(len(arr))))
#                             arr = np.concatenate([arr[piv:], arr[:piv]])
#                             array[aid] = arr
#                     arrays.append(array)
#
#         elif geci == "jrgeco-raw":
#             filename = "downloads/Data-From-Patrick/spikes/rgeco-mar19-pparams-npsub/"
#             if mouse_id == "3963-raw":
#                 filename += "0013963_spikes_0.7npsub_201014_1026.mat"
#             elif mouse_id == "6841-raw":
#                 filename += "5316841_spikes_0.7npsub_201014.mat"
#             elif mouse_id == "8185-raw":
#                 filename += "5358185_spikes_0.7npsub_201014.mat"
#             elif mouse_id == "8191-raw":
#                 filename += "5358191_spikes_0.7npsub_201014.mat"
#             elif mouse_id == "8193-raw":
#                 filename += "5358193_spikes_0.7npsub_201014.mat"
#
#             with h5py.File(filename, 'r') as f:
#                 data = f["data"][()][0]
#                 l = [f[dat] for dat in data]
#                 for recording_id in recording_ids:
#                     array = l[recording_id]["probs"][()].T
#                     if shuffle:
#                         for aid, arr in enumerate(array):
#                             piv = np.random.choice(list(range(len(arr))))
#                             arr = np.concatenate([arr[piv:], arr[:piv]])
#                             array[aid] = arr
#                     arrays.append(array)
#
#         elif geci == "gcamp7s":
#             filename = "downloads/Data-From-Patrick/spikes/gcamp-april19-pparams-npsub-newroi/"
#             if mouse_id == "4835":
#                 filename += "5314835_spikes_0.7npsub_201014.mat"
#             elif mouse_id == "4843":
#                 filename += "5314843_spikes_0.7npsub_201014.mat"
#             elif mouse_id == "5592":
#                 filename += "5275592_spikes_0.7npsub_201014.mat"
#
#             with h5py.File(filename, 'r') as f:
#                 data = f["data"][()][0]
#                 l = [f[dat] for dat in data]
#                 for recording_id in recording_ids:
#                     array = l[recording_id]["probs"][()].T
#                     if "short" in sys.argv:
#                         array = array[:, :1000]
#                     if shuffle:
#                         for aid, arr in enumerate(array):
#                             piv = np.random.choice(list(range(len(arr))))
#                             arr = np.concatenate([arr[piv:], arr[:piv]])
#                             array[aid] = arr
#                     arrays.append(array)
#
#         elif geci == "tiago":
#             for recording_id in recording_ids:
#                 filename = "downloads/Data-From-Patrick/spikes/tiago-10-21/"
#                 if recording_id == "1":
#                     filename += "normdata_3143_1_denoised.mat"
#                 elif recording_id == "2":
#                     filename += "normdata_3143_2_denoised.mat"
#                 elif recording_id == "3":
#                     filename += "normdata_6742_denoised.mat"
#
#                 with h5py.File(filename, 'r') as f:
#                     array = f["spikes"][()].T
#                     if shuffle:
#                         visualstimulus = f["visualstimulus"][()][0]
#                         starts = np.nonzero(np.bitwise_and(visualstimulus[:-1]==1, visualstimulus[1:]==0))[0]
#                         periods = np.concatenate([[(starts[s], starts[s+1]) for s in range(len(starts)-1)], [(starts[-1], len(visualstimulus))]])
#                         shuffled_array = []
#                         for a in range(len(array)):
#                             old_arr = array[a]
#                             new_periods = np.random.permutation(periods)
#                             new_arr = np.concatenate([old_arr[p[0]:p[1]] for p in new_periods])
#                             shuffled_array.append(new_arr)
#                         array = np.array(shuffled_array)
#                     arrays.append(array)
#
#                     bounds = f["visualstimulus"][()]
#                     bounds_traces.append(bounds)
#
#         elif geci == "tiago-raw":
#             for recording_id in recording_ids:
#                 filename = "downloads/Data-From-Patrick/spikes/tiago-10-21/"
#                 if recording_id == "1":
#                     filename += "normdata_3143_10.16.20_VisOnly.mat"
#                 elif recording_id == "2":
#                     filename += "normdata_3143_10.16.20_VisOnly2.mat"
#                 elif recording_id == "3":
#                     filename += "normdata_6742_10.14.20_VisOnly.mat"
#
#                 with h5py.File(filename, 'r') as f:
#                     array = f["spikes"][()].T
#                     if shuffle:
#                         visualstimulus = f["visualstimulus"][()][0]
#                         starts = np.nonzero(np.bitwise_and(visualstimulus[:-1]==1, visualstimulus[1:]==0))[0]
#                         periods = np.concatenate([[(starts[s], starts[s+1]) for s in range(len(starts)-1)], [(starts[-1], len(visualstimulus))]])
#                         shuffled_array = []
#                         for a in range(len(array)):
#                             old_arr = array[a]
#                             new_periods = np.random.permutation(periods)
#                             new_arr = np.concatenate([old_arr[p[0]:p[1]] for p in new_periods])
#                             shuffled_array.append(new_arr)
#                         array = np.array(shuffled_array)
#                     arrays.append(array)
#
#         elif geci == "keshav" or geci == "keshav-closer" or geci == "keshav-subcritical":
#             if "100" in sys.argv:
#                 vname = "pop_vec_full"
#             elif "0.01" in sys.argv:
#                 vname = "pop_vec_sub_001"
#             elif "1" in sys.argv:
#                 vname = "pop_vec_sub_1"
#             elif "10" in sys.argv:
#                 vname = "pop_vec_sub_10"
#             else:
#                 vname = "pop_vec_sub_01"
#
#             if "old" in sys.argv:
#                 N = 1000000
#             else:
#                 N = 100000000
#             if geci == "keshav-closer":
#                 mat = scipy.io.loadmat("downloads/Data-From-Patrick/spikes/keshav-10-20-20/closer_data.mat")
#             elif geci == "keshav-subcritical":
#                 mat = scipy.io.loadmat("downloads/Data-From-Patrick/spikes/keshav-10-20-20/sub_critical_data.mat")
#             else:
#                 mat = scipy.io.loadmat("downloads/Data-From-Patrick/spikes/keshav-10-20-20/data_20210131.mat")
#             data = mat[vname][0][:N]
#             arrays.append([data])
#
#     arrays = [np.ascontiguousarray(arr) for arr in arrays]
#     arrays = [np.nan_to_num(oa) for oa in arrays]
#
#     if "noise" in sys.argv or "subsample" in sys.argv or "drop" in sys.argv or "dante" in sys.argv:
#         if "n0" in sys.argv:
#             n = 0.0
#         elif "n1" in sys.argv:
#             n = 0.01
#         elif "n2" in sys.argv:
#             n = 0.02
#         elif "n3" in sys.argv:
#             n = 0.03
#         elif "n4" in sys.argv:
#             n = 0.04
#         elif "n5" in sys.argv:
#             n = 0.05
#         elif "n6" in sys.argv:
#             n = 0.06
#         elif "n7" in sys.argv:
#             n = 0.07
#         elif "n8" in sys.argv:
#             n = 0.08
#         elif "n9" in sys.argv:
#             n = 0.09
#         elif "n10" in sys.argv:
#             n = 0.10
#         elif "n20" in sys.argv:
#             n = 0.20
#         elif "n30" in sys.argv:
#             n = 0.30
#         elif "n40" in sys.argv:
#             n = 0.40
#         elif "n50" in sys.argv:
#             n = 0.50
#         elif "n60" in sys.argv:
#             n = 0.60
#         elif "n70" in sys.argv:
#             n = 0.70
#         elif "n80" in sys.argv:
#             n = 0.80
#         elif "n90" in sys.argv:
#             n = 0.90
#         elif "n15" in sys.argv:
#             n = 0.15
#         elif "n25" in sys.argv:
#             n = 0.25
#         elif "n35" in sys.argv:
#             n = 0.35
#         elif "n45" in sys.argv:
#             n = 0.45
#         elif "n55" in sys.argv:
#             n = 0.55
#         elif "n65" in sys.argv:
#             n = 0.65
#         elif "n75" in sys.argv:
#             n = 0.75
#         elif "n85" in sys.argv:
#             n = 0.85
#         elif "n95" in sys.argv:
#             n = 0.95
#         elif "n100" in sys.argv:
#             n = 1.0
#         elif "n150" in sys.argv:
#             n = 1.5
#         elif "n200" in sys.argv:
#             n = 2.0
#         elif "n500" in sys.argv:
#             n = 5.0
#         elif "n91" in sys.argv:
#             n = 0.91
#         elif "n92" in sys.argv:
#             n = 0.92
#         elif "n93" in sys.argv:
#             n = 0.93
#         elif "n94" in sys.argv:
#             n = 0.94
#         elif "n96" in sys.argv:
#             n = 0.96
#         elif "n97" in sys.argv:
#             n = 0.97
#         elif "n98" in sys.argv:
#             n = 0.98
#         elif "n99" in sys.argv:
#             n = 0.99
#         if "noise" in sys.argv:
#             new_arrays = []
#             num_samples = 5
#             for original_array in arrays:
#                 population_rate = np.sum(original_array > 0.0) / len(original_array[0])
#                 cell_rate = population_rate / len(original_array)
#                 p_cell = cell_rate * n
#                 for _ in range(num_samples):
#                     new_array = np.ones_like(original_array) * np.nan
#                     for aid, arr in enumerate(original_array):
#                         pvec = np.random.rand(len(arr)) < p_cell
#                         rvec = np.random.choice(arr, pvec.size)
#                         cell_noise = np.where(pvec, rvec, np.zeros_like(arr))
#                         new_array[aid] = arr + cell_noise
#                     new_arrays.append(new_array)
#             arrays = new_arrays
#         elif "subsample" in sys.argv:
#             new_arrays = []
#             num_samples = 5
#             for oid, original_array in enumerate(arrays):
#                 for _ in range(num_samples):
#                     new_array = np.ones_like(original_array) * np.nan
#                     for aid, arr in enumerate(original_array):
#                         pvec = np.random.rand(len(arr)) < n
#                         oa = np.where(pvec, np.zeros_like(arr), arr)
#                         new_array[aid] = oa
#                     new_arrays.append(new_array)
#             arrays = new_arrays
#         elif "drop" in sys.argv:
#             new_arrays = []
#             if "ns5" in sys.argv:
#                 num_samples = 5
#             elif "ns2" in sys.argv:
#                 num_samples = 2
#             elif "ns10" in sys.argv:
#                 num_samples = 10
#             else:
#                 num_samples = 20
#             for oid, original_array in enumerate(arrays):
#                 for _ in range(num_samples):
#                     random_idxs = np.random.choice(len(original_array), int(len(original_array)*n), replace=False)
#                     new_array = original_array[random_idxs]
#                     new_arrays.append(new_array)
#             arrays = new_arrays
#         elif "dante" in sys.argv:
#             new_arrays = []
#             if "ns5" in sys.argv:
#                 num_samples = 5
#             elif "ns10" in sys.argv:
#                 num_samples = 10
#             else:
#                 num_samples = 20
#             for oid, original_array in enumerate(arrays):
#                 for _ in range(num_samples):
#                     new_array = np.ones_like(original_array)*np.nan
#                     n_shift = int(len(original_array)*(1-n))
#                     n_unshift = len(original_array) - n_shift
#                     random_idxs = np.random.choice(len(original_array), len(original_array), replace=False)
#                     pivs = np.random.choice(list(range(len(original_array[0]))), size=n_shift)
#                     new_array[:n_unshift] = original_array[random_idxs[:n_unshift]]
#                     for ii in range(n_unshift, len(original_array)):
#                         arr = original_array[random_idxs[ii-n_unshift]]
#                         shifted_array = np.concatenate([arr[pivs[ii-n_unshift]:], arr[:pivs[ii-n_unshift]]])
#                         new_array[ii] = shifted_array
#                     new_arrays.append(new_array)
#             arrays = new_arrays
#
#     if allen_locomotion:
#         return arrays, locomotions, bounds_traces
#     return arrays


def get_threshold(array, dt, z=np.nan):
    if np.isnan(z):
        z = -2.0

    thetas = np.linspace(0.0, 1.0, 500)
    thetas = np.linspace(0.0, 1.0, 5)
    thresholds = np.log10(np.power(10.0, thetas*5.5-2.5)*np.power(dt, 0.5))
    clusters_per_threshold = []
    for th in thresholds:
        thresh = np.power(10.0, th)
        thresholded_array = np.where(array>thresh, array, np.zeros_like(array))

        num = 0

        padded_arrays = get_coarse_grained_arrays(thresholded_array, dt)

        for arr in padded_arrays:
            ava_sizes, _ = get_avalanches(arr)
            num += len(ava_sizes)

        clusters_per_threshold.append(float(num) / len(padded_arrays))

    def gaussian(x, u, s, A):
        return A * np.exp(-1.0 * np.power(x-u, 2.0) / s / s / 2.0)

    popt, _ = scipy.optimize.curve_fit(gaussian, thresholds, clusters_per_threshold)

    g_mean = popt[0]
    g_std = np.abs(popt[1])

    return np.power(10.0, g_mean + z*g_std)


def get_removed_array(array):
    """
    Просто удаляем неактивные нейроны
    array: (нейроны, время) с бинарными спайками (0 или 1)
    """
    active_neurons = np.where(np.sum(array, axis=1) > 0)[0]
    cleaned_array = array[active_neurons]

    return cleaned_array


def apply_threshold(array, dt, z=np.nan):
    popvec = np.sum(array, axis=0)
    threshold = get_threshold(popvec, dt, z=z)
    thresholded_array = np.where(popvec>threshold, popvec, np.zeros_like(popvec))
    return thresholded_array


def get_coarse_grained_arrays(array, dt):
    arrays = []
    for pad in range(0, dt):
        padded_trace = copy.copy(array)[pad:]
        padded_trace = np.array(padded_trace[:len(padded_trace)//dt*dt])
        assert(len(padded_trace)/dt % 1.0 == 0.0)
        reshaped_trace = padded_trace.reshape(int(len(padded_trace)/dt), dt)
        th_array = np.sum(reshaped_trace, axis=1)
        arrays.append(th_array)
    return arrays


def get_avalanches(array):
    descending = np.bitwise_and(array[1:]==0, array[:-1]>0)
    ascending = np.bitwise_and(array[1:]>0, array[:-1]==0)
    descending_idxs = np.nonzero(descending)[0]+1
    ascending_idxs = np.nonzero(ascending)[0]
    if array[0]>0:
        ascending_idxs = np.concatenate([[0], ascending_idxs])
    if array[-1]>0:
        descending_idxs = np.concatenate([descending_idxs, [len(array)-1]])
    sizes = [np.sum(array[ascending_idxs[j]:descending_idxs[j]]) for j in range(len(descending_idxs))]
    durations = [descending_idxs[j]-ascending_idxs[j]-1 for j in range(len(ascending_idxs))]
    if array[0]>0:
        durations[0] += 1
    if array[-1]>0:
        durations[-1] += 1
        sizes[-1] += array[-1]

    if "synthetic" in sys.argv:
        synthetic_sizes = []
        synthetic_durations = []

        sizemin = min(sizes)
        sizemax = max(sizes)
        durmin = min(durations)
        durmax = max(durations)

        pdf_exp = 1.5
        pmf_exp = 2.0

        pdf_norm = (np.power(sizemax, 1-pdf_exp) / (1-pdf_exp)) - (np.power(sizemin, 1-pdf_exp) / (1-pdf_exp))
        pmf_norm = (np.sum([np.power(i, -pmf_exp) for i in range(durmin, durmax)]))

        def pl_pdf(x):
            pdf = np.power(x, -pdf_exp) / pdf_norm
            return pdf

        def pl_pmf(x):
            pmf = np.power(x, -pmf_exp) / pmf_norm
            return pmf

        while len(synthetic_sizes) < len(sizes):
            scand = np.random.rand()*(sizemax-sizemin) + sizemin
            if pl_pdf(scand) > np.random.rand():
                synthetic_sizes.append(scand)

        while len(synthetic_durations) < len(durations):
            dcand = np.random.randint(durmin, durmax+1)
            if pl_pmf(dcand) > np.random.rand():
                synthetic_durations.append(dcand)

        sizes = synthetic_sizes
        durations = synthetic_durations

    return sizes, durations


def get_avalanches_per_k(array, ks, bootstrap=False):
    sizes = []
    durations = []
    for k in ks:
        thresholded_array = apply_threshold(copy.copy(array), k)
        padded_arrays = get_coarse_grained_arrays(thresholded_array, k)

        sizes_ = []
        durations_ = []
        for arr in padded_arrays:
            ava_sizes, ava_durations = get_avalanches(arr)

            if bootstrap:
                ava_idxs = np.random.choice(list(range(len(ava_sizes))), size=len(ava_sizes))
                ava_sizes = [ava_sizes[s] for s in ava_idxs]
                ava_durations = [ava_durations[s] for s in ava_idxs]

            sizes_ = sizes_ + ava_sizes
            durations_ = durations_ + ava_durations

        sizes.append(sizes_)
        durations.append(durations_)

    return sizes, durations

def get_hist(array, linmode=False, binnum=20):
    if len(array) == 0:
        return np.array([]), np.array([])

    if linmode:
        bins = np.linspace(np.min(array)*0.99, np.max(array)*1.01, binnum+1)
    else:
        bins = np.geomspace(np.min(array)*0.99, np.max(array)*1.01, binnum+1)
    counts, binedges = np.histogram(array, density=True, bins=bins)
    bincenters = (binedges[1:] + binedges[:-1]) / 2.0
    valid_bins = counts > 0.0
    valid_counts = counts[valid_bins]
    valid_binedges = bincenters[valid_bins]
    return valid_binedges, valid_counts


def get_scaling_xs_ys(xarray, yarray, bins=[]):
    binnum = 50
    mincounts = 1

    if len(xarray) == 0:
        return np.array([]), np.array([])

    xarray = np.array(xarray)
    yarray = np.array(yarray)


    if bins == []:
        xbins = np.geomspace(np.min(xarray), np.max(xarray)*1.01, binnum+1)
        xbins = [int(s) for s in xbins]
        xbins = list(np.unique(xbins))
    else:
        xbins = bins

    xidxs = np.argsort(xarray)

    yarray = yarray[xidxs]
    xarray = xarray[xidxs]
    xbounds = [np.searchsorted(xarray, v, side="left") for v in xbins]
    ybinned = [yarray[xbounds[i]:xbounds[i+1]] for i in range(len(xbounds)-1)]

    valid_ymeans = [np.mean(ybinned[i]) if len(ybinned[i]) > 0 else np.nan for i in range(len(xbounds)-1)]
    return np.array(xbins[:-1]), np.array(valid_ymeans)

def get_scaling_exponent(xs, ys, rng=[], high=False, xpt=3):
    def linear(x, y0, k):
        return y0 + k*x

    if high == "allen":
        fit_x = np.log10(xs[np.bitwise_and(xs<100, xs>30)]).astype("float")
        fit_y = np.log10(ys[np.bitwise_and(xs<100, xs>30)]).astype("float")
    elif high == "nonallen":
        fit_x = np.log10(xs[np.bitwise_and(xs<100, xs>10)]).astype("float")
        fit_y = np.log10(ys[np.bitwise_and(xs<100, xs>10)]).astype("float")
    else:
        fit_x = np.log10(xs[xs<xpt]).astype("float")
        fit_y = np.log10(ys[xs<xpt]).astype("float")

    assert(len(fit_x) == len(fit_y))

    if len(fit_x) < 2:
        # print("Not enough avalanche durations, not returning scaling fit.")
        return 0.0, np.power(10.0, fit_x), np.zeros_like(fit_x), np.nan
    else:
        try:
            popt_per_x0 = []
            fit_ys_per_x0 = []
            err_per_x0 = []

            bounds = ([-np.inf, 0.0], [np.inf, 3.0])
            p0 = [1.0, 1.0]
            popt_, _ = scipy.optimize.curve_fit(linear, fit_x, fit_y, maxfev=10000, bounds=bounds, p0=p0)
            fit_y_res_ = linear(fit_x, popt_[0], popt_[1])
            err_ = np.sqrt(np.mean(np.power(fit_y_res_-fit_y, 2.0)))
            popt_per_x0.append(popt_)
            fit_ys_per_x0.append(fit_y_res_)
            err_per_x0.append(err_)

            minidx = np.argsort(err_per_x0)[0]
            popt = popt_per_x0[minidx]
            fit_y_res = fit_ys_per_x0[minidx]
            err = err_per_x0[minidx]

        except Exception as e:
            # print("Curve fitting didn't converge; not fitting.")
            return 0.0, np.power(10.0, fit_x), np.zeros_like(fit_x), np.nan

    return popt[1], np.power(10.0, fit_x), np.power(10.0, fit_y_res), err


def get_pl_exponent(data, discrete=True):
    # Убираем нули и NaN
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = data[data > 0]

    if len(data) < 10:
        # print("Недостаточно данных")
        return np.nan, np.nan, np.nan, np.nan

    try:
        # Подгоняем степенной закон
        if discrete:
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
        else:
            fit = powerlaw.Fit(data, discrete=False, verbose=False)

        # Получаем результаты
        alpha = fit.power_law.alpha
        xmin_fit = fit.power_law.xmin
        D = fit.power_law.D
        p = fit.power_law.sigma

        return alpha, xmin_fit, D, p

    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan
