import mahotas
import numpy as np
import logging
import waterz
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter, maximum_filter

logger = logging.getLogger(__name__)

def apply_watershed(affs, fragments_in_xy=True):
    fragments, n = __watershed_from_affinities(affs,
                                               fragments_in_xy=fragments_in_xy,
                                               return_seeds=False,
                                               epsilon_agglomerate=0.0)
    logger.info('Number of fragments in volume: {}'.format(len(np.unique(fragments))))
    return fragments


def agglomerate_fragments(affs, fragments, threshold,
                          scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>'):
    # waterz agglomerate changes fragments -->
    # Make a copy here

    fragments = fragments.copy()
    generator = waterz.agglomerate(affs, thresholds=[threshold],
                                   fragments=fragments,
                                   scoring_function=scoring_function)
    # segs = []
    for seg in generator:
        # segs.append(seg)
        logger.info('Number of segments in volume: {}'.format(
            len(np.unique(seg))))
    return seg



def __watershed_from_affinities(
        affs,
        fragments_in_xy=False,
        return_seeds=False,
        epsilon_agglomerate=0):
    '''Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.'''

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs = affs.astype(np.float32)
    else:
        max_affinity_value = 1.0

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = __watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset)

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = __watershed_from_boundary_distance(
            boundary_distances,
            return_seeds)

    if epsilon_agglomerate > 0:

        logger.info(
            "Performing initial fragment agglomeration until %f",
            epsilon_agglomerate)

        generator = waterz.agglomerate(
            affs=affs / max_affinity_value,
            thresholds=[epsilon_agglomerate],
            fragments=fragments,
            scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False)
        fragments[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    return ret


def __watershed_from_boundary_distance(
        boundary_distances,
        return_seeds=False,
        id_offset=0):
    max_filtered = maximum_filter(boundary_distances, 10)
    maxima = max_filtered == boundary_distances
    seeds, n = mahotas.label(maxima)

    logger.debug("Found %d fragments", n)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = mahotas.cwatershed(
        boundary_distances.max() - boundary_distances,
        seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret
