"""
BRMSD — Maximum-Entropy RMSD Structural Alignment

Modules
-------
core    : shared utilities (Kabsch rotation, trajectory I/O, logging)
align   : average structure + optimal atom weights
smooth  : trajectory smoothing
cluster : clustering (cluster-center finding and soft K-means)
domain  : rigid domain identification
focus   : focused alignment biased toward a specific domain
"""

from .core    import (compute_weighted_rotation, align_frame, load_frames,
                      compute_rotations_per_frame, write_rotation_data)
from .align   import BRMSD_iterate
from .smooth  import BRMSD_smooth
from .cluster import soft_kmeans
from .domain  import identify_domains, identify_domains_sequential
from .focus   import BRMSD_focus_iterate
