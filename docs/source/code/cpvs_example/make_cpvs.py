from phyddle import utilities as util
import dendropy as dp
import numpy as np

print("Character matrix")
dat = util.convert_nexus_to_integer_array("dat.nex")
print(dat, "\n\n")

print("Serially sampled tree")
phy_cblvs = util.read_tree("cblvs.tre")
print(phy_cblvs, "\n")

print("CBLV+S tensor (unscaled)")
cblvs = util.encode_cblvs(phy_cblvs, dat, tree_width=10, tree_encode_type="height_brlen", rescale=False)
print(cblvs.T, "\n")

print("CBLV+S tensor (rescaled)")
cblvs_rescale = util.encode_cblvs(phy_cblvs, dat, tree_width=10, tree_encode_type="height_brlen", rescale=True)
cblvs_rescale = np.round(cblvs_rescale, 2).T
print(cblvs_rescale, "\n\n")

print("Extant-only tree")
phy_cdvs = util.read_tree("cdvs.tre")
print(phy_cdvs, "\n")

print("CDV+S tensor (unscaled)")
cdvs = util.encode_cdvs(phy_cdvs, dat, tree_width=10, tree_encode_type="height_brlen", rescale=False)
print(cdvs.T, "\n")

print("CDV+S tensor (rescaled)")
cdvs_rescale = util.encode_cdvs(phy_cdvs, dat, tree_width=10, tree_encode_type="height_brlen", rescale=True)
cdvs_rescale = np.round(cdvs_rescale, 2).T
print(cdvs_rescale, "\n")

