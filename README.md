## COMP9517 Project

[Project Spec](https://webcms3.cse.unsw.edu.au/files/86f4ade7950342933e9b59cb4633c5033a01be6b4134b4f70442f4930bbdb267)
[Project Resources](https://webcms3.cse.unsw.edu.au/COMP9517/19T2/resources/26910)

[Watershed Paper](https://www.researchgate.net/publication/303719137_Watershed_Merge_Forest_Classification_for_Electron_Microscopy_Image_Stack_Segmentation)

### Watershed Method

- Train Random Forest Classifier with
    - stencils of intensity features
    - multi-scale Radon-like features
    - SIFT features
- Output per pixel will be probability of pixel being membrane (vs a cell)
- Create probability map using output
- Run watershed on probability map
- Create tree structure based on where regions of watershed merge (Nodes will be regions of watershed)
- Train boundary classifier on probability of merge between 2 regions occurring
    - input being features of the two merging regions
- Assign probability using boundary classifier
    - P = p_1 * (1 - p_2)
        - P is output potential
        - p_1 is probability of child nodes (the 2 regions that make up region in question) merging
        - p_2 is probability the child nodes merge with each other
- Train section classifier on probability of adject node being good reference
    - input being set of 59 features from potentially coresponded regions
        - geometric features (area, perimeter, compactness differences, centroid distance, overlapping)
        - image intensity statistic features (region and boundary pixel intensity statistics from both Gaussian denoised EM images and membrane detection maps)
        - textual features (texton statistics)
- Use most likely adjecent node to update potential from boundary classifier
    - P_new = P_old * e * P_ref
        - P_new is new output potential
        - P_old is old potential
        - P_ref is potential of best reference node
        - e is weight of reference edge (ouput of section classifier)
- Use greedy method to select nodes
    - Prune nodes which are incompatible with solution



#### Extract from Section 2 of paper

Aboundaryclassiﬁer[1]istrainedtopredicthowlikelya potential merge could happen. It takes features from a pair of potentialmergingregions (Ri,Rj) withinasectionandgives a probability pi,j that the two regions merge to one. Then every node Ni in the merge tree receives a potential Pi as Pi = pi0,j0 ·(1−pi,j), (1) wherepi0,j0 istheprobabilitythatthetwochildnodesNi0 and Nj0 merge,andpi,j istheprobabilitythatnodeNi mergewith its sibling node Nj. 