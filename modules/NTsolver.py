

def precompute_geometry(R, I, G, r_division, geo, BC, geometry_code):
    """Run once before training. Returns fixed arrays."""
    Delta_r = R / I
    centers = np.arange(I) * Delta_r + 0.5 * Delta_r
    edges   = np.arange(I + 1) * Delta_r

    # geometry-dependent volumes and surfaces
    if geometry_code == 0:
        S = np.ones(I + 1);  S[0] = 0.0
        V = np.ones(I) * Delta_r
    elif geometry_code == 1:
        S = 2 * np.pi * edges
        V = np.pi * (edges[1:]**2 - edges[:-1]**2)
    elif geometry_code == 2:
        S = 4 * np.pi * edges**2
        V = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)

    # for each cell i, which region index does it belong to?
    region_of_cell = []
    for r in centers:
        for idx_r, bspec in enumerate(geo.boundaries):
            if r <= bspec.radius:
                region_of_cell.append(idx_r)
                break
        else:
            region_of_cell.append(len(geo.boundaries) - 1)

    return dict(
        Delta_r        = Delta_r,
        centers        = centers,
        S              = S,
        V              = V,
        region_of_cell = region_of_cell,  # Python list of ints, length I
        G              = G,
        I              = I,
        BC             = BC,
    )