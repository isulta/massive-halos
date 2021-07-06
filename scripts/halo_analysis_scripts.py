def center_of_mass(coords, masses):
    return np.array([np.sum((coords[:,i] * masses)) for i in range(3)])/np.sum(masses)

def halo_center(coords, masses):
    '''See Power et al. 2003.'''
    com = center_of_mass(coords, masses)

    coordsrel = coords - com
    r = np.linalg.norm(coordsrel, axis=1)
    
    radius = r.max()

    Nconverge = min(1000, len(masses)*0.01)

    # mask = np.full_like(masses, True, bool)

    iteration = 0

    coords_it = coords.copy()
    masses_it = masses.copy()

    comlist = [com]
    radiuslist = [radius]

    while len(masses_it) > Nconverge:
        radius *= (100-2.5)/100

        mask = r <= radius
        coords_it = coords_it[mask, :]
        masses_it = masses_it[mask]

        com = center_of_mass(coords_it, masses_it)

        coordsrel = coords_it - com
        r = np.linalg.norm(coordsrel, axis=1)
        
        iteration += 1
        comlist.append(com)
        radiuslist.append(radius)

        print(iteration, radius, np.format_float_scientific(len(masses_it)), com)
    
    return com, comlist, radiuslist

def halo_center_wrapper(pdata):
    coords = pdata['Coordinates']
    masses = pdata['Masses']
    return halo_center(coords, masses)