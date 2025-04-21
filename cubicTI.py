"""
This file contains the code for trigonometric interpolation for orthogonal cubic volumes.

Fast visualizations of your 3D data can be performed by using either:
a) plotvista() : Plots any 3D numpy array as a cube, with a colorbar/heatmap
b) slicedplot() : Plots the 3D numpy array's 3 orthogonal planes that slice it through it's center, with a colorbar/heatmap
c) dualplot() : Plots the above described heatmaps on the same plot as a single image, with a colorbar/heatmap

If you use this code, please cite:
(i) https://doi.org/10.1007/s40192-024-00389-9
(ii) https://doi.org/10.5281/zenodo.15258119

Mathematical Details of Trigonometric Interpolation: https://en.wikipedia.org/wiki/Trigonometric_interpolation
Implementation Example: https://doi.org/10.1007/s40192-024-00389-9

Created by Pranoy Ray
Contact: pranoy@gatech.edu
"""

import os
import time
import h5py
import numpy as np
import pyvista as pv

#Helper function for cubicTI
def populate_xsp(L,dS,target,nvox):
    fnl=np.zeros(target)
    lim=dS*(target-1)#Length extent of the standardized 2PS (basically 10A)
    L_arr = np.linspace(0,L,num=nvox+1)
    if target<len(L_arr): # Case where the target array is shorter than the lattice array
        cutoff = (len(L_arr)-target)//2
        fnl[:] = L_arr[cutoff:nvox-(cutoff)+1]
    else: # Case where the target array has a longer length scale than the lattice array
        pivot1 = target//2 - nvox//2
        pivot2 = target//2 + nvox//2
        print(f"Pivots:{pivot1},{pivot2}")
        fnl[pivot1:pivot2+1] = L_arr

        #Traverse backward
        idx = pivot1
        while True:
            if idx<0:
                break
            fnl[idx]=fnl[idx+1]-dS 
            idx-=1

        #Traverse forward
        for i in range(pivot2+1,target):
            fnl[i]=fnl[i-1]+dS    

    return fnl

#Modified Trigonometric Interpolation for cubic grid
def cubicTI(datax,target=51,filex=None,dS=0.2,L=0, save_file=True):
    print(filex,f" | Shape before Interpolation: ", np.shape(datax), f", Shape after Interpolation: {[target,target,target]}")

    #Assumptions & Approximations
    #No. of standardized voxels that represent the initial unit cell
    nvox = int(np.ceil(L/dS)) 
    #Adjusted lattice parameter
    L=dS*nvox
    #Ensuring odd number of voxels
    if nvox%2!=0:
        nvox+=1
        L+=dS
    print(f"nvox: {nvox}")

    #TARGET GRID DEFINITION
    #The initial microstructure grid has its origin at the center
    #Next we set the indexing origin & the length scale (xsp & corresponding tsp) origin to the center
    nS = target #no. of target voxels for each side of cube
    xa=(((nS-1)*dS)/2) #Defining the length scale cutoff
    L2=L/2.0
    #basically the length scales of the voxels where the trigonometric polynomial is evaluated
    #if cutoff is -5 to +5: xsp = [-5, -5+dS, -5+2dS,...,0,...., 5-2dS, 5-dS, 5]
    xsp = populate_xsp(L,dS,nS,nvox)
    
    tsp = 2*np.pi*(xsp)/L #has a length of 51 for this particular case
    T1,T2,T3 = np.meshgrid(tsp,tsp,tsp,indexing="ij")
    T = T1,T2,T3

    #INITIAL PERIODIC GRID
    nG = datax.shape[0] #no. of voxels on each side of initial grid (64)
    data3d = np.asarray(datax)
    #FFT
    rho=np.fft.fftn(data3d, norm="backward")
    #Creating the target array (empty) with the target shape
    rho_interp=np.zeros(T1.shape, dtype=complex)
    rho_real=np.real(rho)
    rho_imag=np.imag(rho)
    #indices of flattened initial array: we just need this to get the voxel coordinates stored in K,M,N
    rho_indices=np.arange(0,nG**3)#np.arange(-rho_lim,rho_lim) 
    #N, M, K contain the voxel-coordinates at which a rho is located in [64,64,64] (rho.shape)
    #FOURIER SPACE: N contains the Z coordinates, M contains the Y coordinates, K contains the X coordinates
    N,M,K=np.unravel_index(rho_indices,rho.shape) 

    print()
    print('GRID DEFINITION:')
    print('INITIAL')
    print(f'L: {L}A | Side Length of the cube')
    print(f'The initial grid represents -{L2}A to {L2}A')
    print(f'nG: {nG} | Intital no. of voxels per side')
    print(f'nS: {nS} | Target no. of voxels per side' )

    #converting the real indices to fourier indices
    K[K>nG/2] -= nG
    M[M>nG/2] -= nG
    N[N>nG/2] -= nG

    print('TARGET')
    print(f'xa: {xa}A | Cutoff length scale')


    def compute_cubicTI(K, M, N, rho_real, rho_imag, rho_interp, T):
        T1, T2, T3 = T
        for k,m,n in zip(K, M, N): 
            theta=k*T1+m*T2+n*T3
            rho_interp += rho_real[k,m,n]*np.cos(theta)-rho_imag[k,m,n]*np.sin(theta)#(rho[n,m,k]*np.exp(1j*theta)).real
        return rho_interp

    c1 = time.time()
    rho_interp = compute_cubicTI(K, M, N, rho_real, rho_imag, rho_interp, T)
    del K, M, N, rho_real, rho_imag, T
    c2 = time.time()
    ev_tim=round((c2 - c1),2)
    ev_min=round((ev_tim/60),2)
    print(f"Evaluation Time: {ev_tim}s or {ev_min}mins")

    rho_interp=rho_interp/(nG**3)
    interped=rho_interp.real
    del rho_interp

    if save_file:
        save_path=r'your_desired_path'
        filename=filex+'.h5'
        completename=os.path.join(save_path,filename)
        h5f = h5py.File(completename, 'w')
        h5f.create_dataset(filex, data=interped)
        h5f.close()
        del interped
    
    return interped

#PyVista 3D numpy array plot code
def plotvista(data_3d,filex="None"):
    #setting theme
    pv.set_plot_theme("document")

    #setting title
    pv.global_theme.title = filex

    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(np.shape(data_3d)) + 1

    # Edit the spatial reference
    #grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["data_3d"] = data_3d.flatten(order="F")  # Flatten the array!

    # Now plot the grid!
    grid.plot(show_edges=False)

def slicedplot(data3d,filex="Slice of 2 point stats"):
    pv.set_plot_theme("document")
    #setting title
    pv.global_theme.title = filex
    # Create the spatial reference
    grid = pv.ImageData()
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(data3d.shape) + 1
    # Edit the spatial reference
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
    # Add the data values to the cell data

    print(np.shape(data3d.flatten(order="F")))
    
    grid.cell_data["data_3d"] = data3d.flatten(order="F")  # Flatten the array!
    slices=grid.slice_orthogonal()
    slices.plot()

#Dual PyVista 3D numpy array plot code
def dualplot(data1,data2, fname=None,file1="     ",file2="     ",title="     "):
    #setting theme
    pv.set_plot_theme("document")
    #setting title
    pv.global_theme.title = title

    #Plot 1: Sliced Heatmap
    # Create the spatial reference
    grid1 = pv.ImageData()#UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid1.dimensions = np.array(np.shape(data1)) + 1
    # Edit the spatial reference
    grid1.spacing = (1, 1, 1)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid1.cell_data[file1] = data1.flatten(order="F")  # Flatten the array!

    grid1=grid1.slice_orthogonal()
    #slices.plot()
    # Now plot the grid!
    #grid1.plot(show_edges=False)

    #Plot 2: Boxed Heatmap
    grid2 = pv.ImageData()#UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid2.dimensions = np.array(np.shape(data2)) + 1
    # Edit the spatial reference
    grid2.spacing = (1, 1, 1)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid2.cell_data[file2] = data2.flatten(order="F")  # Flatten the array!
    # Now plot the grid!
    #grid2.plot(show_edges=False)

    #Dual Plot
    # Create a common colorbar for both subplots
    cmap = "coolwarm"
    clim = [0, 1]
    common_bar_args = dict(
        n_labels=5,
        position_x=1.9,
        position_y=0.15,
        width=0.02,
        height=0.7,
        label_font_size=16,
        font_family="arial",
        color="black",
    )

    common_bar_args = dict(
        n_labels=5,
        width=0.02,
        height=0.7,
        label_font_size=16,
        font_family="arial",
        color="black",
    )

    # Create a Plotter object and add the subplots
    plotter = pv.Plotter(shape=(1, 2), border=False)
    plotter.subplot(0, 0)
    plotter.add_mesh(grid1)#.warp_by_scalar())#, cmap=cmap, clim=clim)
    plotter.subplot(0, 1)
    plotter.add_mesh(grid2)#.warp_by_scalar())#, cmap=cmap, clim=clim)

    # Add the common colorbar
    #plotter.add_scalar_bar(**common_bar_args)
    plotter.camera_position = 'iso'
    plotter.show()
    if fname!=None:
        plotter.screenshot(fname, scale =20)#transparent_background=True)