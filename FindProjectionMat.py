import numpy as np

# Linear camera calibration, using parts of the approach from page 13 of  
# https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/CameraCalibration-book-chapter.pdf
# and the matrix equation from 
# https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node4.html

PixelCoords = np.loadtxt("Features2D.txt", delimiter='  ')

WorldCoords = np.loadtxt("Features3D.txt", delimiter='   ')

# Simply going with points 1, 2, and 3 resulted in AC and BC pointing in the 
# same direction, zeroing out the cross product
A = WorldCoords[0, 0:3]
B = WorldCoords[1, 0:3]
C = WorldCoords[6, 0:3]

AC = C - A
BC = C - B

# Find the normal to the plane defined by the points A, B, and C
ACxBC = np.cross(AC, BC)

# The coordinates from the text file all have a "1" appended at the end, 
# so a 0 is appended to the cross product so that the upcoming matrix multiplication 
# won't fail.

ACxBC = np.append(ACxBC, [0])

# If all of the points from WorldCoords lie on the same plane, they won't work for camera 
# calibration. This checks that all of the coordinates don't lie on the same plane.
assert(np.shape(np.unique((np.matmul(WorldCoords, ACxBC))))[0] > 1)

# "G" is not exactly a good variable name, so I should clarify I'm following the convention
# from the chapter I linked above.
G = np.zeros((WorldCoords.shape[0], 12))

for i in range(WorldCoords.shape[0]):

    if i%2 == 0:
        
        G[i,0:4] = WorldCoords[i,:]
        G[i,8:12] = np.multiply(WorldCoords[i,:], -PixelCoords[i,0])

    if i%2 == 1:

        G[i,4:8] = WorldCoords[i,:]
        G[i,8:12] = np.multiply(WorldCoords[i,:], -PixelCoords[i,1])

G_transpose_G = np.matmul(np.transpose(G),G)

eigenvals, eigenvectors = np.linalg.eig(G_transpose_G)

minEigenValInd = np.argmin(eigenvals)

# The solution (according to the book) is the eigenvector associated with the 
# smallest eigenvalue. Interestingly enough, there is one eigenvalue on the 
# order of 10^-9, and the rest are much larger (on the order of 10^-2 -> 10^2)
Solution = eigenvectors[:, minEigenValInd]

# Check that we found a unit eigenvector
assert(np.linalg.norm(Solution) - 1 < 0.01, "Found non-unit eigenvector")

Solution = np.multiply(Solution, 2/Solution[-1])

ProjMat = np.zeros((3, 4))
ProjMat.flat[:len(Solution)]=Solution

for i in range(WorldCoords.shape[0]):

    predPixelCoord = np.matmul(ProjMat, WorldCoords[i,:])
    predPixelCoord = np.multiply(predPixelCoord, 1/predPixelCoord[-1])
    
    # Finally, check that our projection matrix is at least within 0.05 pixels 
    # of correctly approximating a pixel coordinate given a point in the real world.
    assert(predPixelCoord[0] - PixelCoords[i,0] < 0.05)
    assert(predPixelCoord[1] - PixelCoords[i,1] < 0.05)

