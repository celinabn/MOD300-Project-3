import random
import math
import numpy as np
from matplotlib import pyplot as plt

#Topic 1: Calculate DNA volume via Monte Carlo simulation.
#Task 1
def random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function makes a random point in x, y and z direction inside our defined box

    Returns: x, y, z values for our random point
    
    """
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = random.uniform(z_min, z_max)
    return (x, y, z)

def plot_random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function plots our random point in our simulation box
    """
    x, y, z = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
    print("Random point inside our box:", "\nX:", x, "\nY:", y, "\nZ:", z)
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(x,y,z, color="red")
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Random point in our simulation box")
    
    plt.show()


#Task 2
def random_sphere_in_box(x_min,x_max,y_min,y_max,z_min,z_max,r_min=0.1, r_max=2.0):
    """
    This function makes a random sized sphere by varying its radius in a random point in our defined simulation box.
    
    Returns: center and radius of the sphere.
    
    """
    center_of_sphere = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
    radius = np.random.uniform(r_min, r_max)          
    return (center_of_sphere, radius) 


def plot_random_sphere_in_box(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function plot our random sphere in our simulation box.
    """
    center, radi = random_sphere_in_box(x_min,x_max,y_min,y_max,z_min,z_max)
    print("Center of sphere:", center, "\nRadius of sphere:", radi) 
    
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    phi, theta =np.meshgrid(phi,theta)
    
    x = center[0] + radi * np.sin(phi) * np.cos(theta)
    y = center[1] + radi * np.sin(phi) * np.sin(theta)
    z = center[2] + radi * np.cos(phi)
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(x,y,z)
    
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Random sphere in our simulation box")
    
    plt.show()


#Task 3
def distance_between_center_and_point(sphere, x, y, z): 
    """
    This function calculates the distance between the center and the given point
    
    Returns: distance between center and given point.
    """
    
    center, radi = sphere
    cx, cy, cz = center
   
    x1 = (x - cx)**2
    y1 = (y - cy)**2
    z1 = (z - cz)**2
    return (x1 + y1 + z1)

def check_if_point_is_inside_sphere(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function checks if the point is inside or outside the sphere.
    
    """
    sphere = random_sphere_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
    point = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
    
    (center, r) = sphere
    x, y, z = point
    
    ans = distance_between_center_and_point(sphere, x, y, z)
    
    is_inside_if = ans<(r**2)

    print(f"Point being checked: ({x:.3f}, {y:.3f}, {z:.3f})")
    if is_inside_if:
        print("Point is inside the sphere") 
    else:
        print("Point is outside the sphere")   
        
    return point, sphere, is_inside_if


def plot_to_see_if_point_is_inside_sphere(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function plot our point and sphere in our simulation box to see if the point is inside or outside the sphere. 
    If its inside the point will turn green and if its outside it will turn red.
    """
    point, sphere, inside = check_if_point_is_inside_sphere(x_min, x_max, y_min, y_max, z_min, z_max)
    
    center, radi = sphere
    
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    phi, theta =np.meshgrid(phi,theta)

    
    x = center[0] + radi * np.sin(phi) * np.cos(theta)
    y = center[1] + radi * np.sin(phi) * np.sin(theta)
    z = center[2] + radi * np.cos(phi)
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(x,y,z, alpha=0.3)
    ax.scatter(point[0],point[1],point[2], color = ("green" if inside else "red"))
   
    
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Plot to see if the point is inside or outside of the sphere")
    
    plt.show()

#Task 4
def monte_carlo(x_min, x_max, y_min, y_max, z_min, z_max, N=1000000):
    """
    This function calculates the fraction of points inside the sphere
    """
    sphere = random_sphere_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
    center, radi = sphere
    inside_points = 0

    for i in range(N):
        x, y, z = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
        ans = distance_between_center_and_point(sphere, x, y, z)
        if ans < radi**2:
            inside_points += 1

    fraction_inside = inside_points / N
    print("The fraction of points inside the sphere is: ", fraction_inside)
    return fraction_inside, radi

def theoretical(x_min, x_max, y_min, y_max, z_min, z_max,radi):
    """
    This function calculate the theoretical value to compare with the monte carlo:
    """
    V_sphere = (4/3) * math.pi * (radi**3)
    V_box = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    fraction = V_sphere / V_box
    print("Theoretical volume ratio is: ", fraction)
    return fraction

def plot_volume_comparison(fraction_inside, fraction):
    """
    This function plot the volume comparison between the Monte Carlo and theoretical method.
    """
    plt.bar(["Monte Carlo", "Theoretical"], [fraction_inside, fraction], color=["skyblue", "orange"])
    plt.ylabel("Fraction (Volume ratio)")
    plt.title("Monte Carlo estimation of sphere volume fraction")
    plt.show()


#Task 5
def estimate_pi(INTERVAL = 1000):
    """
    This function calculates pi as a function of the number of randomly generated points.
    """
    circle_points = 0
    square_points = 0


    for i in range(INTERVAL**2):
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)

        origin_dist = rand_x**2 + rand_y**2
        if origin_dist <= 1:
            circle_points += 1

        square_points += 1

        pi = 4 * circle_points / square_points


    print("Final Estimation of Pi=", pi)


#Task 6
def ten_spheres_plotted(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function plot ten spheres in our simulation box.
    """
    ten_spheres = [random_sphere_in_box(x_min, x_max, y_min, y_max, z_min, z_max) for s in range(10)]
    
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    phi, theta =np.meshgrid(phi,theta)
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    for center, radi in ten_spheres:
        x = center[0] + radi * np.sin(phi) * np.cos(theta)
        y = center[1] + radi * np.sin(phi) * np.sin(theta)
        z = center[2] + radi * np.cos(phi)
        ax.plot_surface(x,y,z, alpha=0.3)
    
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ten spheres")
    
    plt.show()
    

#Task 7
def point_in_ten_spheres(spheres, x, y, z):
    """
    This function checks if the points are inside the spheres.
    """
    for sphere in spheres:
        center, r = sphere
        distance_squared = distance_between_center_and_point(sphere, x, y, z)
        if distance_squared < (r**2):
            return True
    return False

def fraction_in_ten_spheres(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    This function calculates the fraction of points inside the ten spheres (monte carlo method)
    """
    ten_spheres = [random_sphere_in_box(x_min, x_max, y_min, y_max, z_min, z_max) for s in range(10)]
    
    N = random.randint(1,10000)        
    inside_points = 0

    for t in range(N):
        x, y, z = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
        if point_in_ten_spheres(ten_spheres, x, y, z):
            inside_points += 1
        
    fraction_inside = inside_points / N
    print("The fraction of points inside the ten spheres is: ", fraction_inside)
    return fraction_inside, ten_spheres


def theoretical_ten_spheres(x_min, x_max, y_min, y_max, z_min, z_max, ten_spheres):
    """
    This function calculates the theoretical volume ratio for many spheres.
    """
    V_total = 0
    
    for sphere in ten_spheres: 
        center, radi = sphere
        V_sphere = (4/3) * math.pi * (radi**3)
        V_total += V_sphere
        
    V_box = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    fraction = V_total / V_box
    print("Theoretical volume ratio is: ", fraction)
    return fraction
    

#Task 8
def dimention_of_each_atom():
    """
    This function returns the dimention of each atoms in angstrom.
    """
    atomic_radius = {
        "H": 1.2,
        "C": 1.7,
        "N": 1.55,
        "O": 1.52,
        "P": 1.8}
    print("Dimention of each atom: ")
    for atom, r in atomic_radius.items():
        print(atom, ":", r)
    return atomic_radius

def dna(filename="dna_coords.txt"):
    """
    This function returns the atoms in the txt file with their element, coordinates and radius.
    """
    atomic_radius = {
        "H": 1.2,
        "C": 1.7,
        "N": 1.55,
        "O": 1.52,
        "P": 1.8}
    atoms = []
    
    for line in open(filename):
        split = line.split()
        element = split[0]
        
        x = float(split[1])
        y = float(split[2])
        z = float(split[3])
        r = atomic_radius[element]
        
        atoms.append((element, x, y, z, r))
    return atoms
        
#Task 9
def perfect_simulation_box(atoms):
    """
    This function finds the lowest and highest x, y and z value and it also makes sure that the whole sphere fits inside the box with its radius.
    """
    xmini = []
    xmaxi = []
    
    ymini = []
    ymaxi = []
    
    zmini = []
    zmaxi = []
    
    for (element, x, y, z, r) in atoms:
        xmini.append(x-r)
        xmaxi.append(x+r) 
        
        ymini.append(y-r)
        ymaxi.append(y+r) 
        
        zmini.append(z-r)
        zmaxi.append(z+r) 
        
    x_min = min(xmini)
    x_max = max(xmaxi)
    
    y_min = min(ymini)
    y_max = max(ymaxi)
    
    z_min = min(zmini)
    z_max = max(zmaxi)
    
    print("Simulation box min and max: ", "\nX from:", x_min, "to", x_max, "\nY from:", y_min, "to", y_max, "\nZ from:", z_min, "to", z_max,)
    return x_min, x_max, y_min, y_max, z_min, z_max
    

#Task 10
def point_in_atoms(atoms, x, y, z):
    """
    This function checks if the points are inside one of the atom spheres.
    """
    for atom in atoms:
        element, atomx, atomy, atomz, r = atom
        sphere = ((atomx, atomy, atomz), r)
        distance_squared = distance_between_center_and_point(sphere, x, y, z)
        if distance_squared < (r**2):
            return True
    return False

def fraction_in_atoms(x_min, x_max, y_min, y_max, z_min, z_max, atoms, N=100000):
    """
    This function calculates the fraction of points inside the dna atom spheres (monte carlo method)
    """
          
    inside_points = 0

    for a in range(N):
        x, y, z = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
        if point_in_atoms(atoms, x, y, z):
            inside_points += 1
        
    fraction_inside = inside_points / N
    print("The fraction of points inside the dna atom spheres is: ", fraction_inside)
    return fraction_inside


def theoretical_atoms(x_min, x_max, y_min, y_max, z_min, z_max, atoms):
    """
    This function calculates the theoretical volume ratio for the dna atom spheres.
    """
    V_total = 0
    
    for atom in atoms: 
        element, x, y, z, r = atom
        V_sphere = (4/3) * math.pi * (r**3)
        V_total += V_sphere
        
    V_box = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    fraction = V_total / V_box
    print("Theoretical volume ratio is: ", fraction)
    return fraction


#Topic 2: Random walk for accessible volume calculation
#Task 1
import numpy as np

def generate_random_walkers(amount_walkers, space_size=10):
    """
    This function generate a set of random walkers in 3D starting from different random points.
    """
    
    positions = np.zeros((amount_walkers, 3))
    for i in range(amount_walkers):
        positions[i] = np.random.uniform(0, space_size, 3)

    print("Positions: ", positions)
    
    return positions


#Task 2
def random_walkers_fast(number_of_walkers, max_placement=10):
    """
    This fast function generate a set of random walkers in 3D starting from different random points. it uses numpy in stead of a for loop to make it as a fast function.
    
    """
    walkers = np.random.uniform(-max_placement, max_placement, size=(number_of_walkers, 3))


    print("Positions: ", walkers)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(walkers[:1000,0], walkers[:1000,1], walkers[:1000,2],  s=2)
    plt.show()
    
    return walkers

#Task 5   
def test_approch():
    """
    This function 
    And it prints out the volume of the box, the accessible fraction and the accessible volume.
    """
    atoms_data = dna("dna_coords.txt")
    
    x_min, x_max, y_min, y_max, z_min, z_max = perfect_simulation_box(atoms_data)

    atom_positions = np.array([[a[1], a[2], a[3]] for a in atoms_data], dtype=float)
    atom_radius = np.array([a[4] for a in atoms_data], dtype=float)

    V_box = (x_max -x_min) * (y_max- y_min) *(z_max- z_min)
    assert V_box > 0

    M = 1000000
    probe_radius = 1.4
    occupied_area = atom_radius + probe_radius

    accessible = 0
    
    for i in range(M):
        x, y, z = random_point_in_box(x_min, x_max, y_min, y_max, z_min, z_max)
        p = np.array([x,y,z])
        d = np.linalg.norm(atom_positions - p, axis=1)
        
        if np.any(d < occupied_area):
            continue
        accessible += 1

    f_accessible = accessible / M
    V_accessible = f_accessible * V_box

    assert 0.0 <= f_accessible <= 1.0
    assert 0.0 <= V_accessible <= V_box

    print(f"V_box = {V_box:.3f} Å³")
    print(f"Accessible fraction = {f_accessible:.6f}")
    print(f"Accessible volume = {V_accessible:.3f} Å³")

    