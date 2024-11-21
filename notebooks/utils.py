import numpy as np
import matplotlib.pyplot as plt
# Function to calculate which sector a point belongs to
def calculate_sector(x, y, center_x, center_y):
    angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi  # Convert to degrees
    angle = (angle + 360) % 360  # Normalize to 0–360°
    if 0 <= angle < 120:
        return 0  # Sector 1
    elif 120 <= angle < 240:
        return 1  # Sector 2
    else:
        return 2  # Sector 3
    

# Function to generate random points inside hexagonal areas
def is_point_in_hexagon(x, y, center_x, center_y, radius):
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hex = center_x + radius * np.cos(angles)
    y_hex = center_y + radius * np.sin(angles)
    hex_path = plt.Polygon(list(zip(x_hex, y_hex)))
    return hex_path.contains_point((x, y))


  # Generate random points
def generate_n_random_points_in_hexagons(hex_positions, radius, height, num_points=1):
    points = []
    for center_x, center_y in hex_positions:
        curr = 0
        min_x = center_x - radius
        max_x = center_x + radius
        min_y = center_y - height / 2
        max_y = center_y + height / 2
        
        sector_users = {0: [], 1: [], 2: []}
        while (len(sector_users[0]) < num_points) or (len(sector_users[1]) < num_points) or (len(sector_users[2]) < num_points):
            random_x = np.random.uniform(min_x, max_x, 1)
            random_y = np.random.uniform(min_y, max_y, 1)

            if is_point_in_hexagon(random_x, random_y, center_x, center_y, radius):
                sector = calculate_sector(random_x, random_y, center_x, center_y)
                if (len(sector_users[sector]) < num_points):
                    sector_users[sector].append((random_x, random_y))
                    points.append((random_x, random_y, center_x, center_y))
                    curr += 1
    return points



def v_hex(x, y, D = 1):
    return round(x / (D * np.cos(np.radians(30))))

def u_hex(x, y, D = 1):
    return round((y - D * v_hex(x, y, D) * 0.5) / D)

def x_cartesian(v_hex, u_hex, D = 1):
    return D * v_hex * np.cos(np.radians(30))

def y_cartesian(v_hex, u_hex, D = 1):
    return D * u_hex + D * v_hex * 0.5


def reuse_factor_1(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000):
    SIRs = []
    for i in range(N):  
        # Generate random points
        users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

        # Calculate distances to center (0, 0)
        distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
        pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide 

        sigma = 8 # standard dev used in the normalized distribution of shadowing
        pathloss_shadowing = [p + np.random.normal(0, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
        linear_ps  = [10 ** (p/10) for p in pathloss_shadowing]

        curr = linear_ps[0][0]
        curr_sector = calculate_sector(users[0][0][0], users[0][1][0], users[0][2], users[0][3])
        interference = 0
        for i in range(1, len(linear_ps)):
            directivity_side = calculate_sector(users[i][0][0], users[i][1][0], users[0][2], users[0][3]) == curr_sector
            if directivity_side: 
                interference += linear_ps[i][0]

        SIR = curr / interference
        SIRs.append(SIR)
    return SIRs


def reuse_factor_3(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000):
    SIRs = []
    for i in range(N):  
        # Generate random points
        users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

        # Calculate distances to center (0, 0)
        distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
        pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide 

        sigma = 8 # standard dev used in the normalized distribution of shadowing
        pathloss_shadowing = [p + np.random.normal(0, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
        linear_ps  = [10 ** (p/10) for p in pathloss_shadowing]

        curr = linear_ps[0][0]
        curr_sector = calculate_sector(users[0][0][0], users[0][1][0], users[0][2], users[0][3])
        interference = 0
        for i in range(1, len(linear_ps)):
            directivity_side = calculate_sector(users[i][0][0], users[i][1][0], users[0][2], users[0][3]) == curr_sector
            same_sector = calculate_sector(users[i][0][0], users[i][1][0], users[i][2], users[i][3]) == curr_sector
            if directivity_side and same_sector: 
                interference += linear_ps[i][0]

        SIR = curr / interference
        SIRs.append(SIR)
    return SIRs


def reuse_factor_9(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000):
    SIRs = []
    for i in range(N):  
        # Generate random points
        users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

        # Calculate distances to center (0, 0)
        distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
        pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide 

        sigma = 8 # standard dev used in the normalized distribution of shadowing
        pathloss_shadowing = [p + np.random.normal(0, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
        linear_ps  = [10 ** (p/10) for p in pathloss_shadowing]

        curr = linear_ps[0][0]
        curr_sector = calculate_sector(users[0][0][0], users[0][1][0], users[0][2], users[0][3])
        curr_bw = (2 * users[0][2] + users[0][3]) % 3
        interference = 0
        for i in range(1, len(linear_ps)):
            directivity_side = calculate_sector(users[i][0][0], users[i][1][0], users[0][2], users[0][3]) == curr_sector
            same_sector = calculate_sector(users[i][0][0], users[i][1][0], users[i][2], users[i][3]) == curr_sector
            same_bw = (2 * v_hex(users[i][2], users[i][3], height) + u_hex(users[i][2], users[i][3], height)) % 3 == curr_bw
            if directivity_side and same_sector and same_bw: 
                interference += linear_ps[i][0]

        SIR = curr / interference
        SIRs.append(SIR)
    return SIRs