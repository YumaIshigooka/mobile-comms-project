import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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


def Q1_plot(Nc1, Nc3, Nc9):
    # Function to find the y-value at x = -5
    def find_y_at_x(sorted_data, cdf, x):
        idx = np.searchsorted(sorted_data, x, side='left')  # Find where x would fit
        if idx < len(cdf) and sorted_data[idx] == x:
            return cdf[idx]
        elif idx == 0:  # If x is smaller than all data points
            return 0
        elif idx == len(cdf):  # If x is larger than all data points
            return 1
        else:  # Interpolate between points
            x0, x1 = sorted_data[idx - 1], sorted_data[idx]
            y0, y1 = cdf[idx - 1], cdf[idx]
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    cdf_Nc1 = np.arange(1, len(Nc1) + 1) / len(Nc1)
    cdf_Nc3 = np.arange(1, len(Nc3) + 1) / len(Nc3)
    cdf_Nc9 = np.arange(1, len(Nc9) + 1) / len(Nc9)

    # Find the y-values for x = -5
    x_mark = -5
    y_Nc1 = find_y_at_x(Nc1, cdf_Nc1, x_mark)
    y_Nc3 = find_y_at_x(Nc3, cdf_Nc3, x_mark)
    y_Nc9 = find_y_at_x(Nc9, cdf_Nc9, x_mark)

    # Plot the CDFs
    plt.figure(figsize=(12, 8))
    plt.plot(Nc1, cdf_Nc1, marker='o', linestyle='-', label='Nc1', color='blue', markersize=0)
    plt.plot(Nc3, cdf_Nc3, marker='s', linestyle='-', label='Nc3', color='green', markersize=0)
    plt.plot(Nc9, cdf_Nc9, marker='^', linestyle='-', label='Nc9', color='red', markersize=0)

    # Mark the points where x = -5
    for y, label, color in zip([y_Nc1, y_Nc3, y_Nc9], ['Nc1', 'Nc3', 'Nc9'], ['blue', 'green', 'red']):
        plt.axvline(x=x_mark, color='black', linestyle='--', alpha=0.7)  # Vertical line
        plt.axhline(y=y, color=color, linestyle='--', alpha=0.7)  # Horizontal line
        plt.scatter([x_mark], [y], color=color, label=f'{label} at x = {x_mark:.1f}', zorder=5)
        plt.text(x_mark, y, f'{y:.3f}', fontsize=12, color=color, ha='left', va='bottom', fontweight='bold')

    # Plot customization
    plt.title('Cumulative Distribution Function (CDF) with Markers', fontsize=16)
    plt.xlabel('SIR Values (dB)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.show()




def SIR_power_control(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000, exp_from = 0.0, exp_to = 1.0, N_steps = 0.1):
    if (exp_to < exp_from):
        print("Range is not valid")
        return
    if (N_steps < 0):
        print("Not enough steps")
        return
    exp_from = float(exp_from)
    exp_to = float(exp_to)
    exp_steps = (exp_to - exp_from) / N_steps
    print(f"Computing the range [{exp_from:.4}, {exp_to:.4}], using steps of {exp_steps:.4}")
    _range = np.arange(exp_from, exp_to + exp_steps, exp_steps)
    SIRs = []
    for exp in _range:
        SIRl = []
        for i in range(N):
            # Generate random points
            users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

            # Calculate distances to center (0, 0)
            distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
            pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide
            pathloss_shadowing = [np.random.normal(p, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
            linear_ps  = [10 ** (p/10) for p in pathloss_shadowing]

            # Calculate distances to respective centers
            distances_to_antenna = [10 * np.log10(np.sqrt((x - x_center)**2 + (y - y_center)**2)) for x, y, x_center, y_center in users]
            pathloss_to_antenna = [alpha - nu * d for d in distances_to_antenna] # in dB so subtracting instead of divide
            pathloss_shadowing_antenna  = [np.random.normal(p, sigma) for p in pathloss_to_antenna] # Combining pathloss and Shadowing : Lp * X
            linear_ps_antenna = [10 ** (p/10) for p in pathloss_shadowing_antenna]

            for i in range(3):
                linear_ps[i] = linear_ps_antenna[i]

            # Divide by the pathloss and shadowing to its corresponding antenna
            curr = linear_ps[0][0] / (linear_ps_antenna[0][0] ** exp)
            curr_sector = calculate_sector(users[0][0][0], users[0][1][0], users[0][2], users[0][3])
            interference = 0
            for i in range(1, len(linear_ps)):
                directivity_side = calculate_sector(users[i][0][0], users[i][1][0], users[0][2], users[0][3]) == curr_sector
                same_sector = calculate_sector(users[i][0][0], users[i][1][0], users[i][2], users[i][3]) == curr_sector
                if (directivity_side) and (same_sector):
                    # Divide by the pathloss and shadowing to its corresponding antenna
                    interference += linear_ps[i][0] / (linear_ps_antenna[i] ** exp)

            SIR = curr / interference
            SIRl.append(SIR)
        SIRs.append(SIRl)
    return SIRs


def Q2_plot(SIRs_og, exp_from, exp_to, N_steps):
    exp_steps = (exp_to - exp_from) / N_steps

    SIRs = SIRs_og.copy()

    # Create a single figure
    plt.figure(figsize=(12, 8))

    # Define a colormap and normalize
    colormap = cm.get_cmap('viridis')  # Choose a colormap
    norm = mcolors.Normalize(vmin=0, vmax=len(SIRs) - 1)  # Normalize indices to colormap range
    y_small = 2
    x_small = -1
    # Plot each SIR CDF on the same axes with a color scale
    for i, SIRs in enumerate(SIRs):
        # Step 1: Convert to dB and sort
        SIRs_DB = np.array([10 * np.log10(SIR) for SIR in SIRs if SIR > 0])  # Avoid log of zero
        SIRs_DB = SIRs_DB.ravel()
        sorted_data = np.sort(SIRs_DB)

        # Step 2: Calculate cumulative probabilities
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # Step 3: Get color from colormap
        color = colormap(norm(i))

        # Step 5: Annotate the y-value where x = -5
        if -5 in sorted_data:
            y_value = cdf[np.where(sorted_data == -5)[0][0]]
        else:
            # Interpolate y-value if -5 is not exactly in sorted_data
            y_value = np.interp(-5, sorted_data, cdf)

        # Step 4: Plot on the same figure
        plt.plot(sorted_data, cdf, marker='o', linestyle='-', label=f'Exponent = {(exp_from + i*exp_steps):.3}, y={y_value:.5f}', alpha=0.7, markersize=2, color=color)
        if (y_value < y_small):
            y_small = y_value
            x_small = exp_from + i*exp_steps

    print(f"Smallest y = {y_small}, found at exponent {x_small}")

    # Add title, labels, grid, and legend
    plt.title('CDF of SIRs', fontsize=14)
    plt.xlabel('SIR Values (dB)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Exponents', fontsize=10)

    # Set x-axis range
    plt.xlim(-20, 60)

    # Show the plot
    plt.tight_layout()
    plt.show()


def Q2(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000, exp_from = 0.0, exp_to = 1, N_steps = 10):
    SIRs = SIR_power_control(hex_positions, radius, height, alpha, sigma, nu, N, exp_from, exp_to, N_steps)
    Q2_plot(SIRs, exp_from, exp_to, N_steps)
    return SIRs

def reuse_factor_1_throughput(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000, B = (100 * (10 ** 6)), SNRGap = 4):
    Rs_1 = []
    for i in range(N):
        # Generate random points
        users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

        # Calculate distances to center (0, 0)
        distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
        pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide

        pathloss_shadowing = [np.random.normal(p, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
        linear_ps  = [10 ** (p/10) for p in pathloss_shadowing]

        curr = linear_ps[0][0]
        curr_sector = calculate_sector(users[0][0][0], users[0][1][0], users[0][2], users[0][3])
        interference = 0
        for i in range(1, len(linear_ps)):
            directivity_side = calculate_sector(users[i][0][0], users[i][1][0], users[0][2], users[0][3]) == curr_sector
            if directivity_side:
                interference += linear_ps[i][0]

        SIR = curr / interference
        R = B * np.log2(1 + (SIR / SNRGap))
        Rs_1.append(R)
    return Rs_1

def reuse_factor_3_throughput(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000, B = (100 * (10 ** 6)), SNRGap = 4):
    B /= 3
    Rs_3 = []
    for i in range(N):
        # Generate random points
        users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

        # Calculate distances to center (0, 0)
        distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
        pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide

        pathloss_shadowing = [np.random.normal(p, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
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
        R = B * np.log2(1 + (SIR / SNRGap))
        Rs_3.append(R)
    return Rs_3


def reuse_factor_9_throughput(hex_positions, radius, height, alpha = 1, sigma = 8, nu = 3.8, N = 1000, B = (100 * (10 ** 6)), SNRGap = 4):
    B /= 9
    Rs_9 = []
    for i in range(N):
        # Generate random points
        users = generate_n_random_points_in_hexagons(hex_positions, radius, height)

        # Calculate distances to center (0, 0)
        distances_to_center = [10 * np.log10(np.sqrt((x - 0)**2 + (y - 0)**2)) for x, y, _, _ in users]
        pathloss = [alpha - nu * d for d in distances_to_center] # in dB so subtracting instead of divide

        pathloss_shadowing = [np.random.normal(p, sigma) for p in pathloss] # Combining pathloss and Shadowing : Lp * X
        linear_ps  = [10 ** (p/10) for p in pathloss_shadowing]

        curr = linear_ps[0][0]
        curr_sector = calculate_sector(users[0][0][0], users[0][1][0], users[0][2], users[0][3])
        curr_bw = (2 * users[0][2] + users[0][3]) % 3
        interference = 0
        for i in range(1, len(linear_ps)):
            directivity_side = calculate_sector(users[i][0][0], users[i][1][0], users[0][2], users[0][3]) == curr_sector
            same_sector = calculate_sector(users[i][0][0], users[i][1][0], users[i][2], users[i][3]) == curr_sector
            same_bw = (2 * v_hex(users[i][2], users[i][3]) + u_hex(users[i][2], users[i][3])) % 3 == curr_bw
            if directivity_side and same_sector and same_bw:
                interference += linear_ps[i][0]

        SIR = curr / interference
        R = B * np.log2(1 + (SIR / SNRGap))
        Rs_9.append(R)
    return Rs_9
