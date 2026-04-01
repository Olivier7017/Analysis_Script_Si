import numpy as np
import matplotlib.pyplot as plt

datas = ["2.5e-8", "1e-5"]

for data in datas:
    # Load data
    t = np.load("t_array.npy")
    if data == "2.5e-8":
        dx_relative = np.load("sqrt_2epsilon.npy")  # 2.5e-8 as corrector_step_epsilon
        text_ang_pos = 1e-4
    elif data == "1e-5":
        dx_relative = np.load("sqrt_2epsilon_1em5.npy")  # 1e-5 as corrector_step_epsilon
        text_ang_pos = 1e4
    
    cell_length = 5.43
    dx_cartesian = dx_relative * cell_length
    
    # Physical constant
    phi = 17.51  # eV / Å^2
    
    # Energy per atom (eV)
    E_bruit = 0.5 * 3 * phi * dx_cartesian**2
    
    #for i in range(len(dx_cartesian)):
    #    print(f"{dx_cartesian[i]} ang: {E_bruit[i]} eV")
    
    # Plot
    plt.figure()
    plt.plot(t, E_bruit, label="E$_{\mathrm{noise}}$")
    plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("$E_{\mathrm{noise}}$ (eV/atom)")
    plt.suptitle("Noise-induced energy vs diffusion time")
    plt.title(f"Corrector step size : {data}", fontsize=10)
    #plt.title("Noise-induced energy vs diffusion time\n"
    #          f"Corrector step size : {data}")    
    plt.gca().invert_xaxis()  # since you go from t=1 → 0
    plt.grid()
    
    # Targets in Angstrom
    targets = [1.0, 0.25, 0.04]
    
    for index, val in enumerate(targets):
        # Find closest index
        idx = np.argmin(np.abs(dx_cartesian - val))
        t_val = t[idx]
    
        if index==0:
            plt.axvline(t_val, linestyle='--', color="green", alpha=0.7, label="dx ($\AA$)")
        else:
            plt.axvline(t_val, linestyle='--', color="green", alpha=0.7)
        plt.text(
            t_val,
            text_ang_pos,
            f"{val} Å",
            horizontalalignment='left',
        )
    
    kB = 8.617333262e-5  # eV/K
    T = 300  # K
    
    E_thermal = 1.5 * kB * T  # 3/2 kB T
    
    plt.axhline(E_thermal, linestyle='-', color="red", linewidth=1, alpha=0.7, label="E$_{\mathrm{therm}}$")
    plt.text(
        0.1,
        E_thermal+2e-2,
        r"$\frac{3}{2}k_BT$",
    )
    
    plt.legend()
    plt.savefig(f"ebruit_{data}.pdf")
    plt.clf()
