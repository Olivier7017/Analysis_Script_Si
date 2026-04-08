from ase.io import read
from atoms_analysis.analysis import calc_rdf, calc_edist
from atoms_analysis.rdf_calc import plot_rdf
from atoms_analysis.energydist_calc import plot_edist
from atoms_analysis.energy_quantile import plot_equantile


def main():
    #do_calc()
    do_plot()


def do_plot():
    plot_rdf(["output/aSi555_rdf.dat"],
             "figures/aSi555_rdf.pdf", 
             title="RDF — aSi555")
    plot_edist(["output/aSi555_edist.dat"], 
               "figures/aSi555_edist.pdf", 
               n_atoms=1000, 
               title="Energy distribution — aSi555")

def do_calc():
    traj_list = [f"Si555_{i+1}.traj" for i in range(10)]

    print(f"Calc RDF")
    calc_rdf(traj_list, "output/aSi555_rdf.dat", rcut=5)

    print(f"Calc Energies")
    calc_edist(traj_list, "output/aSi555_edist.dat", recalc=True)

if __name__=="__main__":
    main()
