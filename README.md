This folder analyse the RDF, Energy distribution and Energy quantile of Si with some reference configuration
Here, the data was generated using a DiffusionModel (by default create batch of sample but also some excise and repaint). 


lammps ref contains the reference MD of Si at 300 K.
samples contains multiple subfolder. Each subfolder contains .traj files.
scripts does the analysis, from trajectory reading up to generating the data.

scripts contains -> 
 Python script : 
  analysis.py : The main script to run the analysis
  energy_quantile.py, energydist_calc.py and rdf_calc.py which does the analysis for each quantity.
  prepare lammps is used to evaluate the energy and is used like : python3 prepare_lammps.py to_evaluate.traj

 Subfolder :
  calc_energies : The folder which contains the evaluated files from prepare_lammps. Same subfolder structure as samples
  output : Also subfolder structure. Contains the processed values, ready to be plotted
  figures : The matplotlib figure

In order, you should :
 1. Generate Reference if not already there
 2. Generate sample and put them in samples
 3. Use prepare lammps to create lammps input to evaluate the energy of the sample
 4. Run the lammps input and put them in calc_energies
 5. Run analysis

Analysis can also make special_graph (so for instance if you wanna plot the effect of a different number of corrector step on the same graph). 

Finally, note that a big part of this code was generated using Claude. It is in not well written code.

