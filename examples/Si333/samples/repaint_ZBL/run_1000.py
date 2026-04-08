from pathlib import Path
import numpy as np 
import time
from functools import wraps

from ase.io import Trajectory, read, write
from ase.calculators.emt import EMT
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.ase_for_diffusion_data_module import \
    ASEForDiffusionDataModuleParameters, ASEForDiffusionDataModule
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import EGNNScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import (ZBLForce, ZBLForceParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldAugmentedScoreNetworkParameters) 
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import AtomTypeLossParameters, MSELossParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import \
    ReduceLROnPlateauSchedulerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import AXLDiffusionParameters
from diffusion_for_multi_scale_molecular_dynamics.callbacks.callback_loader import create_all_callbacks
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.sample_maker_factory import create_sample_maker, create_sample_maker_parameters
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.nearest_neighbors_excisor import NearestNeighborsExcisionArguments
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.top_k_atom_selector import TopKAtomSelectorParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.data.utils import traj_to_AXL, AXL_to_traj


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timed
def main():
    nsample_per_struct = 256
    nstruct_to_excise = 1
    nsteps=1000

    for cellsize in [1, 2, 3]:
        for elems in [["Si"], ["Si", "Ge"]]:
            outname = f"{''.join(elems)}{cellsize}{cellsize}{cellsize}_{nsteps}.traj"
            initial_sample, ckpt = get_initial_sample_and_ckpt(cellsize, elems)
            natoms = len(initial_sample)
            cell_dim = np.asarray(initial_sample.get_cell().array, dtype=np.float32)
            model = model_from_checkpoint(ckpt)
            # No ZBL for this launch
            model = model_add_zbl_forces(model, elems)
            sample_maker = create_excise_and_repaint_sample_maker(
                model=model,
                elems=elems,
                natoms=natoms, 
                nsteps=nsteps,
                cell_dim=cell_dim,
                nsample_per_struct=nsample_per_struct,
                nstruct_to_excise=nstruct_to_excise,
            )
            uncertainties = np.random.rand(natoms) 
            do_make_samples(initial_sample, elems, sample_maker, uncertainties, outname)


def do_make_samples(initial_sample, elems, sample_maker, uncertainties, outname):
    initial_axl=traj_to_AXL([initial_sample], elems)
    samples = sample_maker.make_samples(structure=initial_axl[0],
                                        uncertainty_per_atom=uncertainties)
    final_atoms=AXL_to_traj(samples[0], elems)
    write(outname, final_atoms)


def get_initial_sample_and_ckpt(cellsize, elems):
    if elems == ["Si"]:
        if cellsize == 1:
            initial_sample = "/home/pomax/Data/Structures/Si/Si8.cif"
            ckpt = "/home/pomax/Data/DiffusionModel/26-06-2025_models/july26_si_egnn_1x1x1/run1/output/best_model/best_model-epoch=039-step=125040.ckpt"
        elif cellsize == 2:
            initial_sample = "/home/pomax/Data/Structures/Si/Si64.cif"
            ckpt = "/home/pomax/Data/DiffusionModel/26-06-2025_models/july26_si_egnn_2x2x2/run1/output/best_model/best_model-epoch=064-step=203190.ckpt"
        elif cellsize == 3:
            initial_sample = "/home/pomax/Data/Structures/Si/Si216.cif"
            ckpt = "/home/pomax/Data/DiffusionModel/26-06-2025_models/july26_si_egnn_3x3x3/run1/output/best_model/best_model-epoch=004-step=015630.ckpt"
    if elems == ["Si", "Ge"]:
        if cellsize == 1:
            initial_sample = "/home/pomax/Data/Structures/Si/Si4Ge4.cif"
            ckpt = "/home/pomax/Data/DiffusionModel/26-06-2025_models/july26_sige_egnn_1x1x1/run1/output/best_model/best_model-epoch=081-step=256332.ckpt"
        elif cellsize == 2:
            initial_sample = "/home/pomax/Data/Structures/Si/Si32Ge32.cif"
            ckpt = "/home/pomax/Data/DiffusionModel/26-06-2025_models/july26_sige_egnn_2x2x2/run1/output/best_model/best_model-epoch=068-step=215694.ckpt"
        elif cellsize == 3:
            initial_sample = "/home/pomax/Data/Structures/Si/Si108Ge108.cif"
            ckpt = "/home/pomax/Data/DiffusionModel/26-06-2025_models/july26_sige_egnn_3x3x3/run1/output/best_model/best_model-epoch=001-step=006252.ckpt"
    initial_sample = read(initial_sample)
    return initial_sample, ckpt


def model_from_checkpoint(ckpt):
    return AXLDiffusionLightningModel.load_from_checkpoint(ckpt, weights_only=False)


def model_add_zbl_forces(model, elems):
    trained_egnn_score_network = model.axl_network
    zbl_parameters = ZBLForceParameters(
        radial_cutoff=2.0,
        inner_radius_fraction=0.5,
        element_list=elems,
    )
    force_field_parameters = ForceFieldAugmentedScoreNetworkParameters(
        repulsive_force_parameters=zbl_parameters,
        force_activation_scale=100.0,
        use_for_training=False,    
    )
    force_score_network = ForceFieldAugmentedScoreNetwork(
        score_network=trained_egnn_score_network,
        force_field_parameters=force_field_parameters,
    )
    model.use_force_field_augmented_score_network(force_score_network, at_eval=True)
    return model


def create_excise_and_repaint_sample_maker(model, elems, cell_dim, natoms, nsteps, nsample_per_struct=1, nstruct_to_excise=1):
    device="cuda:0"
    sample_maker_params = create_sample_maker_parameters({
        "algorithm": "excise_and_repaint",
        "element_list": elems,
        "sample_box_size": cell_dim,
        "number_of_samples_per_substructure": nsample_per_struct,
        })
    atom_selector_params = TopKAtomSelectorParameters(top_k_environment=nstruct_to_excise)
    excisor_params = NearestNeighborsExcisionArguments(number_of_neighbors=4)
    noise_params = NoiseParameters(total_time_steps=nsteps,
                                   schedule_type="exponential",
                                   sigma_min=0.001,
                                   sigma_max=0.2)
    sampling_params = PredictorCorrectorSamplingParameters(number_of_samples=nsample_per_struct,
                                                           spatial_dimension=3,
                                                           number_of_corrector_steps=20,
                                                           num_atom_types=len(elems),
                                                           number_of_atoms=natoms,
                                                           use_fixed_lattice_parameters=True,
                                                           cell_dimensions=cell_dim,
                                                           record_samples=True)
    sample_maker = create_sample_maker(sample_maker_parameters=sample_maker_params,
                                       atom_selector_parameters=atom_selector_params,
                                       excisor_parameters=excisor_params,
                                       noise_parameters=noise_params,
                                       sampling_parameters=sampling_params,
                                       diffusion_model=model.axl_network,
                                       device=device)
    return sample_maker

if __name__=="__main__":
    main()
