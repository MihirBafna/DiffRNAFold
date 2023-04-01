from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast, List
import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
import inspect
import logging
import traceback

from deepchem.utils import get_print_threshold
from deepchem.utils.typing import PymatgenStructure
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
from deepchem.utils.rdkit_utils import compute_all_pairs_shortest_path
from deepchem.utils.rdkit_utils import compute_pairwise_ring_info
from deepchem.feat import Featurizer

from console import console

class MolecularFeaturizer(Featurizer):
    """Abstract class for calculating a set of features for a
molecule.
    The defining feature of a `MolecularFeaturizer` is that it
    uses SMILES strings and RDKit molecule objects to represent
    small molecules. All other featurizers which are subclasses of
    this class should plan to process input which comes as smiles
    strings or RDKit molecules.
    Child classes need to implement the _featurize method for
    calculating features for a single molecule.
    Note
    ----
    The subclasses of this class require RDKit to be installed.
    """

    def __init__(self, use_original_atoms_order=False):
        """
        Parameters
        ----------
        use_original_atoms_order: bool, default False
            Whether to use original atom ordering or canonical ordering (default)
        """
        self.use_original_atoms_order = use_original_atoms_order

    def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
        """Calculate features for molecules.
        Parameters
        ----------
        datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
            RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
            strings.
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` samples.
        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of `datapoints`.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolfiles
            from rdkit.Chem import rdmolops
            from rdkit.Chem.rdchem import Mol
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if 'molecules' in kwargs:
            datapoints = kwargs.get("molecules")
            raise DeprecationWarning(
                'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
            )

        # Special case handling of single molecule
        if isinstance(datapoints, str) or isinstance(datapoints, Mol):
            datapoints = [datapoints]
        else:
            # Convert iterables to list
            datapoints = list(datapoints)

        features: list = []
        for i, datapoint in enumerate(datapoints):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)

            try:
                kwargs_per_datapoint = {}
                for key in kwargs.keys():
                    kwargs_per_datapoint[key] = kwargs[key][i]
                features.append(self._featurize(datapoint, **kwargs_per_datapoint))
            except Exception as e:
                logger.warning(
                    "Failed to featurize datapoint %d, %s. Appending empty array",
                    i, datapoint)
                logger.warning("Exception message: {}".format(e))
                features.append(np.array([]))

        return np.asarray(features)

def normalize_pc(points, furthest_distance):
	centroid = np.mean(points, axis=0)
	points -= centroid
	points /= furthest_distance
	return points

def _construct_atom_feature(atom: RDKitAtom, h_bond_infos: List[Tuple[int,
                                                                      str]],
                            use_chirality: bool,
                            use_partial_charge: bool, mol=None, furthest_distance=None) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.

    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    h_bond_infos: List[Tuple[int, str]]
        A list of tuple `(atom_index, hydrogen_bonding_type)`.
        Basically, it is expected that this value is the return value of
        `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
        value is "Acceptor" or "Donor".
    use_chirality: bool
        Whether to use chirality information or not.
    use_partial_charge: bool
        Whether to use partial charge data or not.

    Returns
    -------
    np.ndarray
        A one-hot vector of the atom feature.

    """
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
    positions = mol.GetConformer().GetAtomPosition(atom.GetIdx())
    normalized_positions = normalize_pc([positions.x, positions.y, positions.z], furthest_distance)
    atom_feat = np.concatenate([
        atom_type, formal_charge, hybridization, acceptor_donor, aromatic,
        degree, total_num_Hs, [normalized_positions[0]], [normalized_positions[1]], [normalized_positions[2]]
    ])

    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, np.array(chirality)])

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
    return atom_feat



def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    np.ndarray
        A one-hot vector of the bond feature.

    """
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


class PDBFeaturizer(MolecularFeaturizer):
    """This class is a featurizer of general graph convolution networks for molecules.

    The default node(atom) and edge(bond) representations are based on
    `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
    you could use this class as a guide to define your original Featurizer. In many cases, it's enough
    to modify return values of `construct_atom_feature` or `construct_bond_feature`.

    The default node representation are constructed by concatenating the following values,
    and the feature length is 30.

    - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
    - Formal charge: Integer electronic charge.
    - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
    - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
    - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
    - Partial charge: Calculated partial charge. (Optional)

    The default edge representation are constructed by concatenating the following values,
    and the feature length is 11.

    - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
    - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
    - Conjugated: A one-hot vector of whether this bond is conjugated or not.
    - Stereo: A one-hot vector of the stereo configuration of a bond.

    If you want to know more details about features, please check the paper [1]_ and
    utilities in deepchem.utils.molecule_feature_utils.py.

    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
        Journal of computer-aided molecular design 30.8 (2016):595-608.

    Note
    ----
    This class requires RDKit to be installed.

    """

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False):
        """
        Parameters
        ----------
        use_edges: bool, default False
            Whether to use edge features or not.
        use_chirality: bool, default False
            Whether to use chirality information or not.
            If True, featurization becomes slow.
        use_partial_charge: bool, default False
            Whether to use partial charge data or not.
            If True, this featurizer computes gasteiger charges.
            Therefore, there is a possibility to fail to featurize for some molecules
            and featurization becomes slow.
        """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def featurize(self, pdb_path: str,  max_nodes: int, furthest_distance=None, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph with some features.

        """
        from rdkit.Chem import rdmolfiles
        datapoint = rdmolfiles.MolFromPDBFile(pdb_path, removeHs=False)
        if(datapoint is None):
            return None, False
        if(datapoint.GetNumAtoms() > max_nodes) or datapoint.GetNumAtoms() == 0:
            return None, False
        try:

            if self.use_partial_charge:
                try:
                    datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
                except:
                    # If partial charges were not computed
                    try:
                        from rdkit.Chem import AllChem
                        AllChem.ComputeGasteigerCharges(datapoint)
                    except ModuleNotFoundError:
                        raise ImportError(
                            "This class requires RDKit to be installed.")

            # construct atom (node) feature
            h_bond_infos = construct_hydrogen_bonding_info(datapoint)
            atom_feats = []
            for atom in datapoint.GetAtoms():
                atom_feats.append(_construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                            self.use_partial_charge, mol=datapoint, furthest_distance=furthest_distance))
            atom_features = np.asarray(
                atom_feats,
                dtype=float,
            )

            # construct edge (bond) index
            src, dest = [], []
            for bond in datapoint.GetBonds():
                # add edge list considering a directed graph
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                src += [start, end]
                dest += [end, start]

            # construct edge (bond) feature
            bond_features = None  # deafult None
            if self.use_edges:
                features = []
                for bond in datapoint.GetBonds():
                    features += 2 * [_construct_bond_feature(bond)]
                bond_features = np.asarray(features, dtype=float)

            # load_sdf_files returns pos as strings but user can also specify
            # numpy arrays for atom coordinates
            pos = []
            if 'pos_x' in kwargs and 'pos_y' in kwargs and 'pos_z' in kwargs:
                if isinstance(kwargs['pos_x'], str):
                    pos_x = eval(kwargs['pos_x'])
                elif isinstance(kwargs['pos_x'], np.ndarray):
                    pos_x = kwargs['pos_x']
                if isinstance(kwargs['pos_y'], str):
                    pos_y = eval(kwargs['pos_y'])
                elif isinstance(kwargs['pos_y'], np.ndarray):
                    pos_y = kwargs['pos_y']
                if isinstance(kwargs['pos_z'], str):
                    pos_z = eval(kwargs['pos_z'])
                elif isinstance(kwargs['pos_z'], np.ndarray):
                    pos_z = kwargs['pos_z']

                for x, y, z in zip(pos_x, pos_y, pos_z):
                    pos.append([x, y, z])
                node_pos_features = np.asarray(pos)
            else:
                node_pos_features = None
            return GraphData(node_features=atom_features,
                            edge_index=np.asarray([src, dest], dtype=int),
                            edge_features=bond_features,
                            node_pos_features=node_pos_features), True
        except Exception as e:
            console.print_exception()
            return None, False

