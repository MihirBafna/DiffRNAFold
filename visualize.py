import numpy as np
import pandas as pd
import os
from Bio import SeqIO
import Bio
import Bio.PDB
import Bio.SeqRecord
from biopandas.pdb import PandasPdb
import pickle
from tqdm import tqdm
import os
import networkx as nx
import py3Dmol


def visualize3D(pdbfile):
    with open(pdbfile) as ifile:
        system = "".join([x for x in ifile])
        view = py3Dmol.view()
        view.addModelsAsFrames(system)
    return view.setStyle({'cartoon': {'color':'spectrum'}})