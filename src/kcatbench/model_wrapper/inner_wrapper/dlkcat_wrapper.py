import sys
import logging

from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, ensure_data_subfolder
from kcatbench.model_wrapper.wrapper_progress import progress_completed, progress_started

DLKCAT_CODE_DIR = (MODELS_DIR / "DLKcat" / "DeeplearningApproach")
DLKCAT_DATA_DIR = (DATA_DIR / "DLKcat") 

if str(DLKCAT_CODE_DIR / "Code" / "example") not in sys.path:
    sys.path.insert(0, str(DLKCAT_CODE_DIR / "Code" / "example"))


from kcatbench.model_wrapper.base_model import BaseModel
import model
import pandas as pd
import numpy as np
import zipfile
import torch
import requests
import math
from rdkit import Chem
from collections import defaultdict


LOGGER = logging.getLogger(__name__)


class DLKcatWrapper(BaseModel):
    name = "DLKcat"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        ensure_data_subfolder(DLKCAT_DATA_DIR)

        input_zip_file = DLKCAT_CODE_DIR / "Data" / "input.zip"
        
        if not input_zip_file.exists():
            raise FileNotFoundError(f"DLKcat resource zip not found at: {input_zip_file}")

        with zipfile.ZipFile(input_zip_file, 'r') as zip_ref:
            zip_ref.extractall(DLKCAT_DATA_DIR)

        self.fingerprint_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "fingerprint_dict.pickle")
        self.atom_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "atom_dict.pickle")
        self.bond_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "bond_dict.pickle")
        self.edge_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "edge_dict.pickle")
        self.word_dict = model.load_pickle(DLKCAT_DATA_DIR / "input" / "sequence_dict.pickle")

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Based on the predicition_for_input script from DLKcat
        progress_started(LOGGER, "dlkcat.predict", "DLKcat prediction started rows=%s.", len(input_data.index))

        n_fingerprint = len(self.fingerprint_dict)
        n_word = len(self.word_dict)

        radius=2
        ngram=3

        dim=10
        layer_gnn=3
        window=11
        layer_cnn=3
        layer_output=3

        device = torch.device(DEVICE)

        Kcat_model = model.KcatPrediction(device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(device)
        Kcat_model.load_state_dict(torch.load(str(DLKCAT_CODE_DIR / "Results" / "output" / "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"), map_location=device))
        predictor = Predictor(Kcat_model)
        progress_completed(LOGGER, "dlkcat.model", "DLKcat model initialization completed.")

        progress_started(LOGGER, "dlkcat.validation", "DLKcat input validation started.")
        clean_data = self._prepare_data(input_data)
        progress_completed(
            LOGGER,
            "dlkcat.validation",
            "DLKcat input validation completed valid_rows=%s.",
            len(clean_data["valid_indices"]),
        )

        results = []
        progress_started(
            LOGGER,
            "dlkcat.inference",
            "DLKcat inference started valid_rows=%s.",
            len(clean_data["valid_indices"]),
        )

        for id, idx in enumerate(clean_data["valid_indices"]):
            
            smiles = clean_data["substrates"][id]
            sequence = clean_data["sequence"][id]

            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

                atoms = self.create_atoms(mol)

                i_jbond_dict = self.create_ijbonddict(mol)

                fingerprints = self.extract_fingerprints(atoms, i_jbond_dict, radius)
                
                adjacency = np.array(Chem.GetAdjacencyMatrix(mol))

                words = self.split_sequence(sequence, ngram)

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                words = torch.LongTensor(words).to(device)

                inputs = [fingerprints, adjacency, words]

                prediction = predictor.predict(inputs)
                kcat_log_value = prediction.item()
                kcat_value = '%.4f' %math.pow(2, kcat_log_value)

                results.append(kcat_value)
            except Exception:
                LOGGER.exception("DLKcat failed for row index %s.", idx)
                results.append(pd.NA)
                continue

        progress_completed(
            LOGGER,
            "dlkcat.inference",
            "DLKcat inference completed attempted_rows=%s.",
            len(clean_data["valid_indices"]),
        )
        
        output = input_data.copy()
        output['dlkcat_kcat'] = pd.NA
        assign_indices, assign_values = self._expand_predictions(clean_data, results)
        if assign_indices:
            output.loc[assign_indices, 'dlkcat_kcat'] = assign_values
        progress_completed(
            LOGGER,
            "dlkcat.predict",
            "DLKcat predictions assigned rows=%s.",
            len(assign_indices),
        )

        return output
    
    def get_smiles(name):
        try :
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
            req = requests.get(url)
            if req.status_code != 200:
                smiles = pd.NA
            else:
                smiles = req.content.splitlines()[0].decode()
        except :
            smiles = pd.NA

        return smiles

    def create_atoms(self, mol):
        """Create a list of atom (e.g., hydrogen and oxygen) IDs
        considering the aromaticity."""
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atom_dict[a] for a in atoms]

        return np.array(atoms)
    
    def create_ijbonddict(self, mol):
        """Create a dictionary, which each key is a node ID
        and each value is the tuples of its neighboring node
        and bond (e.g., single and double) IDs."""
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict
    
    def extract_fingerprints(self, atoms, i_jbond_dict, radius):
        """Extract the r-radius subgraphs (i.e., fingerprints)
        from a molecular graph using Weisfeiler-Lehman algorithm."""

        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges
                (i.e., r-radius subgraphs or fingerprints)."""
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    try :
                        fingerprints.append(self.fingerprint_dict[fingerprint])
                    except :
                        self.fingerprint_dict[fingerprint] = 0
                        fingerprints.append(self.fingerprint_dict[fingerprint])

                nodes = fingerprints

                """Also update each edge ID considering two nodes
                on its both sides."""
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        try :
                            edge = self.edge_dict[(both_side, edge)]
                        except :
                            self.edge_dict[(both_side, edge)] = 0
                            edge = self.edge_dict[(both_side, edge)]

                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints)
    
    def split_sequence(self, sequence, ngram):
        sequence = '-' + sequence + '='
        words = list()
        for i in range(len(sequence)-ngram+1) :
            try :
                words.append(self.word_dict[sequence[i:i+ngram]])
            except :
                self.word_dict[sequence[i:i+ngram]] = 0
                words.append(self.word_dict[sequence[i:i+ngram]])

        return np.array(words)
    

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        predicted_value = self.model.forward(data)

        return predicted_value


