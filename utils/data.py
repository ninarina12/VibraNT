import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import cmcrameri.cm as cm

import torch
import torch_geometric as tg

from ase import Atom, Atoms
from ase.io import read
from ase.io.jsonio import read_json
from ase.data import atomic_masses, covalent_radii
from ase.visualize.plot import plot_atoms
from ase.neighborlist import neighbor_list, natural_cutoffs

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from e3nn.io import CartesianTensor
from e3nn.o3 import ReducedTensorProducts


# standard display formatting
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

sns.set(font_scale=1, style='ticks')
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'


class CRD:
    def __init__(self):
        self.cmap = cm.lipari
        self.cmap_div = mpl.colors.LinearSegmentedColormap.from_list('cmap_div', [self.cmap(50), 'gray', self.cmap(150)])
        
        
    def load_data(self, data_file, has_unknown=False):
        
        if data_file.endswith('json'):
            # load and parse data dictionary
            data_dict = read_json(data_file)
            data = dict(zip(['mp_id', 'uid', 'structure', 'dim', 'diel', 'eigenvals', 'eigenvecs', 'raman', 'active'],
                            [[], [], [], [], [], [], [], [], []]))

            for i, entry in tqdm(data_dict.items(), total=len(data_dict), bar_format=bar_format):
                if isinstance(i, int):                

                    # dielectric tensor
                    d = entry['data']['diel']
                    d_mp = entry['data']['diel_mp']

                    if isinstance(d, list) and (len(d) == 9):
                        # if original dielectric tensor is available
                        data['diel'].append(np.array([eval(k) for k in d]).reshape(3,3))
                        valid = 1

                    elif d_mp:
                        # if dielectric tensor from Materials Project is available
                        data['diel'].append(np.array(d_mp))
                        valid = 1

                    else:
                        # no valid dielectric tensors available
                        valid = 0

                    if valid:
                        # Raman tensors
                        activity = entry['data']['Ramanactive']
                        raman = entry['data']['raman_tensors']

                        if isinstance(raman, dict):
                            # some Raman tensors are dictionaries indexed by active modes
                            active = list(raman.keys())
                            raman = np.stack(list(raman.values()))

                        else:
                            # Raman tensor lists are truncated if trailing inactive modes
                            activity = activity[:len(raman)]

                            if has_unknown:
                                # include modes with unknown Raman activity
                                active = [k for k in range(len(activity)) if (raman[k] != None) and (activity[k] >= 0)]
                            else:
                                active = [k for k in range(len(activity)) if (raman[k] != None) and (activity[k] > 0)]

                            raman = [raman[k] for k in active]

                            if len(raman):
                                raman = np.stack(raman)

                        # identifiers
                        data['mp_id'].append(entry['key_value_pairs']['mpid'])
                        data['uid'].append(entry['unique_id'])

                        # structure
                        atoms = {k : entry[k] for k in ['positions', 'numbers', 'cell', 'pbc'] if k in entry}
                        data['structure'].append(AseAtomsAdaptor.get_structure(Atoms(**atoms)))

                        # dimensionality
                        dim = entry['key_value_pairs']['dimensionality'][0]
                        try: dim = eval(dim)
                        except:
                            data['dim'].append(-1)
                        else:
                            data['dim'].append(int(dim))  

                        # eigenvalues
                        freqs = entry['key_value_pairs']['frequencies_cm'].replace('[', '').replace(']','').split()
                        data['eigenvals'].append([eval(k) for k in freqs])

                        # eigenvectors
                        data['eigenvecs'].append(np.stack(entry['data']['eigenvec']))

                        # Raman tensors
                        data['raman'].append(raman)

                        # Raman activity
                        data['active'].append(active)

            self.data = pd.DataFrame.from_dict(data)
            self.data['formula'] = self.data['structure'].apply(lambda x: x.formula)
            self.data['spacegroup'] = self.data['structure'].apply(lambda x: x.get_space_group_info()[-1])
            print('Number of samples:', len(self.data))
        
        elif data_file.endswith('xyz'):
            xyz = read(data_file, index=':')

            data = dict(zip(['mp_id', 'uid', 'structure', 'dim', 'diel', 'eigenvals', 'eigenvecs', 'raman', 'active'],
                            [[], [], [], [], [], [], [], [], []]))

            for i, entry in tqdm(enumerate(xyz), total=len(xyz), bar_format=bar_format):
                # info
                d = entry.info

                if isinstance(d['chi'], np.ndarray):
                    data['mp_id'].append(d['id'].split('_')[0])
                    data['uid'].append(d['id'])

                    # structure
                    data['structure'].append(AseAtomsAdaptor.get_structure(entry))

                    # diel
                    data['diel'].append(d['chi'])

                    # other
                    data['dim'].append(None)
                    data['eigenvals'].append(None)
                    data['eigenvecs'].append(None)
                    data['raman'].append(None)
                    data['active'].append(None)

            self.data = pd.DataFrame.from_dict(data)
            self.data['formula'] = self.data['structure'].apply(lambda x: x.formula)
            self.data['spacegroup'] = self.data['structure'].apply(lambda x: x.get_space_group_info()[-1])
            print('Number of samples:', len(self.data))
            
        else:
            print(data_file.split('.')[-1] + ' file parser not implemented.')    
            
    
    def get_species_counts(self):
        self.data['species'] = self.data['structure'].apply(lambda x: list(set(map(str, x.species))))
        species, counts = np.unique(self.data['species'].sum(), return_counts=True)
        self.species_counts = dict(zip(species, counts))
        self.data['species_min'] = self.data['species'].apply(lambda x: min([self.species_counts[k] for k in x]))
        
     
    def set_species_counts(self, species_min):
        self.data = self.data[self.data['species_min'] >= species_min].reset_index(drop=True)
        print('Number of samples:', len(self.data))
        
        
    def plot_species_counts(self, species_min=None):
        counts = list(self.species_counts.values())
        n_min = np.arange(max(counts))
        n_samples = np.zeros_like(n_min)
        for i in tqdm(n_min, total=len(n_min), bar_format=bar_format):
            n_samples[i] = len(self.data[self.data['species_min'] >= i])
            
        fig, ax = plt.subplots(figsize=(3.5,3))
        ax.plot(n_min, n_samples, color='white')
        ax.set_xscale('log')

        ax1 = ax.twiny()
        ax1.plot(n_min/len(self.data), n_samples, color='white')
        ax1.set_xscale('log')
        ax1.spines[['right', 'left', 'top', 'bottom']].set_visible(False)

        ax2 = ax.twinx()
        ax2.plot(n_min, n_samples/len(self.data), color=self.cmap(50))
        ax2.set_xscale('log')
        ax2.spines[['right', 'left', 'top', 'bottom']].set_visible(False)

        if species_min:
            ax.axvline(species_min, color='black', ls=':')
            ax.axhline(n_samples[species_min], color='black', ls=':')

        ax.set_xlabel('Minimum examples')
        ax.set_ylabel('Number of examples')
        ax1.set_xlabel('Minimum examples / Total examples')
        ax2.set_ylabel('Fraction of examples')
        return fig


    def plot_structure(self, i=0, rotation='0x,0y,0z'):
        pystruct = self.data.iloc[i].structure
        composition = [list(map(str, site.species._data.keys())) for site in pystruct.sites.copy()]
        symbols = [k[0] for k in composition]
        atoms = np.unique(symbols)
        struct = Atoms(symbols=symbols, positions=pystruct.cart_coords.copy(), cell=pystruct.lattice.matrix.copy(), pbc=True)
        z_dict = dict(zip(atoms, [Atom(k).number for k in atoms]))

        fig, ax = plt.subplots(figsize=(3.5,3))
        norm = plt.Normalize(vmin=-0.5, vmax=len(atoms)-1)
        color_dict = dict(zip(atoms, [self.cmap(norm(i)) for i in range(len(atoms)-1,-1,-1)]))
        color = [mpl.colors.to_hex(color_dict[k]) for k in symbols]
        plot_atoms(struct, ax, radii=0.25, colors=color, rotation=(rotation))

        for patch in ax.patches:
            if isinstance(patch, mpl.patches.PathPatch):
                patch.set_edgecolor('black')

        numbers = [z_dict[j] for j in atoms]
        radii = 0.25*covalent_radii[numbers]
        x = ax.get_xlim()[1] + 1
        y = ax.get_ylim()[1]
        s = 0.8 if y > 8 else 0.4
        k = 0
        for i, r in enumerate(radii):
            k += r
            ax.add_patch(plt.Circle((x, y - k), r, ec='black', fc=color_dict[atoms[i]], clip_on=False))
            ax.text(x + radii.max() + 0.3, y - k, atoms[i], ha='left', va='center')
            k += r + s

        ax.axis('off')
        return fig
    
    
    def get_lattice_parameters(self):
        a = []
        for entry in self.data.itertuples():
            a.append(entry.structure.lattice.abc)
        self.a = np.stack(a)
        print('Average lattice parameter (a/b/c):', self.a[:,0].mean(), '/', self.a[:,1].mean(), '/', self.a[:,2].mean())


    def plot_lattice_parameters(self, n_bins=50):
        fig, ax = plt.subplots(1,1, figsize=(3.5,3))
        bins = np.linspace(np.floor(self.a.min()), np.ceil(self.a.max()), n_bins)
        colors = self.cmap([50,150,220])

        b = 0.
        for d, c, n in zip(['a', 'b', 'c'], colors, [self.a[:,0], self.a[:,1], self.a[:,2]]):
            y = ax.hist(n, bins=bins, fc=list(c[:-1]) + [0.7], ec=c, bottom=b, label=d)[0]
            b += y

        ax.set_xlabel('Lattice parameter $(\AA)$')
        ax.set_ylabel('Number of examples')
        ax.legend(frameon=False)
        return fig
    
    
    def plot_lattice_parameter_counts(self, a_max=None):
        c_max = int(np.ceil(self.a[:,2].max()))
        n_samples = np.array([len(self.a[self.a[:,2] <= k]) for k in range(1, c_max)])
        
        fig, ax = plt.subplots(figsize=(3.5,3))
        ax.plot(range(1, c_max), n_samples, color='white')
        ax.set_xscale('log')

        ax1 = ax.twinx()
        ax1.plot(range(1, c_max), n_samples/len(self.data), color=self.cmap(50))
        ax1.set_xscale('log')
        ax1.spines[['right', 'left', 'top', 'bottom']].set_visible(False)

        if a_max:
            ax.axvline(a_max, color='black', ls=':')
            ax.axhline(n_samples[int(a_max) - 1], color='black', ls=':')
            print('Number of samples:', n_samples[int(a_max) - 1])

        ax.set_xlabel(r'Maximum lattice constant $(\AA)$')
        ax.set_ylabel('Number of examples')
        ax1.set_ylabel('Fraction of examples')
        return fig
        

class Process:
    def __init__(self, species, Z_max, type_encoding, type_onehot, f_onehot, default_dtype): 
        self.cmap = cm.lipari
        self.datasets = ['Train', 'Valid.', 'Test']
        self.colors = dict(zip(self.datasets, [self.cmap(50), self.cmap(150), self.cmap(220)]))
        
        self.species = species
        self.Z_max = Z_max
        self.type_encoding = type_encoding
        self.type_onehot = type_onehot
        self.f_onehot = f_onehot
        self.default_dtype = default_dtype
        
        self.tij = CartesianTensor("ij=ji")
        self.rtp = ReducedTensorProducts('ij=ji', i='1o') # Note, specifying rtp to speed up conversion between reps
        
        
    def build_data(self, entry, r_max=4., scale=None, has_raman=False, has_d_matrix=False):
        composition = [list(map(str, site.species._data.keys())) for site in entry.structure.sites.copy()]
        occupancy = [list(map(float, site.species._data.values())) for site in entry.structure.sites.copy()]
        symbols = [k[0] for k in composition] # for labeling only, retain first specie if fractional occupancy
        positions = torch.from_numpy(entry.structure.cart_coords.copy())
        lattice = torch.from_numpy(entry.structure.lattice.matrix.copy()).unsqueeze(0)
        struct = Atoms(symbols=symbols, positions=positions.numpy(), cell=lattice.squeeze().numpy(), pbc=True)

        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
        if scale is not None:
            # use covalent radii, where scale indicates the multiplicative factor for natural cutoff
            cutoff = natural_cutoffs(struct, mult=scale)
        else:
            cutoff = r_max
        edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=struct, cutoff=cutoff, self_interaction=True)
        
        edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
        edge_vec = (positions[torch.from_numpy(edge_dst)] - positions[torch.from_numpy(edge_src)]
                    + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=self.default_dtype), lattice[edge_batch]))
        edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
        
        # compute node features and attributes
        x_node = torch.zeros((len(struct), self.Z_max))
        z_node = torch.zeros((len(struct), self.Z_max))
        m_node = torch.zeros((len(struct), 1))
        for i, (comp, occ) in enumerate(zip(composition, occupancy)):
            for specie, k in zip(comp, occ):
                x_node[i,:] += self.f_onehot[self.type_encoding[specie]]      # node feature
                z_node[i,:] += k*self.type_onehot[self.type_encoding[specie]] # atom type (node attribute)
                m_node[i] += k*atomic_masses[self.type_encoding[specie] + 1]

        if has_raman:
            raman = torch.zeros((len(entry.eigenvals),3,3))
            if len(entry.active):
                raman[entry.active] = torch.from_numpy(entry.raman)
            raman = self.tij.from_cartesian(raman, rtp=self.rtp)
        else:
            raman = None
            
        if has_d_matrix:
            d_matrix = torch.from_numpy(entry.d_matrix).reshape(-1,3)
        else:
            d_matrix = None
        
        data = tg.data.Data(
            lattice=lattice, symbol=symbols, pos=positions, x_in=x_node, z_in=z_node, m=m_node, vol=entry.structure.volume,
            eigenvec=torch.from_numpy(entry.eigenvecs).permute(1,0,2).reshape(-1,3),
            diel=self.tij.from_cartesian(torch.from_numpy(entry.diel), rtp=self.rtp).unsqueeze(0),
            raman=raman,
            d_matrix=d_matrix,
            edge_shift=torch.tensor(edge_shift, dtype=self.default_dtype),
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            edge_len=edge_len
        )
        return data


    def train_valid_test_split(self, data, valid_size, test_size, seed=12, plot=False):
        # perform an element-balanced train/valid/test split
        print('Split train/dev ...')
        self.count_max = 0.
        dev_size = valid_size + test_size
        stats = self.get_element_statistics(data)
        idx_train, idx_dev = self.split_data(stats, dev_size, seed)

        print('Split valid/test ...')
        stats_dev = self.get_element_statistics(data.iloc[idx_dev])
        idx_valid, idx_test = self.split_data(stats_dev, test_size/dev_size, seed)
        idx_train += data[~data.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

        print('Number of training examples:', len(idx_train))
        print('Number of validation examples:', len(idx_valid))
        print('Number of testing examples:', len(idx_test))
        print('Total number of examples:', len(idx_train + idx_valid + idx_test))
        assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0
        
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test
        
        if plot:
            # plot element representation in each dataset
            stats['Train'] = stats['data'].map(lambda x: self.element_representation(x, np.sort(idx_train)))
            stats['Valid.'] = stats['data'].map(lambda x: self.element_representation(x, np.sort(idx_valid)))
            stats['Test'] = stats['data'].map(lambda x: self.element_representation(x, np.sort(idx_test)))
            stats = stats.sort_values('symbol')

            fig, ax = plt.subplots(2,1, figsize=(2*3.5,4))
            b0, b1 = 0., 0.
            for i, dataset in enumerate(self.datasets):
                self.split_subplot(ax[0], stats[:len(stats)//2], self.species[:len(stats)//2], dataset, bottom=b0,
                                   legend=True)
                self.split_subplot(ax[1], stats[len(stats)//2:], self.species[len(stats)//2:], dataset, bottom=b1)

                b0 += stats.iloc[:len(stats)//2][dataset].values
                b1 += stats.iloc[len(stats)//2:][dataset].values
            
            fig.supylabel('Counts', fontsize=12, x=0.04)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.15)
            return fig


    def get_element_statistics(self, data):    
        # create dictionary indexed by element names storing index of samples containing given element
        species_dict = {k: [] for k in self.species}
        for entry in data.itertuples():
            for specie in entry.species:
                species_dict[specie].append(entry.Index)

        # create dataframe of element statistics
        stats = pd.DataFrame({'symbol': self.species})
        stats['data'] = stats['symbol'].astype('object')
        for specie in self.species:
            stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
        stats['count'] = stats['data'].apply(len)
        self.count_max += stats['count'].max()
        return stats


    def split_data(self, data, test_size, seed=12):
        # initialize output arrays
        idx_train, idx_test = [], []

        # remove empty examples
        data = data[data['data'].str.len()>0]

        # sort df in order of fewest to most examples
        data = data.sort_values('count')

        for _, entry in tqdm(data.iterrows(), bar_format=bar_format, total=len(data)):
            _data = entry.to_frame().T.explode('data')

            try:
                _idx_train, _idx_test = train_test_split(_data['data'].values, test_size=test_size, random_state=seed)
            except:
                # too few examples to perform split - these examples will be assigned based on
                # other constituent elements (assuming not elemental examples)
                pass

            else:
                # add new examples that do not exist in previous lists
                idx_train += [k for k in _idx_train if k not in idx_train + idx_test]
                idx_test += [k for k in _idx_test if k not in idx_train + idx_test]

        return idx_train, idx_test


    def element_representation(self, x, idx):
        # get number of samples containing given element in dataset
        return len([k for k in x if k in idx])


    def split_subplot(self, ax, data, species, dataset, bottom=0., legend=False):    
        # plot element representation
        width = 0.4
        color = self.colors[dataset]
        bx = np.arange(len(species))

        ax.bar(bx, data[dataset], width, fc=list(color[:-1]) + [0.7], ec=color, bottom=bottom, label=dataset)

        ax.set_xticks(bx)
        ax.set_xticklabels(species)
        ax.tick_params(axis='x', direction='in', length=0, width=1)
        ax.set_yscale('log')
        
        if self.count_max:
            ax.set_ylim(top=1.05*self.count_max)
        if legend:
            ax.legend(frameon=False, ncol=3, loc='upper left')
            
    
    def get_neighbors(self, data):
        n = [[],[],[]]
        for k, idx in enumerate([self.idx_train, self.idx_valid, self.idx_test]):
            for entry in data.iloc[idx].itertuples():
                N = entry.input.pos.shape[0]
                for i in range(N):
                    n[k].append(len((entry.input.edge_index[0] == i).nonzero()))
        self.n_train, self.n_valid, self.n_test = [np.array(k) for k in n]


    def plot_neighbors(self, n_bins=50):
        fig, ax = plt.subplots(1,1, figsize=(3.5,3))
        bins = np.linspace(np.floor(self.n_train.min()), np.ceil(self.n_train.max()), n_bins)

        b = 0.
        for (d, c), n in zip(self.colors.items(), [self.n_train, self.n_valid, self.n_test]):
            y = ax.hist(n, bins=bins, fc=list(c[:-1]) + [0.7], ec=c, bottom=b, label=d)[0]
            b += y

        ax.set_xlabel('Number of neighbors')
        ax.set_ylabel('Number of examples')
        ax.legend(frameon=False)
        return fig