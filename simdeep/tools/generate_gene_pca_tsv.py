import os
import json
import argparse
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

def write_tsv(path, header, samples, mat):
    with open(path, 'w') as f:
        f.write('\t'.join(['Samples'] + header) + '\n')
        for i, s in enumerate(samples):
            row = [('{:.6f}'.format(v)) for v in mat[i]]
            f.write('\t'.join([s] + row) + '\n')

def compute_pca(npy_path, samples, genes, k):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise RuntimeError('Expected 3D array')
    n_s, n_g, n_c = arr.shape
    if n_s != len(samples):
        raise RuntimeError('Sample size mismatch')
    if n_g != len(genes):
        raise RuntimeError('Gene size mismatch')
    out = np.zeros((n_s, n_g * k), dtype=np.float32)
    for j in range(n_g):
        x = arr[:, j, :].astype(np.float32)
        x = np.nan_to_num(x)
        p = PCA(n_components=k, random_state=0)
        z = p.fit_transform(x)
        out[:, j * k:(j + 1) * k] = z
    header = []
    for g in genes:
        for c in range(1, k + 1):
            header.append(f'{g}_PC{c}')
    return out, header

def fit_per_gene_pca(npy_path, samples, genes, k):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise RuntimeError('Expected 3D array')
    n_s, n_g, n_c = arr.shape
    if n_s != len(samples):
        raise RuntimeError('Sample size mismatch')
    if n_g != len(genes):
        raise RuntimeError('Gene size mismatch')
    models = []
    for j in range(n_g):
        x = arr[:, j, :].astype(np.float32)
        x = np.nan_to_num(x)
        p = PCA(n_components=k, random_state=0)
        p.fit(x)
        models.append(p)
    return models

def transform_with_models(npy_path, samples, genes, model_dict):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise RuntimeError('Expected 3D array')
    n_s, n_g, n_c = arr.shape
    if n_s != len(samples):
        raise RuntimeError('Sample size mismatch')
    train_genes = model_dict['genes']
    if list(genes) != list(train_genes):
        raise RuntimeError('Gene order mismatch between transform data and trained models')
    k = int(model_dict['k'])
    out = np.zeros((n_s, n_g * k), dtype=np.float32)
    models = model_dict['models']
    for j in range(n_g):
        x = arr[:, j, :].astype(np.float32)
        x = np.nan_to_num(x)
        z = models[j].transform(x)
        out[:, j * k:(j + 1) * k] = z
    header = []
    for g in genes:
        for c in range(1, k + 1):
            header.append(f'{g}_PC{c}')
    return out, header

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root')
    ap.add_argument('--fit-root')
    ap.add_argument('--transform-root')
    ap.add_argument('--model-dir')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--mods', nargs='*')
    ap.add_argument('--from_raw')
    ap.add_argument('--out_root')
    ap.add_argument('--d_gene', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--mirna_map')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.mods:
        mod_keys = args.mods
    else:
        mod_keys = ['RNA', 'MIR', 'METH']

    if args.from_raw:
        raw_root = args.from_raw
        out_root = args.out_root or os.path.join(os.path.dirname(raw_root), 'processed')
        gl_root = os.path.join(out_root, 'gene_level')
        os.makedirs(gl_root, exist_ok=True)
        def read_table_any(cands):
            for p in cands:
                if os.path.exists(p):
                    return pd.read_csv(p, sep='\t', index_col=0, compression='infer')
            return None
        mrna_df = read_table_any([
            os.path.join(raw_root, 'mrna.tsv'),
            os.path.join(raw_root, 'mrna.tsv.gz'),
            os.path.join(raw_root, 'rna.tsv'),
            os.path.join(raw_root, 'rna.tsv.gz'),
        ])
        meth_df = read_table_any([
            os.path.join(raw_root, 'meth.tsv'),
            os.path.join(raw_root, 'meth.tsv.gz'),
        ])
        mir_df = read_table_any([
            os.path.join(raw_root, 'mirna.tsv'),
            os.path.join(raw_root, 'mirna.tsv.gz'),
            os.path.join(raw_root, 'mir.tsv'),
            os.path.join(raw_root, 'mir.tsv.gz'),
        ])
        if mrna_df is None or meth_df is None:
            raise RuntimeError('Missing required raw files in --from_raw')

        def maybe_transpose(df):
            try:
                idx0 = str(df.index[0]) if len(df.index) > 0 else ''
            except Exception:
                idx0 = ''
            # Heuristic: DeepProg raw often has Samples as index (TCGA.*), features as columns
            if (idx0.startswith('TCGA') or idx0.startswith('Samples')):
                return df.T
            return df

        mrna_df = maybe_transpose(mrna_df)
        meth_df = maybe_transpose(meth_df)
        if mir_df is not None:
            mir_df = maybe_transpose(mir_df)
        common = set(mrna_df.columns) & set(meth_df.columns)
        if mir_df is not None:
            common = common & set(mir_df.columns)
        samples = sorted(list(common))
        mrna_df = mrna_df[samples]
        meth_df = meth_df[samples]
        if mir_df is not None:
            mir_df = mir_df[samples]
        common_genes = list(set(mrna_df.index) & set(meth_df.index))
        common_genes.sort()
        mrna_df = mrna_df.loc[common_genes]
        meth_df = meth_df.loc[common_genes]
        mrna_df = mrna_df.fillna(mrna_df.mean(axis=1))
        meth_df = meth_df.fillna(meth_df.mean(axis=1))
        if mir_df is not None:
            mir_df = mir_df.fillna(mir_df.mean(axis=1))
        def zscore(df):
            m = df.mean(axis=1)
            s = df.std(axis=1)
            s[s == 0] = 1.0
            return (df.sub(m, axis=0)).div(s, axis=0)
        mrna_z = zscore(mrna_df)
        meth_z = zscore(meth_df)
        if mir_df is not None:
            mirna_z = zscore(mir_df)
        X_mrna = mrna_z.values.T.astype(np.float32)
        X_meth = meth_z.values.T.astype(np.float32)
        X_mir = mirna_z.values.T.astype(np.float32) if mir_df is not None else None
        gene_list = mrna_z.index.tolist()
        sample_list = samples
        class GeneEnc(nn.Module):
            def __init__(self, d, hid=128):
                super().__init__()
                self.d = d
                self.enc = nn.Sequential(nn.Linear(1, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Linear(hid, d))
                self.dec = nn.Sequential(nn.Linear(d, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Linear(hid, 1))
            def forward(self, x):
                S, G = x.shape
                z = self.enc(x.reshape(-1, 1))
                z = z.reshape(S, G, self.d)
                recon = self.dec(z.reshape(-1, self.d)).reshape(S, G)
                return recon, z
            def encode(self, x):
                S, G = x.shape
                z = self.enc(x.reshape(-1, 1)).reshape(S, G, self.d)
                return z
        def train_gene_autoencoder(data, d, epochs, bs, seed=0):
            # 设置PyTorch随机种子确保可复现
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # 设置CUDA确定性操作（如果支持）
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except:
                    pass
            np.random.seed(seed)
            import random
            random.seed(seed)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = GeneEnc(d).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            crit = nn.MSELoss()
            X = torch.tensor(data, dtype=torch.float32)
            ds = TensorDataset(X)
            # 使用generator确保DataLoader的shuffle可复现
            generator = torch.Generator()
            generator.manual_seed(seed)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=(device=='cuda'), generator=generator)
            for ep in range(epochs):
                model.train()
                for b in dl:
                    x = b[0].to(device)
                    recon, _ = model(x)
                    loss = crit(recon, x)
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                Z = model.encode(X.to(device)).cpu().numpy()
            return Z
        Z_mrna = train_gene_autoencoder(X_mrna, args.d_gene, args.epochs, args.batch_size, seed=args.seed)
        Z_meth = train_gene_autoencoder(X_meth, args.d_gene, args.epochs, args.batch_size, seed=args.seed+1)
        Z_mir = None
        if X_mir is not None and args.mirna_map and os.path.exists(args.mirna_map):
            map_rows = []
            with open(args.mirna_map, 'r') as f:
                for li, line in enumerate(f):
                    if li == 0 and line.startswith('mirna'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    map_rows.append((parts[0], parts[1], float(parts[2]) if len(parts)>2 else 1.0))
            mir_ids = mirna_z.index.tolist()
            gene_to_idx = {g:i for i,g in enumerate(gene_list)}
            mir_to_idx = {m:i for i,m in enumerate(mir_ids)}
            M = np.zeros((len(mir_ids), len(gene_list)), dtype=np.float32)
            for m,g,w in map_rows:
                if m in mir_to_idx and g in gene_to_idx:
                    M[mir_to_idx[m], gene_to_idx[g]] += w
            row_sums = M.sum(axis=1)
            row_sums[row_sums==0] = 1.0
            M = M / row_sums[:, None]
            X_gene_mir = X_mir @ M
            Z_mir = train_gene_autoencoder(X_gene_mir, args.d_gene, args.epochs, args.batch_size, seed=args.seed+2)
        def fuse_simple_attention(z_list):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            arrs = [torch.tensor(z, dtype=torch.float32, device=device) for z in z_list if z is not None]
            S, G, d = arrs[0].shape
            H = torch.stack(arrs, dim=2)
            Wa = nn.Linear(d, d).to(device)
            va = nn.Linear(d, 1).to(device)
            with torch.no_grad():
                h = torch.tanh(Wa(H.reshape(-1, d)))
                s = va(h).reshape(S, G, len(arrs))
                a = torch.softmax(s, dim=-1).unsqueeze(-1)
                Z = (H * a).sum(dim=2)
            return Z.cpu().numpy()
        Z_all = fuse_simple_attention([Z_mrna, Z_meth, Z_mir])
        np.save(os.path.join(gl_root, 'Z_gene_mrna.npy'), Z_mrna)
        np.save(os.path.join(gl_root, 'Z_gene_meth.npy'), Z_meth)
        if Z_mir is not None:
            np.save(os.path.join(gl_root, 'Z_gene_mirna.npy'), Z_mir)
        np.save(os.path.join(gl_root, 'Z_gene_all.npy'), Z_all)
        with open(os.path.join(gl_root, 'sample_list.json'), 'w') as f:
            json.dump(sample_list, f)
        with open(os.path.join(gl_root, 'gene_list.json'), 'w') as f:
            json.dump(gene_list, f)
        samples = sample_list
        genes = gene_list
        mods = {
            'RNA': os.path.join(gl_root, 'Z_gene_mrna.npy'),
            'METH': os.path.join(gl_root, 'Z_gene_meth.npy'),
        }
        if os.path.exists(os.path.join(gl_root, 'Z_gene_mirna.npy')):
            mods['MIR'] = os.path.join(gl_root, 'Z_gene_mirna.npy')
        for key in mod_keys:
            if key not in mods:
                continue
            p = mods[key]
            mat, header = compute_pca(p, samples, genes, args.k)
            out_path = os.path.join(out_root, f'{key.lower()}_gene_pca.tsv')
            write_tsv(out_path, header, samples, mat)
        return

    if args.fit_root and not args.transform_root:
        gl_root = os.path.join(args.fit_root, 'gene_level')
        samples = load_json(os.path.join(gl_root, 'sample_list.json'))
        genes = load_json(os.path.join(gl_root, 'gene_list.json'))
        mods = {
            'RNA': os.path.join(gl_root, 'Z_gene_mrna.npy'),
            'MIR': os.path.join(gl_root, 'Z_gene_mirna.npy'),
            'METH': os.path.join(gl_root, 'Z_gene_meth.npy'),
        }
        model_dir = args.model_dir or os.path.join(gl_root, 'pca_models')
        os.makedirs(model_dir, exist_ok=True)
        for key in mod_keys:
            p = mods[key]
            models = fit_per_gene_pca(p, samples, genes, args.k)
            payload = {
                'genes': genes,
                'k': args.k,
                'random_state': 0,
                'models': models,
            }
            out_pkl = os.path.join(model_dir, f'pca_models_{key.lower()}.pkl')
            with open(out_pkl, 'wb') as f:
                pickle.dump(payload, f)
        return

    if args.fit_root and args.transform_root:
        gl_fit = os.path.join(args.fit_root, 'gene_level')
        gl_trans = os.path.join(args.transform_root, 'gene_level')
        samples_trans = load_json(os.path.join(gl_trans, 'sample_list.json'))
        genes_trans = load_json(os.path.join(gl_trans, 'gene_list.json'))
        mods_trans = {
            'RNA': os.path.join(gl_trans, 'Z_gene_mrna.npy'),
            'MIR': os.path.join(gl_trans, 'Z_gene_mirna.npy'),
            'METH': os.path.join(gl_trans, 'Z_gene_meth.npy'),
        }
        model_dir = args.model_dir or os.path.join(gl_fit, 'pca_models')
        for key in mod_keys:
            pkl_path = os.path.join(model_dir, f'pca_models_{key.lower()}.pkl')
            with open(pkl_path, 'rb') as f:
                model_dict = pickle.load(f)
            mat, header = transform_with_models(mods_trans[key], samples_trans, genes_trans, model_dict)
            out_path = os.path.join(args.transform_root, f'{key.lower()}_gene_pca.tsv')
            write_tsv(out_path, header, samples_trans, mat)
        return

    if args.root:
        gl_root = os.path.join(args.root, 'gene_level')
        samples = load_json(os.path.join(gl_root, 'sample_list.json'))
        genes = load_json(os.path.join(gl_root, 'gene_list.json'))
        mods = {
            'RNA': os.path.join(gl_root, 'Z_gene_mrna.npy'),
            'MIR': os.path.join(gl_root, 'Z_gene_mirna.npy'),
            'METH': os.path.join(gl_root, 'Z_gene_meth.npy'),
        }
        for key in mod_keys:
            p = mods[key]
            mat, header = compute_pca(p, samples, genes, args.k)
            out_path = os.path.join(args.root, f'{key.lower()}_gene_pca.tsv')
            write_tsv(out_path, header, samples, mat)
        return

    raise RuntimeError('Must provide --root or (--fit-root [--transform-root])')

if __name__ == '__main__':
    main()
