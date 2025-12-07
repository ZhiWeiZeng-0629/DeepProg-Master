import os
import sys
import argparse
import numpy as np
import pandas as pd
import subprocess

def _read_any(path_root, names):
    for n in names:
        p = os.path.join(path_root, n)
        if os.path.exists(p):
            return pd.read_csv(p, sep='\t', index_col=0, compression='infer')
    return None

def _subset_df(df, samples):
    cols = list(df.columns)
    if len(set(samples) & set(cols)) > 0:
        order = [s for s in samples if s in cols]
        return df[order]
    idx = list(df.index)
    order = [s for s in samples if s in idx]
    return df.loc[order]

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _write_tsv(df, path, index=True):
    df.to_csv(path, sep='\t', index=index)

def main():
    print("[DEBUG] Starting external_validation.py main function...")
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--train-ratio', type=float, default=0.75)
    ap.add_argument('--mods', default='RNA METH MIR')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--d-gene', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--mirna-map')
    ap.add_argument('--seed', type=int, default=10045)
    ap.add_argument('--nb-features', type=int, default=200)
    ap.add_argument('--boost-epochs', type=int, default=3)
    ap.add_argument('--nb-it', type=int, default=1)
    ap.add_argument('--nb-threads', type=int, default=1)
    ap.add_argument('--folds', type=int, default=3)
    args = ap.parse_args()

    root = args.data_root
    surv_path = os.path.join(root, 'survival.tsv')
    surv = pd.read_csv(surv_path, sep='\t')
    rng = np.random.RandomState(args.seed)
    ids = surv['Samples'].tolist()
    rng.shuffle(ids)
    n_train = int(max(1, min(len(ids)-1, round(len(ids) * args.train_ratio))))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    _ensure_dir(train_dir)
    _ensure_dir(val_dir)
    _ensure_dir(os.path.join(train_dir, 'raw'))
    _ensure_dir(os.path.join(val_dir, 'raw'))

    train_surv = surv[surv['Samples'].isin(train_ids)]
    val_surv = surv[surv['Samples'].isin(val_ids)]
    _write_tsv(train_surv, os.path.join(train_dir, 'train_survival.tsv'), index=False)
    _write_tsv(val_surv, os.path.join(val_dir, 'val_survival.tsv'), index=False)

    print("[DEBUG] Data split and directories created successfully.")

    mrna = _read_any(root, ['rna.tsv', 'rna.tsv.gz', 'mrna.tsv', 'mrna.tsv.gz'])
    meth = _read_any(root, ['meth.tsv', 'meth.tsv.gz'])
    mir = _read_any(root, ['mir.tsv', 'mir.tsv.gz', 'mirna.tsv', 'mirna.tsv.gz'])

    mods = []
    if mrna is not None:
        mods.append('RNA')
    if meth is not None:
        mods.append('METH')
    if mir is not None:
        mods.append('MIR')
    mods_str = ' '.join(mods)

    if mrna is None or meth is None:
        raise RuntimeError('missing rna.tsv(.gz) or meth.tsv(.gz) in data root')

    mrna_tr = _subset_df(mrna, train_ids)
    meth_tr = _subset_df(meth, train_ids)
    _write_tsv(mrna_tr, os.path.join(train_dir, 'raw', 'rna.tsv'))
    _write_tsv(meth_tr, os.path.join(train_dir, 'raw', 'meth.tsv'))
    if mir is not None:
        mir_tr = _subset_df(mir, train_ids)
        _write_tsv(mir_tr, os.path.join(train_dir, 'raw', 'mir.tsv'))

    mrna_va = _subset_df(mrna, val_ids)
    meth_va = _subset_df(meth, val_ids)
    _write_tsv(mrna_va, os.path.join(val_dir, 'raw', 'rna.tsv'))
    _write_tsv(meth_va, os.path.join(val_dir, 'raw', 'meth.tsv'))
    if mir is not None:
        mir_va = _subset_df(mir, val_ids)
        _write_tsv(mir_va, os.path.join(val_dir, 'raw', 'mir.tsv'))

    cmd_tr = [
        'python','-u','simdeep/tools/generate_gene_pca_tsv.py','--from_raw',os.path.join(train_dir,'raw'),
        '--out_root',os.path.join(train_dir,'integrated'),
        '--k',str(args.k),'--d_gene',str(args.d_gene),'--epochs',str(args.epochs),'--batch_size',str(args.batch_size),
        '--seed',str(args.seed),'--mods'
    ] + mods
    if args.mirna_map:
        cmd_tr += ['--mirna_map',args.mirna_map]
    print(f"[DEBUG] Running subprocess: {' '.join(cmd_tr)}")
    sys.stdout.flush()
    subprocess.check_call(cmd_tr)
    print("[DEBUG] Subprocess finished.")

    # Fit per-gene PCA models on available modalities
    gl_train = os.path.join(train_dir,'integrated','gene_level')
    fit_mods = ['RNA','METH']
    if os.path.exists(os.path.join(gl_train,'Z_gene_mirna.npy')):
        fit_mods.append('MIR')
    print(f"[DEBUG] Running subprocess: python simdeep/tools/generate_gene_pca_tsv.py --fit-root {os.path.join(train_dir,'integrated')} --mods {' '.join(fit_mods)}")
    sys.stdout.flush()
    subprocess.check_call([
        'python','-u','simdeep/tools/generate_gene_pca_tsv.py','--fit-root',os.path.join(train_dir,'integrated'),
        '--mods'] + fit_mods)
    print("[DEBUG] Subprocess finished.")

    cmd_va = [
        'python','-u','simdeep/tools/generate_gene_pca_tsv.py','--from_raw',os.path.join(val_dir,'raw'),
        '--out_root',os.path.join(val_dir,'integrated'),
        '--k',str(args.k),'--d_gene',str(args.d_gene),'--epochs',str(args.epochs),'--batch_size',str(args.batch_size),
        '--seed',str(args.seed),'--mods'
    ] + mods
    if args.mirna_map:
        cmd_va += ['--mirna_map',args.mirna_map]
    print(f"[DEBUG] Running subprocess: {' '.join(cmd_va)}")
    sys.stdout.flush()
    subprocess.check_call(cmd_va)
    print("[DEBUG] Subprocess finished.")

    # Transform validation with trained PCA models (same modality set)
    print(f"[DEBUG] Running subprocess: python simdeep/tools/generate_gene_pca_tsv.py --fit-root {os.path.join(train_dir,'integrated')} --transform-root {os.path.join(val_dir,'integrated')} --mods {' '.join(fit_mods)}")
    sys.stdout.flush()
    subprocess.check_call([
        'python','-u','simdeep/tools/generate_gene_pca_tsv.py','--fit-root',os.path.join(train_dir,'integrated'),
        '--transform-root',os.path.join(val_dir,'integrated'),
        '--mods'] + fit_mods)
    print("[DEBUG] Subprocess finished.")

    from simdeep import coxph_from_r
    from simdeep.simdeep_boosting import SimDeepBoosting
    PATH_DATA = os.path.abspath('.') + '/'
    # Discover actually generated training modalities
    TRAINING_TSV = {}
    available_mods = []
    if os.path.exists(os.path.join('data','train','integrated','rna_gene_pca.tsv')):
        TRAINING_TSV['RNA'] = os.path.join('data','train','integrated','rna_gene_pca.tsv')
        available_mods.append('RNA')
    if os.path.exists(os.path.join('data','train','integrated','meth_gene_pca.tsv')):
        TRAINING_TSV['METH'] = os.path.join('data','train','integrated','meth_gene_pca.tsv')
        available_mods.append('METH')
    if os.path.exists(os.path.join('data','train','integrated','mir_gene_pca.tsv')):
        TRAINING_TSV['MIR'] = os.path.join('data','train','integrated','mir_gene_pca.tsv')
        available_mods.append('MIR')
    SURVIVAL_TSV = os.path.join('data','train','train_survival.tsv')
    # Generate timestamp for filenames and create experiment directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create experiment-specific directory: timestamp_seed
    exp_dir_name = f"{timestamp}_seed{args.seed}"
    exp_dir = os.path.join('data','integrated','external_validation', exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[INFO] Experiment results will be saved to: {exp_dir}")
    
    norm = {k:'none' for k in TRAINING_TSV.keys()}
    boosting = SimDeepBoosting(
        nb_threads=args.nb_threads,
        nb_it=args.nb_it,
        split_n_fold=args.folds,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name='external_validation',
        path_results=os.path.join('data','integrated'),  # 临时路径，后面会修改
        epochs=args.boost_epochs,
        survival_flag={'patient_id':'Samples','survival':'days','event':'event'},
        distribute=False,
        cluster_method='kmeans',
        node_selection='C-index',
        classification_method='ALL_FEATURES',
        use_autoencoders=False,
        feature_surv_analysis=True,
        normalization=norm,
        verbose=True,  # 确保输出详细信息
        seed=args.seed
    )
    # 修改 path_results 指向实验专用文件夹
    boosting.path_results = exp_dir
    print("[DEBUG] Starting SimDeepBoosting training...")
    boosting.fit()
    print("[DEBUG] SimDeepBoosting training finished.")
    
    boosting.do_KM_plot = True
    pv_full, pv_full_proba = boosting.predict_labels_on_full_dataset()
    cidx_full = boosting.compute_c_indexes_for_full_dataset()
    
    # 验证数据是否正确设置
    print(f'[DEBUG] After predict_labels_on_full_dataset:')
    print(f'  - hasattr full_labels: {hasattr(boosting, "full_labels")}')
    print(f'  - hasattr full_labels_proba: {hasattr(boosting, "full_labels_proba")}')
    print(f'  - hasattr survival_full: {hasattr(boosting, "survival_full")}')
    print(f'  - hasattr sample_ids_full: {hasattr(boosting, "sample_ids_full")}')
    if hasattr(boosting, 'full_labels') and boosting.full_labels is not None:
        print(f'  - full_labels shape: {boosting.full_labels.shape if hasattr(boosting.full_labels, "shape") else len(boosting.full_labels)}')
    if hasattr(boosting, 'full_labels_proba') and boosting.full_labels_proba is not None:
        print(f'  - full_labels_proba shape: {boosting.full_labels_proba.shape if hasattr(boosting.full_labels_proba, "shape") else len(boosting.full_labels_proba)}')
    if hasattr(boosting, 'survival_full') and boosting.survival_full is not None:
        print(f'  - survival_full length: {len(boosting.survival_full)}')
    
    out_labels_full = os.path.join(exp_dir, 'external_validation_full_labels.tsv')
    
    # 尝试多种方式获取样本ID
    ids_full = None
    if hasattr(boosting, 'sample_ids_full') and boosting.sample_ids_full is not None:
        ids_full = boosting.sample_ids_full
    if ids_full is None or len(ids_full) == 0:
        try:
            ids_full = boosting._from_model_dataset(boosting.models[0], 'sample_ids_full')
        except Exception:
            ids_full = None
    if ids_full is None or len(ids_full) == 0:
        try:
            ids_full = boosting._from_model_dataset(boosting.models[0], 'dataset').sample_ids_full
        except Exception:
            ids_full = None
    # 如果还是获取不到，从生存数据文件中读取
    if ids_full is None or len(ids_full) == 0:
        try:
            surv_df = pd.read_csv(os.path.join(root, 'survival.tsv'), sep='\t')
            if 'Samples' in surv_df.columns:
                ids_full = surv_df['Samples'].tolist()
            elif surv_df.index.name == 'Samples' or 'Samples' in surv_df.index.names:
                ids_full = surv_df.index.tolist()
        except Exception as e:
            print(f'WARN: cannot read sample IDs from survival file: {e}')
            ids_full = []
    
    def _scalar(x):
        try:
            arr = np.asarray(x)
            return arr.ravel()[0]
        except Exception:
            return x
    
    # 确保样本ID数量与标签数量匹配
    n_samples = 0
    if hasattr(boosting, 'full_labels') and boosting.full_labels is not None:
        n_samples = len(boosting.full_labels)
        print(f'[DEBUG] Found full_labels with {n_samples} samples')
    if n_samples == 0 and hasattr(boosting, 'full_labels_proba') and boosting.full_labels_proba is not None:
        n_samples = len(boosting.full_labels_proba)
        print(f'[DEBUG] Found full_labels_proba with {n_samples} samples')
    if n_samples == 0 and hasattr(boosting, 'survival_full') and boosting.survival_full is not None:
        n_samples = len(boosting.survival_full)
        print(f'[DEBUG] Found survival_full with {n_samples} samples')
    
    if n_samples == 0:
        print('[WARN] No samples found in boosting object. Available attributes:', [attr for attr in dir(boosting) if not attr.startswith('_')])
        print('[WARN] Attempting to get sample count from models...')
        try:
            if boosting.models and len(boosting.models) > 0:
                model = boosting.models[0]
                if hasattr(model, 'dataset') and hasattr(model.dataset, 'sample_ids_full'):
                    n_samples = len(model.dataset.sample_ids_full)
                    print(f'[DEBUG] Found {n_samples} samples from model.dataset.sample_ids_full')
                    ids_full = model.dataset.sample_ids_full.tolist() if hasattr(model.dataset.sample_ids_full, 'tolist') else list(model.dataset.sample_ids_full)
        except Exception as e:
            print(f'[WARN] Error getting samples from model: {e}')
    
    # 如果样本ID数量不匹配，生成默认ID
    if len(ids_full) != n_samples and n_samples > 0:
        if len(ids_full) == 0:
            ids_full = [f'Sample_{i+1}' for i in range(n_samples)]
        elif len(ids_full) < n_samples:
            # 补充缺失的ID
            for i in range(len(ids_full), n_samples):
                ids_full.append(f'Sample_{i+1}')
        else:
            # 截断多余的ID
            ids_full = ids_full[:n_samples]
    
    if n_samples == 0:
        print('[ERROR] Cannot write labels file: n_samples is 0. No data available.')
    else:
        try:
            with open(out_labels_full, 'w') as f:
                f.write('sample_id\tlabel\tproba_0\tdays\tevent\n')  # 写入表头
                # 处理 survival_full 的格式（可能是转置的矩阵）
                survival_full_arr = np.asarray(boosting.survival_full)
                # 如果形状是 (2, n_samples)，需要转置为 (n_samples, 2)
                if survival_full_arr.ndim == 2 and survival_full_arr.shape[0] == 2:
                    survival_full_arr = survival_full_arr.T
                elif survival_full_arr.ndim == 1:
                    # 如果是一维数组，假设只有days，补充event列
                    survival_full_arr = np.column_stack([survival_full_arr, np.zeros(len(survival_full_arr))])
                # 确保形状是 (n_samples, 2)
                if survival_full_arr.shape[0] != n_samples:
                    if survival_full_arr.shape[1] == n_samples:
                        survival_full_arr = survival_full_arr.T
                
                for i in range(n_samples):
                    sid = ids_full[i] if i < len(ids_full) else f'Sample_{i+1}'
                    lbl = boosting.full_labels[i] if hasattr(boosting, 'full_labels') and boosting.full_labels is not None and i < len(boosting.full_labels) else 0
                    proba0 = boosting.full_labels_proba[i,0] if hasattr(boosting, 'full_labels_proba') and boosting.full_labels_proba is not None and i < len(boosting.full_labels_proba) else 0.0
                    day = survival_full_arr[i, 0] if i < survival_full_arr.shape[0] else 0.0
                    ev = survival_full_arr[i, 1] if i < survival_full_arr.shape[0] and survival_full_arr.shape[1] > 1 else 0.0
                    f.write(f"{sid}\t{_scalar(lbl)}\t{_scalar(proba0)}\t{_scalar(day)}\t{_scalar(ev)}\n")
            print(f'[SUCCESS] Written {n_samples} labels/proba/days/events to', out_labels_full)
        except Exception as e:
            print('[ERROR] Cannot write external_validation_full_labels.tsv:', e)
            import traceback
            traceback.print_exc()

    test_dict = {}
    for k in TRAINING_TSV:
        test_dict[k] = TRAINING_TSV[k].replace('/train/','/val/')
    boosting.load_new_test_dataset(
        tsv_dict=test_dict,
        fname_key='val',
        path_survival_file=os.path.join('data','val','val_survival.tsv'),
        normalization=norm
    )
    boosting.do_KM_plot = True
    pv_val, pv_val_proba = boosting.predict_labels_on_test_dataset()
    cidx_val = boosting.compute_c_indexes_for_test_dataset()
    print('train_pvalue_full', pv_full)
    print('train_cindex_full', cidx_full)
    print('val_pvalue', pv_val)
    print('val_cindex', cidx_val)
    
    # 验证验证集数据是否正确设置
    print(f'[DEBUG] After predict_labels_on_test_dataset:')
    print(f'  - hasattr test_labels: {hasattr(boosting, "test_labels")}')
    print(f'  - hasattr test_labels_proba: {hasattr(boosting, "test_labels_proba")}')
    print(f'  - hasattr survival_test: {hasattr(boosting, "survival_test")}')
    print(f'  - hasattr sample_ids_test: {hasattr(boosting, "sample_ids_test")}')
    if hasattr(boosting, 'test_labels') and boosting.test_labels is not None:
        print(f'  - test_labels shape: {boosting.test_labels.shape if hasattr(boosting.test_labels, "shape") else len(boosting.test_labels)}')
    if hasattr(boosting, 'test_labels_proba') and boosting.test_labels_proba is not None:
        print(f'  - test_labels_proba shape: {boosting.test_labels_proba.shape if hasattr(boosting.test_labels_proba, "shape") else len(boosting.test_labels_proba)}')
    if hasattr(boosting, 'survival_test') and boosting.survival_test is not None:
        print(f'  - survival_test length: {len(boosting.survival_test)}')

    out_labels_val = os.path.join(exp_dir, 'external_validation_val_test_labels.tsv')
    
    # 尝试多种方式获取验证集样本ID
    ids = None
    if hasattr(boosting, 'sample_ids_test') and boosting.sample_ids_test is not None:
        ids = boosting.sample_ids_test
    if ids is None or len(ids) == 0:
        try:
            ids = boosting._from_model_dataset(boosting.models[0], 'sample_ids_test')
        except Exception:
            ids = None
    if ids is None or len(ids) == 0:
        try:
            # fallback: build ids from per-modality matrices keys intersection
            ids = boosting._from_model_dataset(boosting.models[0], 'dataset').sample_ids_test
        except Exception:
            ids = None
    # 如果还是获取不到，从验证集生存数据文件中读取
    if ids is None or len(ids) == 0:
        try:
            val_surv_path = os.path.join('data','val','val_survival.tsv')
            if os.path.exists(val_surv_path):
                surv_df = pd.read_csv(val_surv_path, sep='\t')
                if 'Samples' in surv_df.columns:
                    ids = surv_df['Samples'].tolist()
                elif surv_df.index.name == 'Samples' or 'Samples' in surv_df.index.names:
                    ids = surv_df.index.tolist()
        except Exception as e:
            print(f'WARN: cannot read sample IDs from validation survival file: {e}')
        ids = []
    
    # 确保样本ID数量与标签数量匹配
    n_samples_val = 0
    if hasattr(boosting, 'test_labels') and boosting.test_labels is not None:
        n_samples_val = len(boosting.test_labels)
        print(f'[DEBUG] Found test_labels with {n_samples_val} samples')
    if n_samples_val == 0 and hasattr(boosting, 'test_labels_proba') and boosting.test_labels_proba is not None:
        n_samples_val = len(boosting.test_labels_proba)
        print(f'[DEBUG] Found test_labels_proba with {n_samples_val} samples')
    if n_samples_val == 0 and hasattr(boosting, 'survival_test') and boosting.survival_test is not None:
        n_samples_val = len(boosting.survival_test)
        print(f'[DEBUG] Found survival_test with {n_samples_val} samples')
    
    if n_samples_val == 0:
        print('[WARN] No validation samples found in boosting object. Attempting to get from models...')
        try:
            if boosting.models and len(boosting.models) > 0:
                model = boosting.models[0]
                if hasattr(model, 'dataset') and hasattr(model.dataset, 'sample_ids_test'):
                    n_samples_val = len(model.dataset.sample_ids_test)
                    print(f'[DEBUG] Found {n_samples_val} samples from model.dataset.sample_ids_test')
                    ids = model.dataset.sample_ids_test.tolist() if hasattr(model.dataset.sample_ids_test, 'tolist') else list(model.dataset.sample_ids_test)
        except Exception as e:
            print(f'[WARN] Error getting validation samples from model: {e}')
    
    # 如果样本ID数量不匹配，生成默认ID
    if len(ids) != n_samples_val and n_samples_val > 0:
        if len(ids) == 0:
            ids = [f'Val_Sample_{i+1}' for i in range(n_samples_val)]
        elif len(ids) < n_samples_val:
            # 补充缺失的ID
            for i in range(len(ids), n_samples_val):
                ids.append(f'Val_Sample_{i+1}')
        else:
            # 截断多余的ID
            ids = ids[:n_samples_val]
    
    if n_samples_val == 0:
        print('[ERROR] Cannot write validation labels file: n_samples_val is 0. No data available.')
    else:
        try:
            with open(out_labels_val, 'w') as f:
                f.write('sample_id\tlabel\tproba_0\tdays\tevent\n')  # 写入表头
                # 处理 survival_test 的格式（可能是转置的矩阵）
                survival_test_arr = np.asarray(boosting.survival_test)
                # 如果形状是 (2, n_samples)，需要转置为 (n_samples, 2)
                if survival_test_arr.ndim == 2 and survival_test_arr.shape[0] == 2:
                    survival_test_arr = survival_test_arr.T
                elif survival_test_arr.ndim == 1:
                    # 如果是一维数组，假设只有days，补充event列
                    survival_test_arr = np.column_stack([survival_test_arr, np.zeros(len(survival_test_arr))])
                # 确保形状是 (n_samples_val, 2)
                if survival_test_arr.shape[0] != n_samples_val:
                    if survival_test_arr.shape[1] == n_samples_val:
                        survival_test_arr = survival_test_arr.T
                
                for i in range(n_samples_val):
                    sid = ids[i] if i < len(ids) else f'Val_Sample_{i+1}'
                    lbl = boosting.test_labels[i] if hasattr(boosting, 'test_labels') and boosting.test_labels is not None and i < len(boosting.test_labels) else 0
                    proba0 = boosting.test_labels_proba[i,0] if hasattr(boosting, 'test_labels_proba') and boosting.test_labels_proba is not None and i < len(boosting.test_labels_proba) else 0.0
                    day = survival_test_arr[i, 0] if i < survival_test_arr.shape[0] else 0.0
                    ev = survival_test_arr[i, 1] if i < survival_test_arr.shape[0] and survival_test_arr.shape[1] > 1 else 0.0
                    f.write(f"{sid}\t{_scalar(lbl)}\t{_scalar(proba0)}\t{_scalar(day)}\t{_scalar(ev)}\n")
            print(f'[SUCCESS] Written {n_samples_val} labels/proba/days/events to', out_labels_val)
        except Exception as e:
            print('[ERROR] Cannot write external_validation_val_test_labels.tsv:', e)
            import traceback
            traceback.print_exc()

    # persist metrics to disk (save in experiment directory)
    metrics_path = os.path.join(exp_dir, 'metrics.tsv')
    with open(metrics_path, 'w') as f:
        f.write('metric\tvalue\n')
        f.write(f'train_pvalue_full\t{pv_full}\n')
        f.write(f'train_cindex_full\t{cidx_full}\n')
        f.write(f'val_pvalue\t{pv_val}\n')
        f.write(f'val_cindex\t{cidx_val}\n')
    print('metrics written:', metrics_path)
    
    # append to history metrics file (keep in root directory for all experiments)
    out_dir = os.path.join('data','integrated','external_validation')
    os.makedirs(out_dir, exist_ok=True)
    metrics_history_path = os.path.join(out_dir, 'metrics_history.tsv')
    
    # Initialize header if file doesn't exist
    if not os.path.exists(metrics_history_path):
        with open(metrics_history_path, 'w') as f:
            f.write('timestamp\tseed\ttrain_pvalue\ttrain_cindex\tval_pvalue\tval_cindex\n')
            
    # Append current run metrics
    with open(metrics_history_path, 'a') as f:
        f.write(f'{timestamp}\t{args.seed}\t{pv_full}\t{cidx_full}\t{pv_val}\t{cidx_val}\n')
    print('metrics history updated:', metrics_history_path)

if __name__ == '__main__':
    main()
