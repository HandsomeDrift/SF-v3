"""Full 14-metric eval across multiple result directories.

Metrics:
  - Semantic: Img 2/50-way, Vid 2/50-way, CLIP Score, VIFI-Score
  - Pixel:    SSIM, PSNR, Hue-PCC
  - Spatio:   FVD, CTC, DTC, CLIP-PCC, EPE

ViT + CLIP preloaded once. VideoMAE / DINO / open_clip load per call.
"""
import argparse, json, os, time
import numpy as np
import torch
import imageio.v3 as iio

from local_config import get_paths
from models.eval_metrics import (
    load_clip_model, load_vit_model,
    clip_score_only, ssim_score_only, psnr_score_only,
    clip_temporal_consistency, dino_temporal_consistency,
    compute_fvd, compute_epe, hue_pcc, vifi_score, clip_pcc,
    img_classify_metric, video_classify_metric,
)

METRIC_HIGHER_IS_BETTER = {
    'FVD': False, 'EPE': False,
    'SSIM': True, 'PSNR': True, 'Hue-PCC': True,
    'CLIP': True, 'CTC': True, 'DTC': True, 'CLIP-PCC': True,
    'VIFI': True, 'Img-2way': True, 'Img-50way': True,
    'Vid-2way': True, 'Vid-50way': True,
}
METRIC_ORDER = ['FVD','EPE','SSIM','PSNR','Hue-PCC','CLIP','CTC','DTC','CLIP-PCC','VIFI','Img-2way','Img-50way','Vid-2way','Vid-50way']

def load_videos(result_dir, video_ids, n_frames=33):
    videos, missing = [], []
    for vid in video_ids:
        p = os.path.join(result_dir, f'{str(vid).zfill(6)}.mp4')
        if not os.path.exists(p):
            missing.append(vid); continue
        videos.append(iio.imread(p)[:n_frames])
    return (np.stack(videos) if videos else None), missing


def compute_all_metrics(pred, gt, device, tag, preloaded, save_cb=None):
    """Compute 14 metrics with per-block try/except so one failure doesn't kill others.
    save_cb(out): optional callback invoked after each metric step for incremental save.
    """
    import traceback
    (vit_proc, vit_mod), (clip_proc, clip_mod) = preloaded
    out = {}

    def step(label, fn):
        t = time.time()
        try:
            fn()
            el = time.time() - t
            vals = ' '.join(f'{k}={out[k]:.4f}' for k in out if k.startswith(label.split()[0]) or k == label)
            print(f'  [{tag}] {label} done ({el:.1f}s)')
        except Exception as e:
            print(f'  [{tag}] {label} FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
        if save_cb is not None:
            save_cb(out)

    def _fvd():
        out['FVD'] = float(compute_fvd(pred, gt, device=device))
        print(f'    FVD: {out["FVD"]:.4f}')
    def _epe():
        m, _ = compute_epe(pred, gt); out['EPE'] = float(m)
        print(f'    EPE: {out["EPE"]:.4f}')
    def _ctc():
        m, _ = clip_temporal_consistency(pred, device=device, preloaded=(clip_proc, clip_mod))
        out['CTC'] = float(m); print(f'    CTC: {out["CTC"]:.4f}')
    def _per_frame():
        ssim_l, psnr_l, clip_l, img2_l, img50_l = [], [], [], [], []
        for fi in range(pred.shape[1]):
            pf, gf = pred[:, fi], gt[:, fi]
            s, _ = ssim_score_only(pf, gf);  ssim_l.append(s)
            p, _ = psnr_score_only(pf, gf);  psnr_l.append(p)
            c, _ = clip_score_only(pf, gf, device=device, preloaded=(clip_proc, clip_mod)); clip_l.append(c)
            a_list = img_classify_metric(pf, gf, n_way=[2, 50], num_trials=100,
                                         device=device, preloaded=(vit_proc, vit_mod))
            img2_l.append(float(np.mean(a_list[0]))); img50_l.append(float(np.mean(a_list[1])))
        out['SSIM'] = float(np.mean(ssim_l))
        out['PSNR'] = float(np.mean(psnr_l))
        out['CLIP'] = float(np.mean(clip_l))
        out['Img-2way']  = float(np.mean(img2_l))
        out['Img-50way'] = float(np.mean(img50_l))
        print(f'    SSIM={out["SSIM"]:.4f} PSNR={out["PSNR"]:.2f} CLIP={out["CLIP"]:.4f} '
              f'Img-2way={out["Img-2way"]:.4f} Img-50way={out["Img-50way"]:.4f}')
    def _hue():
        m, _ = hue_pcc(pred, gt); out['Hue-PCC'] = float(m); print(f'    Hue-PCC: {out["Hue-PCC"]:.4f}')
    def _dtc():
        m, _ = dino_temporal_consistency(pred, device=device); out['DTC'] = float(m); print(f'    DTC: {out["DTC"]:.4f}')
    _vifi_pv = [None]
    def _vifi():
        r = vifi_score(pred, gt, device=device)
        if len(r) == 3:
            out['VIFI'] = float(r[0]); _vifi_pv[0] = r[2]
        else:
            out['VIFI'] = float(r[0]); _vifi_pv[0] = None
        print(f'    VIFI: {out["VIFI"]:.4f}')
    def _cpcc():
        m, _ = clip_pcc(pred, vifi_per_video=_vifi_pv[0], device=device, preloaded=(clip_proc, clip_mod))
        out['CLIP-PCC'] = float(m); print(f'    CLIP-PCC: {out["CLIP-PCC"]:.4f}')
    def _vid():
        v = video_classify_metric(pred, gt, n_way=[2, 50], num_trials=100,
                                  num_frames=pred.shape[1], device=device)
        out['Vid-2way']  = float(np.mean(v[0]))
        out['Vid-50way'] = float(np.mean(v[1]))
        print(f'    Vid-2way={out["Vid-2way"]:.4f} Vid-50way={out["Vid-50way"]:.4f}')

    step('FVD', _fvd)
    step('EPE', _epe)
    step('CTC', _ctc)
    step('per-frame (SSIM/PSNR/CLIP/Img-2,50way)', _per_frame)
    step('Hue-PCC', _hue)
    step('DTC', _dtc)
    step('VIFI', _vifi)
    step('CLIP-PCC', _cpcc)
    step('Vid-2,50way', _vid)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-jsonpath', required=True)
    ap.add_argument('--result-dir', action='append', required=True)
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--output', default=None)
    ap.add_argument('--n-frames', type=int, default=33)
    args = ap.parse_args()

    experiments = []
    for entry in args.result_dir:
        if '=' not in entry:
            raise SystemExit(f'--result-dir must be NAME=PATH: {entry}')
        name, path = entry.split('=', 1)
        experiments.append((name, path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = get_paths()

    items = json.load(open(args.gt_jsonpath))
    video_ids = sorted(int(os.path.basename(d['video']).split('.')[0]) for d in items)
    print(f'Evaluating {len(video_ids)} videos: {video_ids[0]}-{video_ids[-1]}')

    gt_videos, missing_gt = [], []
    for vid in video_ids:
        p = os.path.join(paths['video_dir'], f'{str(vid).zfill(6)}.mp4')
        if not os.path.exists(p):
            missing_gt.append(vid); continue
        gt_videos.append(iio.imread(p)[:args.n_frames])
    gt = np.stack(gt_videos)
    print(f'GT shape: {gt.shape}, missing: {len(missing_gt)}')

    print('Preloading ViT + CLIP ...')
    preloaded = (load_vit_model(device=device), load_clip_model(device=device))

    results = {}
    def _save():
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    for name, path in experiments:
        print('\n' + '=' * 60)
        print(f'  {name}  (dir={path})')
        print('=' * 60)
        pred, missing = load_videos(path, video_ids, n_frames=args.n_frames)
        if pred is None:
            print(f'  [SKIP] no videos under {path}'); continue
        if missing:
            print(f'  [WARN] {len(missing)} missing videos')
        results[name] = {}
        def _cb(out):
            results[name] = out; _save()
        results[name] = compute_all_metrics(pred, gt, device, tag=name, preloaded=preloaded, save_cb=_cb)
        _save()

        # Save incrementally
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    # Print markdown table (graceful on missing metrics)
    names = list(results.keys())
    print('\n| Experiment | ' + ' | '.join(METRIC_ORDER) + ' |')
    print('|' + '---|' * (len(METRIC_ORDER) + 1))
    for n in names:
        vals = [f'{results[n].get(m, float("nan")):.4f}' if m in results[n] else 'n/a'
                for m in METRIC_ORDER]
        print(f'| {n} | ' + ' | '.join(vals) + ' |')

    if args.output:
        print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
