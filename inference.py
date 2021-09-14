import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sleap.nn.inference import TopDownPredictor

model_dir = './models/baseline_model.topdown_4'
predictor = TopDownPredictor.from_trained_models(
    confmap_model_path=model_dir
)
from sleap.io.dataset import Labels
from sleap.nn.viz import plot_img, plot_confmaps, plot_peaks, plot_pafs

labels_gt = Labels.load_imjoy(filename='/data/wei/actin-comet-tail/valid/manifest.json')
labels_pr = predictor.predict(labels_gt)


def visualize_example(frame_pred, frame_gt, show_labels=True):
    img = frame_gt.image
    # cms = preds["confmaps"].numpy()[0]
    scale = 1.0
    if img.shape[0] < 512:
        scale = 2.0
    if img.shape[0] < 256:
        scale = 4.0
    fig = plot_img(img, dpi=72 * scale, scale=scale)
    if show_labels:
        for instance_pr, instance_gt in zip(frame_pred.instances, frame_gt.instances):
            pts_gt = np.array([[point.x, point.y] for point in instance_gt.points])
            pts_pr = np.array([[point.x, point.y] for point in instance_pr.points])
            # plot_confmaps(cms, output_scale=cms.shape[0] / img.shape[0])
            plot_peaks(pts_gt, pts_pr, paired=True)
    return fig


print("Saving results...")
os.makedirs(f'{model_dir}/validation', exist_ok=True)
for frame_pred, frame_gt in zip(labels_pr.labeled_frames, labels_gt.labeled_frames):
    figure = visualize_example(frame_pred, frame_gt, True)
    figure.savefig(f'{model_dir}/validation/output_{frame_gt.frame_idx}.png', format="png", pad_inches=0)
    figure = visualize_example(frame_pred, frame_gt, False)
    figure.savefig(f'{model_dir}/validation/input_{frame_gt.frame_idx}.png', format="png", pad_inches=0)