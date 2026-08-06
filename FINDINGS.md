# Findings & Methodology

Companion document to the [README](../README.md). This is the "what we actually learned" write-up: every design decision and the reasoning behind it, the evaluation protocol, what each metric proves and — more importantly — what it doesn't.

**Scope note:** numbers marked `TBD` are filled from `python -m src.evaluation.evaluate_all --config configs/config.yaml`. Nothing in this document reports a metric that wasn't measured.

---

## 1. The design question this project started from

Most published brain-MRI deep learning does one task. A classifier says *glioma* and stops. A segmenter draws an outline and never names it. A clinician reading either output has to go find the other one.

So the question was: **can detection, classification, and segmentation share a single preprocessing pass and a single inference call, without the three tasks degrading each other?**

The answer shaped the architecture: shared preprocessing, three independently-trained heads, unified only at inference. That's a deliberate middle ground between two alternatives:

| Approach | Why not chosen |
| --- | --- |
| **Three fully separate pipelines** | The status quo. Triples preprocessing code, guarantees drift between how each model sees the same image, and makes the deployment story three deployments. |
| **One true multi-task network** (shared encoder, three heads, joint loss) | Elegant, and the obvious "next paper." But it forces a single loss-weighting compromise across three objectives with very different gradient scales — and critically, it would tie the segmentation head (trained on *synthetic* masks, see §5) to the classification head (trained on *real* labels). Contaminating good supervision with bad is a poor trade. |
| **Shared preprocessing, decoupled heads** ← chosen | Each task keeps its own optimizer, schedule, and stopping criterion. Any head can be retrained or replaced without touching the others. The unification cost is paid once, at inference. |

That last point turned out to matter more than expected: because segmentation is the weak link, being able to swap it out later without retraining the classifier is the difference between an afternoon and a week.

---

## 2. Preprocessing — where a surprising amount of the value lives

Three steps, identical for all three tasks:

**Brain-region cropping.** Raw MRI slices carry a large black border. On a 224×224 downsample that border consumes a meaningful fraction of the input while carrying zero signal. Cropping uses the largest external contour and its extreme points to tightly bound the brain.

*Finding:* this is not a cosmetic step. It changes the effective resolution of the anatomy the network sees — the same tumor occupies noticeably more pixels post-crop. Any convolutional model benefits, but a pretrained backbone benefits disproportionately, because ImageNet features expect objects to fill the frame, not sit in a small central patch.

**CLAHE (Contrast Limited Adaptive Histogram Equalization).** Applied after cropping. Global histogram equalization washes out MRI because intensity distributions differ wildly between regions; CLAHE equalizes in tiles with a contrast ceiling, so local tumor–tissue boundaries sharpen without amplifying background noise into artifacts.

*Finding:* qualitatively validated on paired before/after samples — tumor margins that were low-contrast against surrounding white matter become visibly separable. This matters most for the segmentation task, where the decision is made per-pixel at exactly those margins.

**Resize to 224×224 + normalize.** Matches the EfficientNetB3 ImageNet input convention. Keeping the native pretrained resolution avoids re-learning scale statistics.

**Design note:** doing this once and sharing it across all three tasks is the reason the tasks stay comparable. If detection saw CLAHE-enhanced images and segmentation saw raw ones, any performance difference between them would be uninterpretable.

---

## 3. Detection and classification

### Backbone: EfficientNetB3

Chosen over ResNet50/VGG16 and over larger EfficientNets (B5–B7) for a specific reason: **compound scaling gives B3 near-B5 accuracy at roughly a third of the parameters**, which keeps the whole three-model system loadable in a single process on a single consumer GPU. That constraint — *all three models coexist at inference* — was a hard requirement, and it ruled out the bigger backbones.

ImageNet pretraining is doing real work here despite the domain gap. Early conv layers learn edges, textures, and intensity gradients; those transfer to grayscale medical imaging even though the semantic layers do not. ~5,600 training images is far too few to learn those primitives from scratch.

### Freeze-then-fine-tune, and why the warm-up matters

Two phases: **5 epochs with the backbone frozen**, then **up to 64 epochs end-to-end**.

The warm-up is not optional. A randomly-initialized classification head produces large, essentially random gradients on its first passes. Backpropagating those into carefully-pretrained ImageNet weights damages exactly the features you loaded them for. Freezing the backbone lets the head converge to something sane first, so that when the backbone unfreezes, the gradients reaching it are already meaningful.

*This is the single highest-leverage training decision in the classification pipeline* — more than learning rate, more than augmentation.

### Handling class imbalance

`sklearn.utils.class_weight.compute_class_weight('balanced')`, fed to Keras `class_weight`.

Chosen over oversampling (duplicates images → the model memorizes the duplicates → optimistic validation) and over undersampling (discards real data from a dataset that has ~1,400 images per class, which is not a lot). Loss reweighting keeps every image exactly once and shifts only the gradient contribution.

The 4-class split is close to balanced, so this is a modest correction there. It matters much more for **detection**, where collapsing three tumor classes against one non-tumor class creates a genuine ~3:1 imbalance.

### Early stopping on recall, not accuracy — for detection specifically

The detection model's `EarlyStopping` monitors **validation recall**. This is a clinical-cost decision, not a modeling convenience.

The two error types are not symmetric:

| Error | Consequence |
| --- | --- |
| **False positive** (says tumor, isn't) | A radiologist reviews a clean scan. Cost: minutes. |
| **False negative** (says clear, isn't) | A tumor is not flagged. Cost: potentially a delayed diagnosis. |

Accuracy weights those identically. Recall doesn't. On an imbalanced binary problem, accuracy can look excellent while sensitivity quietly degrades, and selecting a checkpoint by accuracy would select for exactly that failure. So the checkpoint is chosen by the metric whose failure mode is the one that actually matters.

Classification, by contrast, monitors `val_accuracy` — there's no asymmetric-cost argument between glioma and meningioma the way there is between "tumor" and "no tumor."

### Optimization

Adam (`lr=1e-4`, weight decay `1e-5`), mixed precision FP16, `ReduceLROnPlateau` (factor 0.5, patience 4), `EarlyStopping` (patience 8).

`1e-4` is deliberately an order of magnitude below the usual Adam default — standard practice when fine-tuning pretrained weights, where the goal is to nudge features rather than relearn them. Dropout 0.4 before the head, which is aggressive, and justified by the small dataset.

Mixed precision roughly doubles throughput and halves activation memory. On a project where three models must fit on one GPU, that's not a nice-to-have.

---

## 4. Segmentation

### Architecture

U-Net (via `segmentation-models-pytorch`) with an **ImageNet-pretrained EfficientNetB3 encoder** and a 5-stage decoder `[256, 128, 64, 32, 16]`; a 1×1 convolution produces a single-channel logit map.

Reusing the same encoder family as the classifiers isn't incidental — it means one architectural family is understood, tuned, and debugged across the whole project. Skip connections carry the high-resolution spatial detail that the encoder's downsampling destroys, which is precisely what boundary-accurate segmentation needs.

### Loss: `0.6 × Dice + 0.4 × BCE-with-logits`

Tumor pixels are a small minority of every slice — often under 5%. That breaks the two obvious loss choices in opposite ways:

- **Pure BCE** — a model that predicts "background" everywhere scores well. The trivial solution is a local optimum, and the model can sit in it.
- **Pure Dice** — directly optimizes region overlap, so it's immune to that. But early in training, when predictions are near-random, Dice gradients are tiny and unstable; the model can fail to get moving at all.

The combination uses each where it's strong: **BCE supplies dense per-pixel gradient early**, when Dice is flat; **Dice dominates the objective** (0.6 weight) once training is underway, so the model optimizes the metric it's actually judged on. The 60/40 split leans toward the target metric while keeping enough BCE to stabilize the start.

### Schedule

32 epochs, Adam `lr=1e-4`, **cosine-annealing LR**, `torch.cuda.amp`, early stopping on validation loss.

Cosine annealing (rather than step decay) suits a short 32-epoch run: it decays smoothly toward zero without the manual milestone tuning step schedules need, so the model takes progressively finer steps as it converges rather than dropping abruptly.

---

## 5. The central finding — segmentation supervision is the bottleneck

**This is the most important thing in this document.**

The Kaggle Brain Tumor MRI dataset provides *class labels only*. It ships **no expert-annotated tumor masks**. Segmentation is supervised learning; supervised learning needs targets. With no radiologist masks available, the pipeline generates **pseudo-masks via Otsu thresholding**.

### What that actually means

Otsu picks an intensity threshold that maximizes between-class variance — it separates "bright" from "dark." It knows nothing about anatomy, tumor biology, or pathology. It is a *global intensity heuristic*.

So when the U-Net trains against Otsu masks and scores well on Dice, the correct reading is:

> **The network has successfully learned to reproduce a thresholding heuristic.**

Not: *the network segments tumors accurately.*

The Dice number is a **pipeline-validity metric**. It confirms the encoder–decoder, the loss, the schedule, and the data pipeline all work — that a segmentation model *can* be trained here and converges to its target. That's genuinely useful engineering evidence. It is not clinical evidence, and reporting it without this caveat would be reporting a number that doesn't mean what it appears to mean.

### The concrete failure modes of Otsu masks

| Failure | Why it happens | Effect on the trained model |
| --- | --- | --- |
| **Over-segmentation of bright non-tumor structure** | Skull remnants, ventricles, and enhancement artifacts can be as bright as tumor | The U-Net learns to label them as tumor, because that's the target it was given |
| **Under-segmentation of low-contrast margins** | Infiltrative tumor edges are near-isointense with surrounding tissue | Boundaries — the clinically decisive part — are systematically pulled inward |
| **CLAHE interaction** | CLAHE alters local intensity distributions, which shifts where Otsu's threshold lands | Mask quality becomes partly a function of a preprocessing choice made for other reasons |
| **Ceiling effect** | The model can at best match Otsu | **Dice against pseudo-masks has a hard ceiling at "as good as thresholding"** — better than that is scored as *worse* |

That last row is the crux. If the U-Net learns something genuinely more anatomically correct than Otsu, the pseudo-mask evaluation *penalizes* it. The metric actively opposes the goal.

### Paths forward, ranked

1. **Real expert masks — BraTS.** The BraTS challenge datasets provide radiologist-annotated multi-sequence volumetric masks. This is the correct fix and makes Dice mean what everyone assumes it means. Cost: different data distribution, volumetric rather than 2D, requires reworking the loader.
2. **Weak supervision from the classifier.** Grad-CAM on the *already-trained, well-supervised* classifier produces attention maps that are semantically grounded — they highlight what the network used to decide *glioma*. Using those as seeds (or as region priors that constrain Otsu) makes the supervision at least tumor-*aware* rather than purely intensity-based. This reuses signal the project already has and is the cheapest meaningful improvement.
3. **Better classical masks.** Region-growing seeded from a CAM peak, or morphological post-processing (largest-connected-component + hole-filling) on Otsu output. Removes the most obvious artifacts. Still a heuristic — a raised ceiling, not a removed one.
4. **Semi-supervised.** Hand-annotate a small held-out subset for *honest evaluation* even if training stays pseudo-supervised. Even 50–100 expert-labeled slices would convert Dice from uninterpretable to meaningful.

**Recommendation:** (2) as the immediate next step, since the classifier is already trained and the machinery is cheap; (1) as the real solution, since nothing else removes the ceiling.

---

## 6. Evaluation protocol

Single script — `src/evaluation/evaluate_all.py` — so every reported number comes from one code path and no metric is computed ad hoc in a notebook.

| Task | Metrics | Why these |
| --- | --- | --- |
| **Detection** | Accuracy, Precision, **Recall**, Specificity, F1, AUC-ROC | Recall is the clinical priority. AUC-ROC is threshold-independent, so it separates "the model ranks well" from "the 0.5 cutoff happens to be right" — and the cutoff is tunable at deployment. |
| **Classification** | Accuracy, macro F1, weighted F1, per-class P/R, confusion matrix | Macro F1 treats all four classes equally, so a rare-class collapse can't hide behind the majority. Weighted F1 reflects the actual test distribution. Reporting both makes the gap between them visible, and that gap is itself diagnostic. |
| **Segmentation** | Dice (DSC), IoU | Dice is the field standard for medical segmentation. IoU is stricter and monotonically related, so a large Dice–IoU gap indicates many partial-overlap predictions rather than clean hits or misses. |

Held-out test set is the dataset's own `Testing/` split — never touched during training or model selection.

### Reading the confusion matrix

The most informative single output is the 4-class confusion matrix, and the pair to watch is **glioma ↔ meningioma**.

They're the known hard pair on this dataset, and the reason is real rather than an artifact: gliomas are intra-axial (arising within brain tissue), meningiomas are extra-axial (arising from the meninges). Distinguishing them properly depends on **location relative to the brain surface and dural attachment** — spatial-relational cues that survive downsampling to 224×224 poorly. Pituitary tumors, by contrast, sit in a stereotyped anatomical location (sella turcica) and are usually the easiest class.

*Expected pattern, to verify against your matrix:* pituitary and notumor should show the highest per-class recall; glioma↔meningioma should hold most of the off-diagonal mass. **If errors are distributed differently than that, something non-obvious is happening and is worth chasing** — that's a more interesting result than the headline accuracy.

### Results

Fill from `evaluate_all` output:

| Task | Metric | Value |
| --- | --- | --- |
| Detection | Accuracy | `TBD` |
| Detection | **Recall** | `TBD` |
| Detection | Precision | `TBD` |
| Detection | Specificity | `TBD` |
| Detection | F1 | `TBD` |
| Detection | AUC-ROC | `TBD` |
| Classification | Accuracy | `TBD` |
| Classification | Macro F1 | `TBD` |
| Classification | Weighted F1 | `TBD` |
| Segmentation | Dice *(pseudo-masks)* | `TBD` |
| Segmentation | IoU *(pseudo-masks)* | `TBD` |

Per-class classification breakdown:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| glioma | `TBD` | `TBD` | `TBD` | ~400 |
| meningioma | `TBD` | `TBD` | `TBD` | ~400 |
| pituitary | `TBD` | `TBD` | `TBD` | ~400 |
| notumor | `TBD` | `TBD` | `TBD` | ~400 |

---

## 7. Engineering findings

**TensorFlow and PyTorch in one process.** TF allocates all available VRAM on first device touch by default, which starves PyTorch when it initializes afterward. The fix is a GPU-setup utility that calls `set_memory_growth` **before any model import**, so TF grows its allocation on demand instead of claiming everything. Import order is load-bearing — a subtle failure that presents as an opaque CUDA OOM from PyTorch even on a GPU with plenty of free memory.

**Mixed precision is a capacity decision, not just a speed one.** FP16 roughly halves activation memory. On a project whose defining constraint is *three models resident simultaneously*, that's what makes the architecture feasible at all.

**Config as the single source of truth.** Paths, hyperparameters, and metric settings all live in `configs/config.yaml`, and every script takes `--config`. This is what makes results reproducible — there's no hyperparameter that exists only in someone's shell history.

**Graceful dataset-path fallback.** The loader tries `data/raw/` first and falls back to `Dataset/`. Small thing, but it's the difference between the repo running on a clean clone and the repo throwing a path error at the first user.

---

## 8. Honest limitations

Restated compactly, because these bound every conclusion above:

1. **Segmentation supervision is synthetic** (§5). The Dice score validates the pipeline, not clinical accuracy. This is the dominant limitation.
2. **2D slices, single sequence.** Real neuro-oncology reads are volumetric and multi-sequence (T1, T1-CE, T2, FLAIR). Through-plane extent is invisible to this model.
3. **No external validation.** Train and test are the same collection, same acquisition distribution. Numbers will drop on a different scanner or site — the only open question is by how much.
4. **Detection labels are derived**, so detection inherits any noise in the 4-class annotations.
5. **No calibration.** Softmax outputs are presented as probabilities but are uncalibrated; modern networks are systematically overconfident. Temperature scaling on a validation split would fix this and hasn't been done.
6. **No explainability yet.** The classifier is a black box to the end user. Grad-CAM is on the roadmap and is cheap — it should be there before anyone trusts a prediction.
7. **Not a medical device.** No regulatory validation, no clinical trial, no intended use in patient care.

---

## 9. What the next iteration should do

In priority order:

1. **Fix segmentation supervision** — Grad-CAM-seeded weak supervision as the fast path, BraTS expert masks as the real one. Everything else is secondary to this.
2. **Add Grad-CAM to the app** — makes the classifier auditable, and doubles as the seed generator for (1). Highest value-per-hour item on the list.
3. **Calibrate** — temperature scaling on a held-out split, so displayed confidences mean something.
4. **Test-time augmentation** — horizontal flips and small rotations at inference, averaged. Cheap, reliably worth a point or two, no retraining.
5. **External validation** — a second, independently-sourced dataset. Nothing else so directly tests whether these numbers generalize.
6. **A proper multi-task network** — shared encoder with three heads, *once segmentation supervision is trustworthy*. Doing it before then would propagate bad supervision into good.

---

*Research and educational project. Not a medical device. No output from this system should inform patient care.*
