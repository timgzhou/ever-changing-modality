"""Training utilities for EVAN."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from einops import rearrange
from data_utils import create_multimodal_batch
from eurosat_data_utils import print_and_reset_rgb_stats
import wandb


# ---------------------------------------------------------------------------
# Shared training utilities
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 1, eta_min: float = 1e-6):
    """
    Linear warmup (lr*0.1 → lr) for warmup_epochs, then cosine decay to eta_min.
    Call scheduler.step() once per epoch (not per step).
    """
    scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])


def make_criterion(task_config) -> nn.Module:
    """Build loss criterion from TaskConfig."""
    if task_config.multilabel:
        return nn.BCEWithLogitsLoss()
    ignore = getattr(task_config, 'ignore_index', -100)
    return nn.CrossEntropyLoss(ignore_index=ignore)


class TrainMetricAccumulator:
    """
    Accumulates per-batch logits/predictions and computes epoch-level metric.

    Supports classification (accuracy), multilabel (mAP), and segmentation (mIoU).
    """

    def __init__(self, segmentation: bool, multilabel: bool, num_classes: int, ignore_index: int):
        self.segmentation = segmentation
        self.multilabel = multilabel
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self._correct = 0
        self._total = 0
        self._seg_preds = []
        self._seg_labels = []
        self._ml_outputs = []
        self._ml_labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """Register one batch. logits: [B,C] or [B,C,H,W]; labels: [B] or [B,H,W]."""
        if self.segmentation:
            self._seg_preds.append(logits.detach().argmax(dim=1).cpu())
            self._seg_labels.append(labels.detach().cpu())
        elif self.multilabel:
            self._ml_outputs.append(logits.detach().cpu())
            self._ml_labels.append(labels.detach().cpu())
        else:
            _, predicted = logits.max(1)
            self._total += labels.size(0)
            self._correct += predicted.eq(labels).sum().item()

    def compute(self) -> float:
        """Return epoch metric as a percentage."""
        if self.segmentation:
            return compute_miou(
                torch.cat(self._seg_preds), torch.cat(self._seg_labels),
                self.num_classes, ignore_index=self.ignore_index,
            )
        elif self.multilabel:
            return _compute_map(torch.cat(self._ml_outputs), torch.cat(self._ml_labels))
        else:
            return 100.0 * self._correct / max(1, self._total)

    @property
    def metric_name(self) -> str:
        if self.segmentation:
            return "mIoU"
        elif self.multilabel:
            return "mAP"
        return "Acc"


# ---------------------------------------------------------------------------
# Trainer — shared state container for all training types
# ---------------------------------------------------------------------------

class Trainer:
    """
    Thin state container for training. Holds model, optimizer, device, and
    task-level config. Provides concrete training methods for the four training
    modes: supervised, distillation, LP (linear probe), and future semi-supervised/SHOT.

    The old standalone functions (single_modality_training_loop,
    distillation_training_loop, train_classifier_with_frozen_backbone) remain
    as backward-compatible wrappers that construct a Trainer and delegate.
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        task_config,
        *,
        clip_norm: float = 1.0,
        use_wandb: bool = False,
        wandb_prefix: str | None = None,
        warmup_epochs: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.task_config = task_config
        self.clip_norm = clip_norm
        self.use_wandb = use_wandb
        self.wandb_prefix = wandb_prefix
        self.warmup_epochs = warmup_epochs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_accumulator(self) -> TrainMetricAccumulator:
        tc = self.task_config
        return TrainMetricAccumulator(
            segmentation=(tc.task_type == 'segmentation'),
            multilabel=tc.multilabel,
            num_classes=tc.num_classes,
            ignore_index=getattr(tc, 'ignore_index', -100),
        )

    def _step(self, loss, params):
        """Backward + grad clip + optimizer step. Returns grad_norm."""
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self.clip_norm)
        self.optimizer.step()
        return grad_norm

    def _log_step(self, loss: float, grad_norm: float, global_step: int):
        if self.use_wandb and self.wandb_prefix:
            wandb.log({
                f'{self.wandb_prefix}/train_loss': loss,
                f'{self.wandb_prefix}/grad_norm': grad_norm,
                f'{self.wandb_prefix}/step': global_step,
            })

    def _log_epoch(self, epoch: int, train_loss: float, train_metric: float,
                   metric_name: str, lr: float,
                   test_loss=None, test_metric=None,
                   val_loss=None, val_metric=None):
        if self.use_wandb and self.wandb_prefix:
            d = {
                f'{self.wandb_prefix}/train_loss_epoch': train_loss,
                f'{self.wandb_prefix}/train_{metric_name.lower()}': train_metric,
                f'{self.wandb_prefix}/epoch': epoch + 1,
                f'{self.wandb_prefix}/lr': lr,
            }
            if test_metric is not None:
                d[f'{self.wandb_prefix}/eval_loss'] = test_loss
                d[f'{self.wandb_prefix}/eval_{metric_name.lower()}'] = test_metric
            if val_metric is not None:
                d[f'{self.wandb_prefix}/val_loss'] = val_loss
                d[f'{self.wandb_prefix}/val_{metric_name.lower()}'] = val_metric
            wandb.log(d)

    # ------------------------------------------------------------------
    # Supervised fine-tuning (replaces single_modality_training_loop body)
    # ------------------------------------------------------------------

    def train_supervised(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int,
        modalities: tuple[str, ...],
        criterion=None,
        best_checkpoint_path: str | None = None,
        val_per_epoch: int = 1,
        phase_name: str = "Training",
    ) -> tuple:
        """
        Supervised training loop for single or multiple modalities.

        Args:
            modalities: Modalities to feed to the model (e.g. ('s2',) or ('s2', 's1')).
            criterion: Loss fn. If None, built from task_config via make_criterion().
            best_checkpoint_path: If set (requires val_loader), save best-val checkpoint here.
            val_per_epoch: Run val/test every N epochs and always on the last epoch.

        Returns:
            (train_metric, test_metric, best_val_metric, best_val_test_metric)
            best_val_* are None when val_loader is None.
        """
        tc = self.task_config
        if criterion is None:
            criterion = make_criterion(tc)
        modality_bands_dict = tc.modality_bands_dict
        label_key = tc.label_key
        segmentation = (tc.task_type == 'segmentation')
        multilabel = tc.multilabel
        num_classes = tc.num_classes if segmentation else None
        ignore_index = getattr(tc, 'ignore_index', -100)

        scheduler = make_scheduler(self.optimizer, num_epochs, self.warmup_epochs)
        accum = self._make_accumulator()
        mod_str = '+'.join(m.upper() for m in modalities)

        best_val_metric = 0.0 if val_loader is not None else None
        best_val_test_metric = None
        if val_loader is not None:
            assert best_checkpoint_path is not None, \
                "best_checkpoint_path required when val_loader is provided"

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        global_step = 0

        eval_kwargs = dict(
            modality_bands_dict=modality_bands_dict,
            modalities_to_use=modalities,
            multilabel=multilabel,
            label_key=label_key,
            segmentation=segmentation,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        train_metric = test_metric = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            accum.reset()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{num_epochs} [{mod_str}]")
            for batch in pbar:
                labels = batch[label_key].to(self.device)
                if multilabel:
                    labels = labels.float()
                modal_input = create_multimodal_batch(
                    batch, modality_bands_dict=modality_bands_dict, modalities=modalities
                )
                modal_input = {k: v.to(self.device) for k, v in modal_input.items()}

                self.optimizer.zero_grad()
                outputs = self.model(modal_input)
                loss = criterion(outputs, labels)

                grad_norm = self._step(loss, trainable_params)
                accum.update(outputs.detach(), labels.detach())
                train_loss += loss.item()

                pbar_postfix = {'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'}
                if not segmentation and not multilabel:
                    pbar_postfix['acc'] = f'{accum.compute():.2f}%'
                pbar.set_postfix(pbar_postfix)

                self._log_step(loss.item(), float(grad_norm), global_step)
                global_step += 1

            train_loss /= len(train_loader)
            train_metric = accum.compute()
            scheduler.step()
            print_and_reset_rgb_stats()

            do_eval = ((epoch + 1) % val_per_epoch == 0) or (epoch + 1 == num_epochs)
            test_loss = val_loss = test_metric = val_metric = None

            if do_eval:
                test_loss, test_metric = evaluate(
                    self.model, test_loader, criterion, self.device, **eval_kwargs
                )
                if val_loader is not None:
                    val_loss, val_metric = evaluate(
                        self.model, val_loader, criterion, self.device, **eval_kwargs
                    )

                print(f"  Train ({mod_str}): Loss={train_loss:.4f}, {accum.metric_name}={train_metric:.2f}%")
                print(f"  Test  ({mod_str}): Loss={test_loss:.4f}, {accum.metric_name}={test_metric:.2f}%")
                if val_metric is not None:
                    print(f"  Val   ({mod_str}): Loss={val_loss:.4f}, {accum.metric_name}={val_metric:.2f}%")
                    if val_metric > best_val_metric:
                        print(f"    New val record: {val_metric:.2f} > {best_val_metric:.2f} — saving checkpoint")
                        best_val_metric = val_metric
                        best_val_test_metric = test_metric
                        self.model.save_checkpoint(best_checkpoint_path)

            self._log_epoch(epoch, train_loss, train_metric, accum.metric_name,
                            self.optimizer.param_groups[0]['lr'],
                            test_loss, test_metric, val_loss, val_metric)

        return train_metric, test_metric, best_val_metric, best_val_test_metric

    # ------------------------------------------------------------------
    # Knowledge distillation (replaces distillation_training_loop body)
    # ------------------------------------------------------------------

    def train_distillation(
        self,
        train_loader,
        val2_loader,
        test_loader,
        teacher_model,
        num_epochs: int,
        student_modality: str,
        teacher_modality: str,
        student_modality_bands_dict: dict,
        teacher_modality_bands_dict: dict,
        temperature: float = 2.0,
        alpha: float = 1.0,
        distillation_mode: str = 'regular',
        kl_type: str = 'kd',
        best_checkpoint_path: str | None = None,
        student_modalities: tuple | None = None,
        student_pseudo_modalities: list | None = None,
    ) -> tuple:
        """
        Knowledge distillation from teacher (teacher_modality) to student.

        student_modalities: tuple of all student modality names (e.g. ('s1', 's2') for a
            multimodal student). Defaults to (student_modality,) for single-modality students.
            student_modality is the primary modality (used for display / backwards compat).

        Val checkpoint criterion: teacher-agreement on val2 (unlabeled multimodal split).
        teacher-agreement = fraction of samples where student pred == teacher pred.

        Args:
            val2_loader: Unlabeled multimodal val loader for teacher-agreement checkpoint selection.
            best_checkpoint_path: If set (requires val2_loader), save best-agreement checkpoint here.

        Returns:
            (train_metric, test_metric, best_test_metric, best_epoch)
        """
        if student_modalities is None:
            student_modalities = (student_modality,)

        tc = self.task_config
        label_key = tc.label_key
        segmentation = (tc.task_type == 'segmentation')
        multilabel = tc.multilabel
        num_classes = tc.num_classes if segmentation else None
        ignore_index = getattr(tc, 'ignore_index', -100)

        if segmentation:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        elif multilabel:
            ce_criterion = nn.BCEWithLogitsLoss()
        else:
            ce_criterion = nn.CrossEntropyLoss()

        scheduler = make_scheduler(self.optimizer, num_epochs, self.warmup_epochs)
        accum = self._make_accumulator()
        teacher_model.eval()

        best_test_metric = 0.0
        best_epoch = 0
        best_agreement = -1.0
        test_loss = test_metric = 0.0

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        global_step = 0

        student_label = '+'.join(student_modalities).upper()
        eval_kwargs = dict(
            modality_bands_dict=student_modality_bands_dict,
            modalities_to_use=student_modalities,
            multilabel=multilabel,
            label_key=label_key,
            segmentation=segmentation,
            num_classes=num_classes,
            ignore_index=ignore_index,
            pseudo_modalities=student_pseudo_modalities,
        )

        train_metric = test_metric = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader,
                        desc=f"Distill Epoch {epoch+1}/{num_epochs} [{student_label}]")
            for batch in pbar:
                labels = batch[label_key].to(self.device) if alpha < 1.0 or distillation_mode == 'with_guidance' else None
                if multilabel and labels is not None:
                    labels = labels.float()

                student_input = create_multimodal_batch(
                    batch, modality_bands_dict=student_modality_bands_dict,
                    modalities=student_modalities,
                )
                student_input = {k: v.to(self.device) for k, v in student_input.items()}

                teacher_input = create_multimodal_batch(
                    batch, modality_bands_dict=teacher_modality_bands_dict,
                    modalities=(teacher_modality,)
                )
                teacher_input = {k: v.to(self.device) for k, v in teacher_input.items()}

                self.optimizer.zero_grad()

                if distillation_mode == 'feature':
                    with torch.no_grad():
                        teacher_features = teacher_model.evan.forward_features(teacher_input)
                        teacher_feat = teacher_features[teacher_modality]
                    student_features = self.model.evan.forward_features(student_input)
                    student_feat = student_features[student_modality]
                    loss = feature_distillation_loss(student_feat, teacher_feat)
                    with torch.no_grad():
                        if segmentation:
                            student_logits = self.model.segment_from_features(student_features)
                        else:
                            student_logits = self.model.classify_from_features(student_features)
                else:
                    with torch.no_grad():
                        teacher_logits = teacher_model(teacher_input)
                    student_logits = self.model(student_input, pseudo_modalities=student_pseudo_modalities)
                    distill_loss = distillation_loss(
                        student_logits, teacher_logits, temperature,
                        kl_type=kl_type, task_type=tc.task_type,
                    )
                    if distillation_mode == 'with_guidance':
                        raise NotImplementedError("not implemented with guidance")
                    elif alpha < 1.0:
                        raise NotImplementedError("not implemented with guidance")
                    else:
                        loss = distill_loss

                grad_norm = self._step(loss, trainable_params)
                train_loss += loss.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})

                self._log_step(loss.item(), float(grad_norm), global_step)
                global_step += 1

            train_loss /= len(train_loader)
            scheduler.step()

            metric_name = accum.metric_name
            should_eval = (epoch % 2 == 0) or (epoch + 1 == num_epochs)
            if should_eval:
                test_loss, test_metric = evaluate(
                    self.model, test_loader, ce_criterion, self.device, **eval_kwargs
                )
                best_test_metric = max(best_test_metric, test_metric)
                best_epoch = epoch + 1
                print(f"  Train ({student_label}): Loss={train_loss:.4f}")
                print(f"  Test  ({student_label}): Loss={test_loss:.4f}, {metric_name}={test_metric:.2f}%")

            # Val checkpoint: teacher-agreement on val2 (unlabeled)
            if should_eval and val2_loader is not None:
                agreement = _compute_teacher_agreement(
                    self.model, teacher_model, val2_loader, self.device,
                    student_modality_bands_dict, teacher_modality_bands_dict,
                    student_modalities, teacher_modality, label_key,
                    segmentation=segmentation, ignore_index=ignore_index,
                    student_pseudo_modalities=student_pseudo_modalities,
                )
                print(f"  Val2 teacher-agreement: {agreement:.2f}%")
                if agreement > best_agreement:
                    best_agreement = agreement
                    if best_checkpoint_path is not None:
                        print(f"    New val record: {agreement:.2f} > {best_agreement:.2f} — saving checkpoint")
                        self.model.save_checkpoint(best_checkpoint_path)
                if self.use_wandb and self.wandb_prefix:
                    wandb.log({
                        f'{self.wandb_prefix}/val2_teacher_agreement': agreement,
                        f'{self.wandb_prefix}/epoch': epoch + 1,
                    })

            if should_eval:
                self._log_epoch(epoch, train_loss, 0.0, metric_name,
                                self.optimizer.param_groups[0]['lr'],
                                test_loss, test_metric)

        return 0.0, test_metric, best_test_metric, best_epoch, best_agreement

    # ------------------------------------------------------------------
    # Linear probe / LP-FT (replaces train_classifier_with_frozen_backbone body)
    # ------------------------------------------------------------------

    def train_lp(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int,
        modality: str,
        best_checkpoint_path: str | None = None,
    ) -> tuple:
        """
        Train classifier head with frozen backbone.

        Val checkpoint criterion: val1 labeled metric.

        Returns:
            (test_metric, best_val_test_metric)
        """
        tc = self.task_config
        label_key = tc.label_key
        multilabel = tc.multilabel
        modality_bands_dict = tc.modality_bands_dict

        ce_criterion = make_criterion(tc)
        scheduler = make_scheduler(self.optimizer, num_epochs, self.warmup_epochs)

        best_val_metric = 0.0
        best_val_test_metric = 0.0

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        global_step = 0

        eval_kwargs = dict(
            modality_bands_dict=modality_bands_dict,
            modalities_to_use=(modality,),
            multilabel=multilabel,
            label_key=label_key,
        )
        test_metric = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"LP Epoch {epoch+1}/{num_epochs} [{modality.upper()}]")
            for batch in pbar:
                labels = batch[label_key].to(self.device)
                if multilabel:
                    labels = labels.float()
                student_input = create_multimodal_batch(
                    batch, modality_bands_dict=modality_bands_dict, modalities=(modality,)
                )
                student_input = {k: v.to(self.device) for k, v in student_input.items()}

                self.optimizer.zero_grad()
                with torch.no_grad():
                    features = self.model.evan.forward_features(student_input)
                logits = self.model.classify_from_features(features)
                loss = ce_criterion(logits, labels)

                grad_norm = self._step(loss, trainable_params)
                train_loss += loss.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})
                self._log_step(loss.item(), float(grad_norm), global_step)
                global_step += 1

            train_loss /= len(train_loader)
            scheduler.step()

            _, test_metric = evaluate(self.model, test_loader, ce_criterion, self.device, **eval_kwargs)

            val_metric = None
            if val_loader is not None:
                _, val_metric = evaluate(self.model, val_loader, ce_criterion, self.device, **eval_kwargs)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_val_test_metric = test_metric
                    if best_checkpoint_path is not None:
                        self.model.save_checkpoint(best_checkpoint_path)

            metric_name = "mAP" if multilabel else "Acc"
            print(f"  LP Epoch {epoch+1}: Test {metric_name}={test_metric:.2f}%"
                  + (f", Val {metric_name}={val_metric:.2f}% (best={best_val_metric:.2f}%)"
                     if val_metric is not None else ""))

            self._log_epoch(epoch, train_loss, 0.0, metric_name,
                            self.optimizer.param_groups[0]['lr'],
                            None, test_metric, None, val_metric)

        return test_metric, best_val_test_metric

    def train_freematch(
        self,
        train1_loader,
        train2_loader,
        val1_loader,
        test_loader,
        num_epochs: int,
        modality: str,
        *,
        weak_aug,
        strong_aug,
        freematch_state,
        temperature: float = 0.5,
        lambda_u: float = 1.0,
        lambda_e: float = 0.01,
        best_checkpoint_path: str | None = None,
        val_per_epoch: int = 1,
    ) -> tuple:
        """
        FreeMatch semi-supervised training loop.

        Uses train1_loader as labeled source, train2_loader as unlabeled source
        (labels ignored). Checkpoint selection via val1_loader (labeled).

        An epoch = one full pass through train2_loader (unlabeled).
        train1_loader is cycled if shorter.

        Returns:
            (train_metric, test_metric, best_val_metric, best_val_test_metric)
        """
        tc = self.task_config
        label_key = tc.label_key
        segmentation = (tc.task_type == 'segmentation')
        multilabel = tc.multilabel
        num_classes = tc.num_classes
        ignore_index = getattr(tc, 'ignore_index', -100)
        modality_bands_dict = tc.modality_bands_dict

        criterion = make_criterion(tc)
        scheduler = make_scheduler(self.optimizer, num_epochs, self.warmup_epochs)
        accum = self._make_accumulator()
        mod_upper = modality.upper()

        best_val_metric = 0.0 if val1_loader is not None else None
        best_val_test_metric = None
        if val1_loader is not None and best_checkpoint_path is not None:
            pass  # will save best checkpoint

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        global_step = 0

        eval_kwargs = dict(
            modality_bands_dict=modality_bands_dict,
            modalities_to_use=(modality,),
            multilabel=multilabel,
            label_key=label_key,
            segmentation=segmentation,
            num_classes=num_classes if segmentation else None,
            ignore_index=ignore_index,
        )

        train_metric = test_metric = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            accum.reset()
            labeled_iter = iter(train1_loader)
            epoch_sup_loss = 0.0
            epoch_unsup_loss = 0.0
            epoch_ent_loss = 0.0
            epoch_mask_ratio = 0.0
            n_steps = 0

            pbar = tqdm(train2_loader,
                        desc=f"FreeMatch Epoch {epoch+1}/{num_epochs} [{mod_upper}]")

            for unlabeled_batch in pbar:
                # --- Labeled batch (cycle if exhausted) ---
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(train1_loader)
                    labeled_batch = next(labeled_iter)

                labels = labeled_batch[label_key].to(self.device)
                if multilabel:
                    labels = labels.float()

                x_l = create_multimodal_batch(
                    labeled_batch, modality_bands_dict=modality_bands_dict,
                    modalities=(modality,),
                )
                x_l = {k: v.to(self.device) for k, v in x_l.items()}

                # Supervised loss
                logits_l = self.model(x_l)
                sup_loss = criterion(logits_l, labels)
                accum.update(logits_l.detach(), labels.detach())

                # --- Unlabeled batch ---
                x_u = create_multimodal_batch(
                    unlabeled_batch, modality_bands_dict=modality_bands_dict,
                    modalities=(modality,),
                )
                x_u_tensor = x_u[modality].to(self.device)

                # Apply augmentations (kornia, on GPU)
                x_weak = weak_aug(x_u_tensor)
                x_strong = strong_aug(x_u_tensor)

                # Weak forward → pseudo-labels (no grad)
                with torch.no_grad():
                    logits_weak = self.model({modality: x_weak})

                    if multilabel:
                        probs = torch.sigmoid(logits_weak / temperature)  # [B, C]
                        pseudo_labels = (probs > 0.5).float()
                        confidence = torch.abs(probs - 0.5) * 2  # [B, C]
                        max_probs = confidence
                        pred_classes = None
                        probs_for_update = probs
                    elif segmentation:
                        probs = F.softmax(logits_weak / temperature, dim=1)  # [B, C, H, W]
                        max_probs_spatial, pseudo_labels = probs.max(dim=1)  # [B, H, W]
                        # Flatten for EMA update
                        B, C_dim, H, W = probs.shape
                        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C_dim)
                        max_probs_flat = max_probs_spatial.reshape(-1)
                        pred_flat = pseudo_labels.reshape(-1)
                        probs_for_update = probs_flat
                        max_probs = max_probs_flat
                        pred_classes = pred_flat
                    else:
                        probs = F.softmax(logits_weak / temperature, dim=1)  # [B, C]
                        max_probs, pseudo_labels = probs.max(dim=1)  # [B]
                        pred_classes = pseudo_labels
                        probs_for_update = probs

                # Update EMA state + compute mask
                freematch_state.update(probs_for_update, max_probs, pred_classes)
                if segmentation:
                    mask = freematch_state.compute_mask(max_probs_spatial.reshape(-1),
                                                        pseudo_labels.reshape(-1))
                    mask = mask.reshape(max_probs_spatial.shape)  # [B, H, W]
                else:
                    mask = freematch_state.compute_mask(max_probs, pred_classes)

                # Strong forward → unsupervised loss
                logits_strong = self.model({modality: x_strong})

                if multilabel:
                    unsup_loss_raw = F.binary_cross_entropy_with_logits(
                        logits_strong, pseudo_labels, reduction='none',
                    )  # [B, C]
                    unsup_loss = (unsup_loss_raw * mask).mean()
                elif segmentation:
                    unsup_loss_raw = F.cross_entropy(
                        logits_strong, pseudo_labels,
                        ignore_index=ignore_index, reduction='none',
                    )  # [B, H, W]
                    unsup_loss = (unsup_loss_raw * mask).mean()
                else:
                    unsup_loss_raw = F.cross_entropy(
                        logits_strong, pseudo_labels, reduction='none',
                    )  # [B]
                    unsup_loss = (unsup_loss_raw * mask).mean()

                # Entropy fairness loss
                if lambda_e > 0:
                    ent_loss = freematch_state.entropy_loss(
                        logits_strong.detach() if False else logits_strong,
                        mask, segmentation=segmentation,
                    )
                else:
                    ent_loss = torch.tensor(0.0, device=self.device)

                # Total loss
                total_loss = sup_loss + lambda_u * unsup_loss + lambda_e * ent_loss

                self.optimizer.zero_grad()
                grad_norm = self._step(total_loss, trainable_params)

                # Track stats
                mask_ratio = mask.float().mean().item()
                epoch_sup_loss += sup_loss.item()
                epoch_unsup_loss += unsup_loss.item()
                epoch_ent_loss += ent_loss.item()
                epoch_mask_ratio += mask_ratio
                n_steps += 1

                pbar_postfix = {
                    'sup': f'{sup_loss.item():.3f}',
                    'unsup': f'{unsup_loss.item():.3f}',
                    'mask': f'{mask_ratio:.2f}',
                    'tp': f'{freematch_state.time_p.mean().item():.3f}',
                }
                pbar.set_postfix(pbar_postfix)

                if self.use_wandb and self.wandb_prefix:
                    wandb.log({
                        f'{self.wandb_prefix}/sup_loss': sup_loss.item(),
                        f'{self.wandb_prefix}/unsup_loss': unsup_loss.item(),
                        f'{self.wandb_prefix}/ent_loss': ent_loss.item(),
                        f'{self.wandb_prefix}/mask_ratio': mask_ratio,
                        f'{self.wandb_prefix}/time_p': freematch_state.time_p.mean().item(),
                        f'{self.wandb_prefix}/total_loss': total_loss.item(),
                        f'{self.wandb_prefix}/grad_norm': float(grad_norm),
                        f'{self.wandb_prefix}/step': global_step,
                    })
                global_step += 1

            # End of epoch
            n_steps = max(n_steps, 1)
            train_metric = accum.compute()
            scheduler.step()

            print(f"  Epoch {epoch+1}: sup_loss={epoch_sup_loss/n_steps:.4f}, "
                  f"unsup_loss={epoch_unsup_loss/n_steps:.4f}, "
                  f"ent_loss={epoch_ent_loss/n_steps:.4f}, "
                  f"mask_ratio={epoch_mask_ratio/n_steps:.3f}, "
                  f"time_p={freematch_state.time_p.mean().item():.4f}")

            do_eval = ((epoch + 1) % val_per_epoch == 0) or (epoch + 1 == num_epochs)
            test_loss = val_loss = test_metric_ep = val_metric = None

            if do_eval:
                test_loss, test_metric_ep = evaluate(
                    self.model, test_loader, criterion, self.device, **eval_kwargs
                )
                test_metric = test_metric_ep

                if val1_loader is not None:
                    val_loss, val_metric = evaluate(
                        self.model, val1_loader, criterion, self.device, **eval_kwargs
                    )

                metric_name = accum.metric_name
                print(f"  Train ({mod_upper}): {metric_name}={train_metric:.2f}%")
                print(f"  Test  ({mod_upper}): Loss={test_loss:.4f}, {metric_name}={test_metric:.2f}%")
                if val_metric is not None:
                    print(f"  Val   ({mod_upper}): Loss={val_loss:.4f}, {metric_name}={val_metric:.2f}%")
                    if val_metric > best_val_metric:
                        print(f"    New val record: {val_metric:.2f} > {best_val_metric:.2f} — saving checkpoint")
                        best_val_metric = val_metric
                        best_val_test_metric = test_metric
                        if best_checkpoint_path is not None:
                            self.model.save_checkpoint(best_checkpoint_path)

            self._log_epoch(epoch, epoch_sup_loss / n_steps, train_metric,
                            accum.metric_name, self.optimizer.param_groups[0]['lr'],
                            test_loss, test_metric_ep, val_loss, val_metric)

        return train_metric, test_metric, best_val_metric, best_val_test_metric

    def train_mixmatch(
        self,
        train1_loader,
        train2_loader,
        val1_loader,
        test_loader,
        num_epochs: int,
        modality: str,
        *,
        weak_aug,
        K: int = 2,
        temperature: float = 0.5,
        alpha: float = 0.75,
        lambda_u: float = 75.0,
        warmup_epochs: int = 4,
        best_checkpoint_path: str | None = None,
        val_per_epoch: int = 1,
    ) -> tuple:
        """
        MixMatch semi-supervised training loop (Berthelot et al., NeurIPS 2019).

        Uses train1_loader as labeled source, train2_loader as unlabeled source
        (labels ignored). An epoch = one full pass through train2_loader.
        train1_loader is cycled if shorter.

        lambda_u is linearly ramped from 0 to lambda_u per-step over the first
        warmup_epochs epochs (following the original paper's step-level ramp).

        Returns:
            (train_metric, test_metric, best_val_metric, best_val_test_metric)
        """
        import numpy as np

        tc = self.task_config
        label_key = tc.label_key
        segmentation = (tc.task_type == 'segmentation')
        multilabel = tc.multilabel
        num_classes = tc.num_classes
        ignore_index = getattr(tc, 'ignore_index', -100)
        modality_bands_dict = tc.modality_bands_dict

        criterion = make_criterion(tc)
        scheduler = make_scheduler(self.optimizer, num_epochs, self.warmup_epochs)
        accum = self._make_accumulator()
        mod_upper = modality.upper()

        best_val_metric = 0.0 if val1_loader is not None else None
        best_val_test_metric = None

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        global_step = 0

        eval_kwargs = dict(
            modality_bands_dict=modality_bands_dict,
            modalities_to_use=(modality,),
            multilabel=multilabel,
            label_key=label_key,
            segmentation=segmentation,
            num_classes=num_classes if segmentation else None,
            ignore_index=ignore_index,
        )

        train_metric = test_metric = 0.0
        warmup_steps = warmup_epochs * len(train2_loader)

        for epoch in range(num_epochs):
            self.model.train()
            accum.reset()
            labeled_iter = iter(train1_loader)
            epoch_sup_loss = 0.0
            epoch_unsup_loss = 0.0
            epoch_mean_lam = 0.0
            n_steps = 0

            pbar = tqdm(train2_loader,
                        desc=f"MixMatch Epoch {epoch+1}/{num_epochs} [{mod_upper}]")

            for unlabeled_batch in pbar:
                # Lambda_u ramp: 0 → lambda_u linearly over warmup_steps (per step)
                lam_u = lambda_u * min(1.0, global_step / max(1, warmup_steps))

                # --- Labeled batch (cycle if exhausted) ---
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(train1_loader)
                    labeled_batch = next(labeled_iter)

                labels = labeled_batch[label_key].to(self.device)

                x_l = create_multimodal_batch(
                    labeled_batch, modality_bands_dict=modality_bands_dict,
                    modalities=(modality,),
                )
                x_l_tensor = x_l[modality].to(self.device)

                # --- Unlabeled batch ---
                x_u = create_multimodal_batch(
                    unlabeled_batch, modality_bands_dict=modality_bands_dict,
                    modalities=(modality,),
                )
                x_u_tensor = x_u[modality].to(self.device)

                B_l = x_l_tensor.size(0)
                B_u = x_u_tensor.size(0)

                # --- Step 1: K augmented views of unlabeled → average → sharpen ---
                # Capture (view, aug_params) together so we can invert the spatial
                # transform on segmentation prediction maps before averaging.
                views_and_params = []
                for _ in range(K):
                    v = weak_aug(x_u_tensor)
                    views_and_params.append((v, copy.deepcopy(weak_aug._params)))
                views = [v for v, _ in views_and_params]

                self.model.eval()
                with torch.no_grad():
                    if segmentation:
                        # Average probs in original frame, then argmax → hard pseudo-labels.
                        # Re-apply each view's flip to the hard label map so image and
                        # pseudo-label are in the same (augmented) coordinate frame.
                        avg_prob = None
                        for v, aug_params in views_and_params:
                            logits_v = self.model({modality: v})        # [B_u, C, H, W]
                            prob_v = F.softmax(logits_v, dim=1)         # [B_u, C, H, W]
                            # Invert flips manually so this works for any C, not just RGB.
                            for param in reversed(aug_params):
                                bp = param.data['batch_prob'].bool().to(prob_v.device)
                                if 'Horizontal' in param.name:
                                    prob_v[bp] = prob_v[bp].flip(-1)
                                elif 'Vertical' in param.name:
                                    prob_v[bp] = prob_v[bp].flip(-2)
                            avg_prob = prob_v if avg_prob is None else avg_prob + prob_v
                        avg_prob = avg_prob / K                         # [B_u, C, H, W]
                        hard_pseudo = avg_prob.argmax(dim=1)            # [B_u, H, W]
                        # Build one hard-label map per view, flipped to match that view.
                        q_views = []
                        for _, aug_params in views_and_params:
                            lbl = hard_pseudo.clone()
                            for param in aug_params:
                                bp = param.data['batch_prob'].bool().to(lbl.device)
                                if 'Horizontal' in param.name:
                                    lbl[bp] = lbl[bp].flip(-1)
                                elif 'Vertical' in param.name:
                                    lbl[bp] = lbl[bp].flip(-2)
                            q_views.append(lbl)                         # [B_u, H, W]
                        q = q_views
                    elif multilabel:
                        avg_prob = sum(
                            torch.sigmoid(self.model({modality: v})) for v in views
                        ) / K  # [B_u, C]
                        # Sharpen in logit space: sigmoid(logit(p) / T)
                        avg_prob_c = avg_prob.clamp(1e-6, 1 - 1e-6)
                        avg_logit = torch.log(avg_prob_c / (1 - avg_prob_c))
                        q = torch.sigmoid(avg_logit / temperature)  # [B_u, C]
                    else:
                        avg_logits = sum(
                            self.model({modality: v}) for v in views
                        ) / K  # [B_u, C]
                        q = F.softmax(avg_logits / temperature, dim=1)  # [B_u, C]

                self.model.train()
                # --- Step 2: Build labeled targets; MixUp labeled batch ---
                x_l_aug = weak_aug(x_l_tensor)

                if segmentation:
                    H, W = x_l_tensor.shape[2], x_l_tensor.shape[3]
                    # Flip integer label map to match x_l_aug.
                    labels_aug = labels.clone()
                    for param in weak_aug._params:
                        bp = param.data['batch_prob'].bool().to(labels_aug.device)
                        if 'Horizontal' in param.name:
                            labels_aug[bp] = labels_aug[bp].flip(-1)
                        elif 'Vertical' in param.name:
                            labels_aug[bp] = labels_aug[bp].flip(-2)
                    valid = (labels_aug != ignore_index)
                    labels_clipped = labels_aug.clone()
                    labels_clipped[~valid] = 0
                    y_l = F.one_hot(labels_clipped.long(), num_classes).float()  # [B_l, H, W, C]
                    y_l[~valid] = 0.0
                    y_l = y_l.reshape(B_l, H * W, num_classes)  # [B_l, H*W, C]
                elif multilabel:
                    y_l = labels.float()  # [B_l, C]
                else:
                    y_l = F.one_hot(labels.long(), num_classes).float()  # [B_l, C]

                if segmentation:
                    # --- Segmentation: MixUp labeled only; unlabeled uses hard CE ---
                    # MixUp labeled with a random shuffle of itself.
                    lam_vals_l = np.random.beta(alpha, alpha, size=B_l)
                    lam_vals_l = np.maximum(lam_vals_l, 1 - lam_vals_l)
                    mean_lam = float(lam_vals_l.mean())
                    lam_l = torch.tensor(lam_vals_l, dtype=torch.float32, device=self.device)
                    perm_l = torch.randperm(B_l, device=self.device)
                    lam_img = lam_l.view(-1, 1, 1, 1)
                    lam_y   = lam_l.view(-1, 1, 1)
                    mixed_x_l = lam_img * x_l_aug + (1 - lam_img) * x_l_aug[perm_l]
                    mixed_y_l = lam_y   * y_l     + (1 - lam_y)   * y_l[perm_l]

                    # --- Step 5: Forward ---
                    logits_lab = self.model({modality: mixed_x_l})  # [B_l, C, H, W]
                    # Forward each unlabeled view separately (no MixUp).
                    logits_unl_views = [
                        self.model({modality: v}) for v in views
                    ]  # K × [B_u, C, H, W]

                    # --- Step 6: Losses ---
                    # Supervised: soft CE over valid pixels (MixUp targets).
                    log_p_lab = F.log_softmax(logits_lab, dim=1)           # [B_l, C, H, W]
                    log_p_flat = log_p_lab.permute(0, 2, 3, 1).reshape(-1, num_classes)
                    y_lab_flat = mixed_y_l.reshape(-1, num_classes)
                    valid_mask = y_lab_flat.sum(dim=1) > 0.5
                    if valid_mask.sum() > 0:
                        sup_loss = -(y_lab_flat[valid_mask] * log_p_flat[valid_mask]).sum(dim=1).mean()
                    else:
                        sup_loss = torch.tensor(0.0, device=self.device)
                    # Unsupervised: hard CE against per-view pseudo-labels.
                    unsup_loss = torch.tensor(0.0, device=self.device)
                    for logits_v, pseudo_v in zip(logits_unl_views, q_views):
                        unsup_loss = unsup_loss + F.cross_entropy(
                            logits_v, pseudo_v.long(), ignore_index=ignore_index,
                        )
                    unsup_loss = unsup_loss / K
                else:
                    # --- Classification / multilabel: original MixMatch steps 3-6 ---
                    q_list = [q] * K
                    all_x = torch.cat([x_l_aug] + views, dim=0)
                    all_y = torch.cat([y_l] + q_list, dim=0)

                    perm = torch.randperm(all_x.size(0), device=self.device)
                    W_x = all_x[perm]
                    W_y = all_y[perm]

                    N_total = all_x.size(0)
                    lam_vals = np.random.beta(alpha, alpha, size=N_total)
                    lam_vals = np.maximum(lam_vals, 1 - lam_vals)
                    mean_lam = float(lam_vals.mean())
                    lam_x = torch.tensor(lam_vals, dtype=torch.float32, device=self.device)
                    lam_x_img = lam_x.view(-1, *([1] * (all_x.ndim - 1)))
                    lam_y_t   = lam_x.view(-1, *([1] * (all_y.ndim - 1)))

                    mixed_x_l = lam_x_img[:B_l] * x_l_aug + (1 - lam_x_img[:B_l]) * W_x[:B_l]
                    mixed_y_l = lam_y_t[:B_l]   * y_l     + (1 - lam_y_t[:B_l])   * W_y[:B_l]
                    u_views_x = torch.cat(views, dim=0)
                    u_views_y = torch.cat(q_list, dim=0)
                    mixed_x_u = lam_x_img[B_l:] * u_views_x + (1 - lam_x_img[B_l:]) * W_x[B_l:]
                    mixed_y_u = lam_y_t[B_l:]   * u_views_y + (1 - lam_y_t[B_l:])   * W_y[B_l:]

                    mixed_x = torch.cat([mixed_x_l, mixed_x_u], dim=0)
                    mixed_y = torch.cat([mixed_y_l, mixed_y_u], dim=0)

                    logits_all = self.model({modality: mixed_x})
                    logits_lab = logits_all[:B_l]
                    logits_unl = logits_all[B_l:]
                    y_lab = mixed_y[:B_l]
                    y_unl = mixed_y[B_l:]

                    if multilabel:
                        y_lab_c = y_lab.clamp(0.0, 1.0)
                        log_sig = F.logsigmoid(logits_lab)
                        log_1msig = F.logsigmoid(-logits_lab)
                        sup_loss = -(y_lab_c * log_sig + (1 - y_lab_c) * log_1msig).mean()
                        p_unl = torch.sigmoid(logits_unl)
                        unsup_loss = F.mse_loss(p_unl, y_unl.clamp(0.0, 1.0))
                    else:
                        sup_loss = -(y_lab * F.log_softmax(logits_lab, dim=1)).sum(dim=1).mean()
                        p_unl = F.softmax(logits_unl, dim=1)
                        unsup_loss = F.mse_loss(p_unl, y_unl)

                total_loss = sup_loss + lam_u * unsup_loss

                self.optimizer.zero_grad()
                grad_norm = self._step(total_loss, trainable_params)

                # Accumulate train metric using hard labeled logits (before mixing)
                with torch.no_grad():
                    hard_logits = self.model({modality: x_l_aug})
                accum.update(hard_logits.detach(),
                             (labels_aug if segmentation else labels).detach())

                epoch_sup_loss += sup_loss.item()
                epoch_unsup_loss += unsup_loss.item()
                epoch_mean_lam += mean_lam
                n_steps += 1

                pbar.set_postfix({
                    'sup': f'{sup_loss.item():.3f}',
                    'unsup': f'{unsup_loss.item():.3f}',
                    'mixup_lam': f'{mean_lam:.3f}',
                    'lambda_u': f'{lam_u:.1f}',
                })

                if self.use_wandb and self.wandb_prefix:
                    wandb.log({
                        f'{self.wandb_prefix}/sup_loss': sup_loss.item(),
                        f'{self.wandb_prefix}/unsup_loss': unsup_loss.item(),
                        f'{self.wandb_prefix}/lambda_u': lam_u,
                        f'{self.wandb_prefix}/mean_lam': mean_lam,
                        f'{self.wandb_prefix}/total_loss': total_loss.item(),
                        f'{self.wandb_prefix}/grad_norm': float(grad_norm),
                        f'{self.wandb_prefix}/step': global_step,
                    })
                global_step += 1

            # End of epoch
            n_steps = max(n_steps, 1)
            train_metric = accum.compute()
            scheduler.step()

            print(f"  Epoch {epoch+1}: sup_loss={epoch_sup_loss/n_steps:.4f}, "
                  f"unsup_loss={epoch_unsup_loss/n_steps:.4f}, "
                  f"lambda_u={lam_u:.1f}, mean_lam={epoch_mean_lam/n_steps:.3f}")

            do_eval = ((epoch + 1) % val_per_epoch == 0) or (epoch + 1 == num_epochs)
            test_loss = val_loss = test_metric_ep = val_metric = None

            if do_eval:
                test_loss, test_metric_ep = evaluate(
                    self.model, test_loader, criterion, self.device, **eval_kwargs
                )
                test_metric = test_metric_ep

                if val1_loader is not None:
                    val_loss, val_metric = evaluate(
                        self.model, val1_loader, criterion, self.device, **eval_kwargs
                    )

                metric_name = accum.metric_name
                print(f"  Train ({mod_upper}): {metric_name}={train_metric:.2f}%")
                print(f"  Test  ({mod_upper}): Loss={test_loss:.4f}, {metric_name}={test_metric:.2f}%")
                if val_metric is not None:
                    print(f"  Val   ({mod_upper}): Loss={val_loss:.4f}, {metric_name}={val_metric:.2f}%")
                    if val_metric > best_val_metric:
                        print(f"    New val record: {val_metric:.2f} > {best_val_metric:.2f} — saving checkpoint")
                        best_val_metric = val_metric
                        best_val_test_metric = test_metric
                        if best_checkpoint_path is not None:
                            self.model.save_checkpoint(best_checkpoint_path)

            self._log_epoch(epoch, epoch_sup_loss / n_steps, train_metric,
                            accum.metric_name, self.optimizer.param_groups[0]['lr'],
                            test_loss, test_metric_ep, val_loss, val_metric)

        return train_metric, test_metric, best_val_metric, best_val_test_metric

    def train_shot(self, *args, **kwargs):
        raise NotImplementedError


def _compute_teacher_agreement(
    student_model, teacher_model, val_loader, device,
    student_modality_bands_dict, teacher_modality_bands_dict,
    student_modalities, teacher_modality, label_key,
    segmentation: bool = False,
    ignore_index: int = -100,
    student_pseudo_modalities=None,
) -> float:
    """
    Compute fraction of val2 samples where student prediction matches teacher prediction.
    Used as val criterion for distillation checkpoint selection.

    student_modalities: tuple of modality names for the student (may be multimodal).
    """
    if isinstance(student_modalities, str):
        student_modalities = (student_modalities,)
    student_model.eval()
    teacher_model.eval()
    agree = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            student_input = create_multimodal_batch(
                batch, modality_bands_dict=student_modality_bands_dict,
                modalities=student_modalities,
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}
            teacher_input = create_multimodal_batch(
                batch, modality_bands_dict=teacher_modality_bands_dict,
                modalities=(teacher_modality,)
            )
            teacher_input = {k: v.to(device) for k, v in teacher_input.items()}

            student_logits = student_model(student_input, pseudo_modalities=student_pseudo_modalities)
            teacher_logits = teacher_model(teacher_input)

            if segmentation:
                s_preds = student_logits.argmax(dim=1)
                t_preds = teacher_logits.argmax(dim=1)
                # exclude pixels where teacher predicts the ignored class
                valid = t_preds != ignore_index
                agree += (s_preds == t_preds)[valid].sum().item()
                total += valid.sum().item()
            else:
                s_preds = student_logits.argmax(dim=1)
                t_preds = teacher_logits.argmax(dim=1)
                agree += (s_preds == t_preds).sum().item()
                total += s_preds.size(0)

    return 100.0 * agree / max(1, total)


class SequenceProjector(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ffn_factor=4, num_layers=1, output_dim=None):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(embed_dim, output_dim) if output_dim is not None else None

    def forward(self, x):
        x = self.layers(x)
        if self.proj_out is not None:
            x = self.proj_out(x)
        return x


class FullSequenceMAEDecoder(nn.Module):
    """
    MAE decoder that takes the full sequence including mask token positions.

    Unlike SimpleMAEDecoder which expects only unmasked tokens + ids_restore,
    this decoder takes the full sequence where masked positions already contain
    the mask token. Simpler interface for fusion MAE training.
    """

    def __init__(self, embed_dim, num_channels, patch_size, decoder_depth=1, decoder_heads=8, ffn_factor=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Transformer decoder
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        decoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.decoder = TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Linear projection to reconstruct pixels
        self.pred = nn.Linear(embed_dim, patch_size * patch_size * num_channels)

    def forward(self, x):
        """
        Args:
            x: [B, num_patches, embed_dim] - Full sequence with mask tokens at masked positions

        Returns:
            reconstructed patches: [B, num_patches, patch_size^2 * channels]
        """
        x = self.decoder(x)
        return self.pred(x)


def _compute_map(all_outputs, all_labels):
    """Compute mean Average Precision for multilabel classification.

    Args:
        all_outputs: [N, C] float logits
        all_labels:  [N, C] float binary targets

    Returns:
        mAP as a percentage (0-100)
    """
    scores = torch.sigmoid(all_outputs)  # [N, C]
    num_classes = scores.shape[1]
    aps = []
    for c in range(num_classes):
        gt = all_labels[:, c]
        if gt.sum() == 0:
            continue  # skip classes with no positives in this split
        sc = scores[:, c]
        # Sort by descending score
        order = sc.argsort(descending=True)
        gt_sorted = gt[order]
        tp_cumsum = gt_sorted.cumsum(0)
        precision_at_k = tp_cumsum / torch.arange(1, len(gt_sorted) + 1, dtype=torch.float32, device=gt.device)
        ap = (precision_at_k * gt_sorted).sum() / gt.sum()
        aps.append(ap.item())
    return 100.0 * (sum(aps) / len(aps)) if aps else 0.0


def evaluate(model, dataloader, criterion, device, modality_bands_dict,
             modalities_to_use=('rgb',), pseudo_modalities=None, intermediate_projectors=None,
             multilabel=False, label_key='label', segmentation=False, num_classes=None,
             ignore_index=-100):
    """
    Evaluate model on a dataloader.

    Args:
        model: EVAN classifier or EvanSegmenter
        dataloader: DataLoader
        criterion: Loss function
        device: torch device
        modality_bands_dict: Dict mapping modality names to their band tuples or slices
        modalities_to_use: Tuple of modality names to use for evaluation
        pseudo_modalities: Optional list of modalities to hallucinate using sequence projection
        intermediate_projectors: Required if pseudo_modalities is provided
        multilabel: If True, report mAP instead of top-1 accuracy (for BEN-v2 etc.)
        label_key: Key for labels in batch dict ('label' or 'mask')
        segmentation: If True, compute mIoU over [B,H,W] predictions (for PASTIS etc.)
        num_classes: Required when segmentation=True.
        ignore_index: Label value excluded from mIoU (e.g. 19 for PASTIS void_label).
    """
    model.eval()
    if intermediate_projectors is not None:
        intermediate_projectors.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_outputs_list = []
    all_labels_list = []
    all_seg_preds = []
    all_seg_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch[label_key].to(device)
            if multilabel:
                labels = labels.float()

            # Create multi-modal input with specified modalities
            modal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=modalities_to_use
            )
            modal_input = {k: v.to(device) for k, v in modal_input.items()}

            if pseudo_modalities is not None:
                outputs = model(modal_input, pseudo_modalities=pseudo_modalities)
            else:
                outputs = model(modal_input)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if segmentation:
                # outputs: [B, C, H, W]; labels: [B, H, W]
                all_seg_preds.append(outputs.argmax(dim=1).cpu())
                all_seg_labels.append(labels.cpu())
            elif multilabel:
                all_outputs_list.append(outputs.cpu())
                all_labels_list.append(labels.cpu())
            else:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    if segmentation:
        metric = compute_miou(torch.cat(all_seg_preds), torch.cat(all_seg_labels), num_classes, ignore_index=ignore_index)
    elif multilabel:
        metric = _compute_map(torch.cat(all_outputs_list), torch.cat(all_labels_list))
    else:
        metric = 100. * correct / total

    return avg_loss, metric


# ==================== MAE Helper Functions ====================

def random_mask_patches(x, mask_ratio=0.75):
    """
    Randomly mask patches for MAE training.

    Args:
        x: Patch embeddings [B, num_patches, embed_dim]
        mask_ratio: Fraction of patches to mask (default 0.75)

    Returns:
        x_masked: Unmasked patches only [B, num_unmasked, embed_dim]
        mask: Boolean mask [B, num_patches] where True = masked (hidden from encoder)
        ids_restore: Indices to restore original order [B, num_patches]
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    # Random shuffle
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep first len_keep patches
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Create mask: 0 is keep, 1 is masked
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask.bool(), ids_restore


def mae_reconstruction_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Mean over pixels in patch

    # Compute loss only on masked patches
    loss = (loss * mask).sum() / mask.sum()
    return loss


def patchify(imgs, patch_size):
    """
    Convert image to patches for MAE target.

    Args:
        imgs: [B, C, H, W]
        patch_size: Size of each patch

    Returns:
        patches: [B, num_patches, num_pixel_per_patch * C]
    """
    patches = rearrange(imgs, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_size, pw=patch_size)
    return patches

def single_modality_training_loop(model, train_loader, test_loader, device,
                                   modality_bands_dict, criterion, optimizer, num_epochs,
                                   modality, phase_name="Training",
                                   use_wandb=False, wandb_prefix=None, clip_norm=1.0,
                                   multilabel=False, label_key='label',
                                   segmentation=False, num_classes=None,
                                   ignore_index=-100,
                                   val_loader=None, best_checkpoint_path=None,
                                   val_per_epoch=1, warmup_epochs=1):
    """
    Supervised training loop for single-modality EVAN training (Stage 0).

    Backward-compatible wrapper around Trainer.train_supervised.

    Returns:
        (train_metric, test_metric, best_val_metric, best_val_test_metric)
    """
    from data_utils import TaskConfig
    task_config = TaskConfig(
        dataset_name='',
        task_type='segmentation' if segmentation else ('multilabel' if multilabel else 'classification'),
        modality_a=modality,
        modality_b='',
        modality_a_channels=0,
        modality_b_channels=0,
        num_classes=num_classes or 0,
        multilabel=multilabel,
        label_key=label_key,
        modality_bands_dict=modality_bands_dict,
        img_size=0,
        ignore_index=ignore_index,
    )
    trainer = Trainer(
        model, optimizer, device, task_config,
        clip_norm=clip_norm, use_wandb=use_wandb, wandb_prefix=wandb_prefix,
        warmup_epochs=warmup_epochs,
    )
    return trainer.train_supervised(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        modalities=(modality,),
        criterion=criterion,
        best_checkpoint_path=best_checkpoint_path,
        val_per_epoch=val_per_epoch,
        phase_name=phase_name,
    )

def compute_miou(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Compute mean Intersection-over-Union over present classes.

    Args:
        preds:  [B, H, W] long — argmax predictions.
        labels: [B, H, W] long — ground truth class indices.
        num_classes: Number of classes.
        ignore_index: Label value to exclude (default 255).

    Returns:
        mIoU as a percentage (0–100).
    """
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        gt_mask = labels == cls
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return 100.0 * (sum(ious) / len(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Distillation loss functions (shared between train_utils.Trainer and
# baseline/baseline_distillation.py)
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, temperature=2.0,
                      ignore_index=None, labels=None, kl_type="kd",
                      task_type="classification"):
    """
    Distillation loss between student and teacher logits.

    task_type:
        'classification' — softmax KL divergence [B, C]
        'multilabel'     — per-class BCE against sigmoid teacher probs [B, C]
        'segmentation'   — softmax KL divergence on [B, C, H, W]; pass labels+ignore_index to mask void pixels

    kl_type choices (classification/segmentation only):
        'kd'   — standard KD (temperature-scaled softmax, multiplied by T²)
        'ttm'  — teacher-temp only (teacher scaled, student not)
        'wttm' — weighted TTM (weight by teacher confidence)
    """
    if task_type == "segmentation" and student_logits.dim() == 4:
        C = student_logits.shape[1]
        student_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
        teacher_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)
        if ignore_index is not None and labels is not None:
            mask = labels.reshape(-1) != ignore_index
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
    if task_type == "multilabel":
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        return F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    if kl_type == "kd":
        student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
        return F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    elif kl_type == "ttm":
        student_log_soft = F.log_softmax(student_logits, dim=-1)
        return F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    elif kl_type == "wttm":
        gamma = 1.0 / temperature
        student_log_soft = F.log_softmax(student_logits, dim=-1)
        per_sample_kl = F.kl_div(student_log_soft, teacher_soft, reduction="none").sum(dim=-1)
        power_sum = (teacher_soft ** gamma).sum(dim=-1)
        return (power_sum * per_sample_kl).mean()
    else:
        raise ValueError(f"Unknown kl_type: {kl_type!r}")


def feature_distillation_loss(student_features, teacher_features):
    """MSE loss between student and teacher intermediate features (cls + patch tokens)."""
    cls_loss = F.mse_loss(student_features['x_norm_clstoken'], teacher_features['x_norm_clstoken'])
    patch_loss = F.mse_loss(student_features['x_norm_patchtokens'], teacher_features['x_norm_patchtokens'])
    return (cls_loss + patch_loss) / 2
