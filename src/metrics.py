from typing import Optional, Dict, Union, Any, Callable
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import math

class Error(torchmetrics.Metric):
    """Calculate the error for classification."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        preds = torch.argmax(preds, dim=1)
        self.sum += torch.sum(preds != target)
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the error."""
        return self.sum / self.count


class ClassificationNegativeLogLikelihood(torchmetrics.Metric):
    """Calculate the negative log-likelihood for classification."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        one_hot = F.one_hot(target, num_classes=preds.shape[1]).float()
        self.sum += torch.sum(-one_hot * torch.log(preds + 1e-8))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the negative log-likelihood."""
        return self.sum / self.count


class BrierScore(torchmetrics.Metric):
    """Calculate the Brier score for classification."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        one_hot = F.one_hot(target, num_classes=preds.shape[1]).float()
        self.sum += torch.sum((preds - one_hot) ** 2)
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the Brier score."""
        return self.sum / self.count


class PredictiveEntropy(torchmetrics.Metric):
    """Calculate the predictive entropy for classification."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model.
            target (torch.Tensor): Ground truth values.
        """
        self.sum += torch.sum(-preds * torch.log(preds + 1e-8))
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the predictive entropy."""
        return self.sum / self.count


class RegressionNegativeLogLikelihood(torchmetrics.Metric):
    """Calculate the negative log-likelihood for regression.

    The negative log-likelihood is a proper metric for evaluating the uncertainty
    in regression. It measures the mean log-likelihood of the true outcome.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def _gaussian_nll_loss(self, mean: torch.Tensor, target: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Calculate the negative log-likelihood for a Gaussian distribution.

        Args:
            mean (torch.Tensor): Mean of the Gaussian distribution.
            target (torch.Tensor): Target.
            variance (torch.Tensor): Variance of the Gaussian distribution.
        """
        return torch.sum(0.5 * torch.log(2 * math.pi * variance+1e-8) + (target - mean) ** 2 / (2 * variance+1e-8))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        variance = preds[:, 1] if preds.shape[1] > 1 else torch.ones_like(mean)
        self.sum += self._gaussian_nll_loss(mean.squeeze(), target.squeeze(), variance.squeeze())
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the negative log-likelihood."""
        return self.sum / self.count


class MeanSquaredError(torchmetrics.Metric):
    """Calculate the mean squared error for regression."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        self.sum += F.mse_loss(mean.squeeze(), target.squeeze(), reduction="sum")
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the mean squared error."""
        return self.sum / self.count


class RootMeanSquaredError(MeanSquaredError):
    """Calculate the root mean squared error for regression."""

    def compute(self) -> torch.Tensor:
        """Compute the root mean squared error."""
        return torch.sqrt(self.sum / self.count)


class MeanAbsoluteError(torchmetrics.Metric):
    """Calculate the mean absolute error for regression."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model, mean and or variance.
                                    Mean is assumed to be the first element. Variance
                                    is assumed to be the second element.
            target (torch.Tensor): Ground truth values.
        """
        mean = preds[:, 0]
        self.sum += F.l1_loss(mean.squeeze(), target.squeeze(), reduction="sum")
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the mean absolute error."""
        return self.sum / self.count


METRICS_MAPPING = {
    "error": "Error [$\downarrow$,\%]",
    "ece": "Expected Calibration Error [$\downarrow$,\%]",
    "entropy": "Entropy [nats]",
    "brier": "Brier Score [$\downarrow$]",
    "nll": "Negative LL [$\downarrow$,nats]",
    "mse": "Mean Squared Error [$\downarrow$]",
    "rmse": "Root Mean Squared Error [$\downarrow$]",
    "mae": "Mean Absolute Error [$\downarrow$]",
    "obj": "Objective [$\downarrow$]",
    "main_obj": "Main Objective [$\downarrow$]",
    "kl": "KL Divergence [$\downarrow$]",
}

METRICS_DESIRED_TENDENCY_MAPPING = {
    "error": "down",
    "ece": "down",
    "entropy": "up",
    "brier": "down",
    "nll": "down",
    "mse": "down",
    "rmse": "down",
    "mae": "down",
    "f1": "up",
    "obj": "down",
    "main_obj": "down",
    "kl": "down",
}


class Metric:
    """This is a general metric class that can be used for any task. It is not specific to classification or regression.

    Args:
        output_size (int): The size of the output of the model.
        writer (SummaryWriter): Tensorboard writer.
    """

    metric_labels = ["obj", "main_obj", "kl"]

    def __init__(
        self,
        output_size: int,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.writer = writer
        self.output_size = output_size
        
        self.obj = AverageMeter()
        self.main_obj = AverageMeter()
        self.kl = AverageMeter()

        self.metrics = [self.obj, self.main_obj, self.kl]

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            if hasattr(metric, "reset"):
                metric.reset()
            else:
                raise ValueError(f"Metric {metric} does not have a reset method.")

    def get_metric_value(self, metric: Union[Callable, torchmetrics.Metric]) -> float:
        """Get the value of a metric."""
        val = None
        if hasattr(metric, "avg"):
            val = metric.avg
        elif hasattr(metric, "compute"):
            val = metric.compute()
        else:
            val = metric()
        return val if isinstance(val, float) else val.item()

    def scalar_logging(self, info: str, iteration: int) -> None:
        """Log all metrics to tensorboard if `SummaryWriter` is provided."""
        if self.writer is None:
            return
        for i, metric in enumerate(self.metrics):
            val = self.get_metric_value(metric)
            self.writer.add_scalar(
                info + "/" + METRICS_MAPPING[self.metric_labels[i]], val, iteration
            )

    def get_str(self) -> str:
        """Get a string representation of all metrics."""
        s = ""
        for i, metric in enumerate(self.metrics):
            val = self.get_metric_value(metric)
            s += f"{METRICS_MAPPING[self.metric_labels[i]]}: {str(val)} "
        return s

    def get_packed(self) -> Dict[str, float]:
        """Get a dictionary of all metrics."""
        d = {}
        for i, metric in enumerate(self.metrics):
            val = self.get_metric_value(metric)
            d[self.metric_labels[i].lower()] = val
        return d
    
    def get_key_metric(self) -> float:
        """Get the key metric."""
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def update(
        self,
        obj: Optional[torch.Tensor] = 0.0,
        main_obj: Optional[torch.Tensor] = 0.0,
        kl: Optional[torch.Tensor] = 0.0,
    ) -> None:
        """Update all metrics.

        Args:
            
        """
        for metric, container in zip(
            [obj, main_obj, kl],
            [self.obj, self.main_obj, self.kl],
        ):
            if metric is not None:
                metric = metric if not hasattr(metric, "item") else metric.item()
                container.update(metric, 1)


class ClassificationMetric(Metric):
    """This is a metric class for classification tasks.

    Args:
        output_size (int): Number of output classes.
        writer (SummaryWriter): Tensorboard writer.
    """

    metric_labels = [
        "obj",
        "main_obj",
        "kl",
        "nll",
        "error",
        "entropy",
        "brier",
        "ece",
    ]

    def __init__(
        self,
        output_size: int,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        super(ClassificationMetric, self).__init__(output_size, writer)
        self.entropy = PredictiveEntropy()
        self.ece = torchmetrics.CalibrationError(
            n_bins=10, task="multiclass", norm="l1", num_classes=output_size
        )
        self.nll = ClassificationNegativeLogLikelihood()
        self.brier = BrierScore()
        self.error = Error()

        self.metrics += [
            self.nll,
            self.error,
            self.entropy,
            self.brier,
            self.ece,
        ]

    @torch.no_grad()
    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Union[float, torch.Tensor],
    ) -> None:
        """Update all metrics.

        The output has to be a tensor of shape `(batch_size, output_size)`.

        Args:
            output (torch.Tensor): Model output.
            target (torch.Tensor): Target.
        """
        super(ClassificationMetric, self).update(**kwargs)
        output = output.detach()

        # Check that the metrics are in the right device
        if not self.nll.device == output.device:
            self.nll.to(output.device)
            self.error.to(output.device)
            self.entropy.to(output.device)
            self.brier.to(output.device)
            self.ece.to(output.device)

        self.nll.update(output, target)
        self.error.update(output, target)
        self.entropy.update(output, target)
        self.brier.update(output, target)
        self.ece.update(output, target)
        
    def get_key_metric(self) -> float:
        """Get the key metric."""
        return self.error.compute()


class RegressionMetric(Metric):
    """This is a metric class for regression tasks.

    Args:
        output_size (int): Output size. Not used.
        writer (SummaryWriter): Tensorboard writer.
    """

    metric_labels = [
        "obj",
        "main_obj",
        "kl",
        "nll",
        "rmse",
        "mse",
        "mae",
    ]

    def __init__(
        self,
        output_size: int,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        super(RegressionMetric, self).__init__(
            output_size=output_size, writer=writer
        )

        self.rmse = RootMeanSquaredError()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.nll = RegressionNegativeLogLikelihood()

        self.metrics += [self.nll, self.rmse, self.mse, self.mae]

    @torch.no_grad()
    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Union[float, torch.Tensor],
    ) -> None:
        """Update all metrics.

        The output has to be a tensor of shape `(batch_size, 2)`.

        The first column is the mean and the second column is the variance.

        Args:
            output (torch.Tensor): Model output.
            target (torch.Tensor): Target.
        """
        super(RegressionMetric, self).update(**kwargs)

        mean = output[0].detach()
        var = output[1].detach()
        output = torch.cat([mean.unsqueeze(1), var.unsqueeze(1)], dim=1)

        # Check that the metrics are in the right device
        if not self.nll.device == output.device:
            self.nll.to(output.device)
            self.rmse.to(output.device)
            self.mse.to(output.device)
            self.mae.to(output.device)

        self.nll.update(output, target)
        self.rmse.update(output, target)
        self.mse.update(output, target)
        self.mae.update(output, target)
        
    def get_key_metric(self) -> float:
        """Get the key metric."""
        return self.rmse.compute()

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
