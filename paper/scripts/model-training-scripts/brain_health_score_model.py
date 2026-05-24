import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.linear_model import ElasticNet, LogisticRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error


from typing import Dict, Tuple

###############################################################################
# Utilities for Stage 1: Independent Heads for Each Task
###############################################################################

# CLOSED FORM LINEAR REGRESSION VERSION (currently not in use, prone to overfitting)

def fit_linear_regression_closed_form(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve linear regression with closed form:
       w,b = argmin ||y - Xw - b||^2
    by augmenting X with a column of ones and using np.linalg.lstsq.
    Returns:
        w: shape (D,)  - weights
        b: float       - bias (intercept)
    """
    mask = np.isfinite(y)
    if not mask.any():
        print("Warning: no valid data for this regression target.")
        return np.zeros(X.shape[1]), 0.0

    X_valid = X[mask]
    y_valid = y[mask]

    # Augment X with ones for intercept
    X_aug = np.hstack([X_valid, np.ones((X_valid.shape[0], 1))])  # (N, D+1)
    w_aug, *_ = np.linalg.lstsq(X_aug, y_valid, rcond=None)

    w = w_aug[:-1]
    b = w_aug[-1]
    return w, b

def fit_logistic_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a logistic regression for binary classification using scikit-learn's LBFGS.
    Returns:
        w: shape (D,)
        b: float
    """
    mask = np.isfinite(y)
    if not mask.any():
        print("Warning: no valid data for this classification target.")
        return np.zeros(X.shape[1]), 0.0

    X_valid = X[mask]
    y_valid = y[mask]

    clf = LogisticRegression(solver='lbfgs', fit_intercept=True,
                                max_iter=1000, tol=1e-4)
    clf.fit(X_valid, y_valid)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    return w, b


# SOLUTIONS WITH NESTED CROSS-VALIDATION (currently in use, more robust)

def fit_linear_regression_nested_cv(
    X: np.ndarray, 
    y: np.ndarray,
    param_grid: dict = None,
    n_inner_splits: int = 5,
    random_state: int = 42
) -> tuple[np.ndarray, float]:
    """
    Fits an ElasticNet regression to (X, y) using an internal cross-validation
    to select alpha + l1_ratio. After best hyperparams are found, 
    retrains on the entire (X, y) and returns (w, b).

    Example param_grid:
        {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.0, 0.5, 1.0]
        }
    """
    if param_grid is None:
        param_grid = {
            'alpha': [0.05],
            'l1_ratio': [0.5],            
        }
        

    # Filter out rows with NaN in y if desired
    mask = np.isfinite(y)
    if not np.any(mask):
        print("Warning: no valid data for regression target.")
        w = np.zeros(X.shape[1])
        b = 0.0
        return w, b

    X_valid = X[mask]
    y_valid = y[mask]

    # The model we'll do a grid search over
    enet = ElasticNet(max_iter=5000, tol=1e-3, random_state=random_state)

    # Negative MSE as the scoring (so higher = better)
    def neg_mse(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)

    scorer = make_scorer(neg_mse, greater_is_better=True)

    if 0:
        # Cross validation folds inside the grid search
        inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

        gs = GridSearchCV(
            estimator=enet,
            param_grid=param_grid,
            scoring=scorer,
            cv=inner_cv,
            refit=True
        )
        gs.fit(X_valid, y_valid)

        # Now refit best model on full data
        if 1:
            # print best hyperparameters:
            print(f"Regression, best hyperparameters: {gs.best_params_}")
            
        best_enet = gs.best_estimator_
        best_enet.fit(X_valid, y_valid)

        # Extract weights
        w = best_enet.coef_  # shape (D,)
        b = best_enet.intercept_
        
    else:
        alpha = 0.05
        l1_ratio = 0.5
        enet.set_params(alpha=alpha, l1_ratio=l1_ratio)
        enet.fit(X_valid, y_valid)
        w = enet.coef_
        b = enet.intercept_
        
    return w, b


# SOLUTION WITH USING HUBER LOSS AS TRAINING LOSS FUNCTION

def fit_linear_regression_huber_loss(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    l1_ratio: float = 0.5,
    delta_huber: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-3,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    
    # Filter out rows with NaN in y if needed
    mask = np.isfinite(y)
    if not np.any(mask):
        print("Warning: no valid data for regression target.")
        w = np.zeros(X.shape[1])
        b = 0.0
        return w, b

    X_valid = X[mask]
    y_valid = y[mask]

    # 1) Define custom Huber metric for scoring (not for training)
    def huber_loss(y_true, y_pred, delta=1.0):
        err = y_pred - y_true
        abs_err = np.abs(err)
        is_small_err = abs_err <= delta
        squared_loss = 0.5 * err**2
        linear_loss = delta * (abs_err - 0.5 * delta)
        return np.mean(np.where(is_small_err, squared_loss, linear_loss))

    def neg_huber_loss(y_true, y_pred, delta_huber=1.0):
        return -huber_loss(y_true, y_pred, delta=delta_huber)

    huber_scorer = make_scorer(neg_huber_loss, greater_is_better=True)

    do_grid_search = True
    
    if do_grid_search:
        # 2) Use an SGDRegressor that trains with Huber loss + elastic net penalty
        sgd = SGDRegressor(loss='huber', 
                        penalty='elasticnet', 
                        epsilon=delta_huber,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=42)

        param_grid = {
            # 'alpha': [0.01, 0.05, 0.1, 0.25],
            # 'l1_ratio': [0.1, 0.25, 0.5, 0.75]
            # ['constant', 'invscaling', 'adaptive'],
            'alpha': [0.01, 0.05, 0.1, 0.2],
            'l1_ratio': [0.15,0.25, 0.5],
            'learning_rate': ['invscaling'],
            'eta0': [0.01, 0.001],
        }

        # 3) Grid search with custom scoring
        grid_search = GridSearchCV(sgd,
                                param_grid=param_grid,
                                scoring=huber_scorer,  # or 'neg_mean_squared_error'
                                cv=5)
        grid_search.fit(X_valid, y_valid)

        if 0:
            print("Linear regression, best params:", grid_search.best_params_)
            print("Best score (Huber):", grid_search.best_score_)
        
        best_model = grid_search.best_estimator_
    
    else:
        # 4) Train with best hyperparameters
        sgd = SGDRegressor(loss='huber', 
                        penalty='elasticnet', 
                        epsilon=delta_huber,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=42,
                        alpha=alpha,
                        l1_ratio=l1_ratio)
        
        sgd.fit(X_valid, y_valid)
        best_model = sgd

    # 5) Extract weights
    w = best_model.coef_
    b = best_model.intercept_
    
    return w, b


###############################################################################
# Classification with LogisticRegression(penalty='elasticnet') + Nested CV
###############################################################################
def fit_logistic_regression_nested_cv(
    X: np.ndarray, 
    y: np.ndarray,
    param_grid: dict = None,
    n_inner_splits: int = 5,
    random_state: int = 42
) -> tuple[np.ndarray, float]:
    """
    Fits a logistic regression with elastic net penalty to (X, y) using an 
    internal CV to select (C, l1_ratio). Then retrains on the entire (X, y) 
    and returns (w, b).

    Example param_grid:
        {
            'C': [0.01, 0.1, 1.0],
            'l1_ratio': [0.0, 0.5, 1.0]
        }
    """
    if param_grid is None:
        param_grid = {
            'C': [0.05, 0.1, 0.2],
            'l1_ratio': [0.25, 0.5, 0.75],
        }
        
    # Filter out rows with NaN in y if needed
    mask = np.isfinite(y)
    if not np.any(mask):
        print("Warning: no valid data for classification target.")
        w = np.zeros(X.shape[1])
        b = 0.0
        return w, b

    X_valid = X[mask]
    y_valid = y[mask]

    logreg = LogisticRegression(
        penalty='elasticnet',
        solver='saga',   # needed for elasticnet
        max_iter=5000,
        tol=1e-3,
        random_state=random_state
    )

    # We can use e.g. 'roc_auc' or 'accuracy' as scoring
    if 1:
        inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            estimator=logreg,
            param_grid=param_grid,
            scoring='roc_auc',  # or 'accuracy'
            cv=inner_cv,
            refit=True
        )
        gs.fit(X_valid, y_valid)
        
        if 0:
            # print best hyperparameters:
            print(f"Logistic regression, best hyperparameters: {gs.best_params_}")

        # Now refit best model on the entire set
        best_logreg = gs.best_estimator_
        best_logreg.fit(X_valid, y_valid)

        w = best_logreg.coef_[0]  # shape (D,)
        b = best_logreg.intercept_[0]
    else: 
        c = 0.1
        l1_ratio = 0.5
        logreg.set_params(C=c, l1_ratio=l1_ratio)
        logreg.fit(X_valid, y_valid)
        w = logreg.coef_[0]
        b = logreg.intercept_[0]
        
    return w, b

### The training functions return (w, b) for each task, which are stored in the IndependentHeads object.

def predict_linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict with linear model: y_pred = Xw + b"""
    return X @ w + b

def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict logistic probabilities = 1 / (1 + e^(-logits))
    """
    logits = X @ w + b
    return 1.0 / (1.0 + np.exp(-logits))

class IndependentHeads:
    """
    Trains each target independently.
      - regression_heads: name->(w, b) for continuous tasks
      - classif_heads:   name->(w, b) for binary classification tasks
    """
    def __init__(self):
        self.regression_heads: Dict[str, Tuple[np.ndarray, float]] = {}
        self.classif_heads: Dict[str, Tuple[np.ndarray, float]] = {}

    def fit_regression(self, X: np.ndarray, y: np.ndarray, name: str, delta_huber=1.0):
        # w, b = fit_linear_regression_closed_form(X, y) # not used, prone to overfitting
        # w, b = fit_linear_regression_nested_cv(X, y)
        w, b = fit_linear_regression_huber_loss(X, y, delta_huber=delta_huber)
        self.regression_heads[name] = (w, b)

    def fit_classification(self, X: np.ndarray, y: np.ndarray, name: str):
        # w, b = fit_logistic_regression(X, y) # not used, prone to overfitting
        w, b = fit_logistic_regression_nested_cv(X, y)
        self.classif_heads[name] = (w, b)

    def predict_regression(self, X: np.ndarray, name: str) -> np.ndarray:
        w, b = self.regression_heads[name]
        return predict_linear(X, w, b)

    def predict_classification_proba(self, X: np.ndarray, name: str) -> np.ndarray:
        w, b = self.classif_heads[name]
        return predict_logistic(X, w, b)

###############################################################################
# Utilities for Stage 2: Single-Dimension Brain Health Score Aggregator
###############################################################################
def masked_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta=1.0):
    """
    Returns mean Huber loss across valid (finite) points.
    y_pred, y_true: shape (N,)
    """
    mask = torch.isfinite(y_true)
    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    diff = y_pred[mask] - y_true[mask]
    abs_diff = torch.abs(diff)
    is_quad = abs_diff < delta

    # Quadratic region
    loss_quad = 0.5 * diff[is_quad]**2
    # Linear region
    loss_lin = delta * (abs_diff[~is_quad] - 0.5*delta)
    loss_total = torch.cat([loss_quad, loss_lin], dim=0).mean()
    return loss_total

def masked_bce_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    Returns mean BCE across valid points, interpreting y_pred as logits.
    y_true should be 0/1.
    """
    mask = torch.isfinite(y_true)
    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)
    return F.binary_cross_entropy_with_logits(y_pred[mask], y_true[mask])

class BrainHealthAggregator(nn.Module):
    """
    Single linear layer: input_dim -> 1 dimension (the BHS).
    We'll train it with a combined cost:
      - Huber across all regression tasks
      - BCE across all classification tasks (with sign inverted if 'healthy' is positive)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)  # shape (N,)

def _aggregator_loss(
    model: BrainHealthAggregator, 
    Xb: torch.Tensor,
    y_reg_dict: Dict[str, np.ndarray],
    y_cls_dict: Dict[str, np.ndarray],
    cf_weight_regression: float,
    cf_weight_classification: float,
    delta_huber: float
) -> torch.Tensor:
    """
    Combined aggregator cost for a batch (which might be the entire dataset):
      total_loss = cf_weight_regression * mean_reg_loss + cf_weight_classification * mean_cls_loss
    """
    device = Xb.device
    bhs_pred = model(Xb)  # shape (B,)

    # Summation for regression tasks
    reg_losses = []
    for _, arr_np in y_reg_dict.items():
        arr_t = torch.tensor(arr_np, dtype=torch.float32, device=device)
        loss_k = masked_huber_loss(bhs_pred, arr_t, delta=delta_huber)
        reg_losses.append(loss_k)
    if len(reg_losses) > 0:
        reg_loss_mean = torch.mean(torch.stack(reg_losses))
    else:
        reg_loss_mean = torch.tensor(0.0, device=device)

    # Summation for classification tasks (note: invert sign => -bhs_pred)
    cls_losses = []
    for _, arr_np in y_cls_dict.items():
        arr_t = torch.tensor(arr_np, dtype=torch.float32, device=device)
        loss_k = masked_bce_with_logits(-bhs_pred, arr_t)
        cls_losses.append(loss_k)
    if len(cls_losses) > 0:
        cls_loss_mean = torch.mean(torch.stack(cls_losses))
    else:
        cls_loss_mean = torch.tensor(0.0, device=device)

    total_loss = cf_weight_regression * reg_loss_mean \
                 + cf_weight_classification * cls_loss_mean
    return total_loss

def train_brain_health_aggregator(
    df: pd.DataFrame,
    cols_cog: list,
    cols_dx: list,
    cols_covariates: list,
    cols_lhl: list,
    n_epochs: int = 1000,
    lr: float = 1e-4,
    cf_weight_regression: float = 0.7,
    cf_weight_classification: float = 0.3,
    delta_huber: float = 1.0,
    max_no_improve: int = 5,
    max_convergence_tries: int = 5
) -> Tuple[BrainHealthAggregator, torch.Tensor]:
    """
    Trains a single aggregator dimension 'brain_health_score' on the entire 
    dataset (minus the internal 92%/8% split for early stopping).
    Returns the best aggregator (lowest val_loss) and the feature matrix as a torch.Tensor.

    The aggregator's cost is:
        = cf_weight_regression * average( Huber(bhs_pred, each cog target) )
          + cf_weight_classification * average( BCE(-bhs_pred, each dx target) )
    """
    # 1) Build input features
    cols_features = cols_covariates + cols_lhl
    X = df[cols_features].values
    X_torch = torch.tensor(X, dtype=torch.float32)

    # 2) Build ground-truth dicts
    y_reg_dict = {}
    for c in cols_cog:
        vals = df[c + "_true"].values
        y_reg_dict[c] = np.where(pd.notna(vals), vals, np.nan)
    y_cls_dict = {}
    for d in cols_dx:
        vals = df[d + "_true"].values
        y_cls_dict[d] = np.where(pd.notna(vals), vals, np.nan)

    # 3) We'll do multiple tries if not converged
    best_model_overall = None
    best_val_loss_overall = float("inf")
    tries = 0
    converged = False

    while tries < max_convergence_tries and not converged:
        tries += 1

        # New aggregator
        aggregator = BrainHealthAggregator(input_dim=len(cols_features))

        optimizer = optim.AdamW(aggregator.parameters(), lr=lr, weight_decay=0.0)

        # Internal train/val split
        N = len(df)
        idx_all = np.arange(N)
        np.random.shuffle(idx_all)
        split_idx = int(0.92 * N)
        train_idx = idx_all[:split_idx]
        val_idx   = idx_all[split_idx:]

        # Subset for X
        X_train_torch = X_torch[train_idx]
        X_val_torch   = X_torch[val_idx]

        # Subset for y dict
        y_reg_train = {}
        y_reg_val   = {}
        for k, arr_np in y_reg_dict.items():
            y_reg_train[k] = arr_np[train_idx]
            y_reg_val[k]   = arr_np[val_idx]

        y_cls_train = {}
        y_cls_val   = {}
        for k, arr_np in y_cls_dict.items():
            y_cls_train[k] = arr_np[train_idx]
            y_cls_val[k]   = arr_np[val_idx]

        best_val_loss = float("inf")
        best_model = None
        no_improve_count = 0
        learning_rate_adjustment = True

        for epoch in range(n_epochs):
            aggregator.train()
            optimizer.zero_grad()
            train_loss = _aggregator_loss(
                aggregator, X_train_torch,
                y_reg_train, y_cls_train,
                cf_weight_regression, cf_weight_classification, delta_huber
            )
            train_loss.backward()
            optimizer.step()

            aggregator.eval()
            with torch.no_grad():
                val_loss = _aggregator_loss(
                    aggregator, X_val_torch,
                    y_reg_val, y_cls_val,
                    cf_weight_regression, cf_weight_classification, delta_huber
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = BrainHealthAggregator(len(cols_features))
                best_model.load_state_dict(aggregator.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= max_no_improve:
                print(f"[Aggregator Try {tries}] Early stopping epoch={epoch+1}, "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val={best_val_loss:.4f}")
                break

            # Halve LR if no improvement after 2 epochs
            if no_improve_count == 2 and learning_rate_adjustment:
                for g in optimizer.param_groups:
                    g['lr'] = 0.5 * g['lr']
                no_improve_count = 0
                learning_rate_adjustment = False

            if 0:
                if (epoch+1) % 50 == 0 or epoch == 0:
                    print(f"[Aggregator Try {tries}] Epoch {epoch+1}/{n_epochs}, "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Compare vs. global best
        if best_val_loss < best_val_loss_overall:
            best_val_loss_overall = best_val_loss
            best_model_overall = best_model

        # Mark convergence if we trained > 50 epochs
        converged = (epoch > 50)
        if not converged:
            print(f"Aggregator not converged after {epoch+1} epochs, restarting. (try={tries})")

    if best_model_overall is None:
        # Fallback if no good training
        best_model_overall = aggregator

    return best_model_overall, X_torch

###############################################################################
# Stage 3: Full Pipeline + Cross-Validation for Out-of-Fold Predictions
###############################################################################
def cross_val_brain_health_score_model(
    df: pd.DataFrame,
    cols_cog: list,
    cols_dx: list,
    cols_covariates: list,
    cols_lhl: list,
    n_folds: int = 20,
    n_epochs: int = 300,
    lr: float = 1e-3,
    cf_weight_regression: float = 0.7,
    cf_weight_classification: float = 0.3,
    seed: int = 42,
    global_fold_id: int = -1
) -> pd.DataFrame:
    """
    Perform k-fold cross-validation to produce out-of-fold predictions for:
      1) Each regression task (as [taskname]_pred).
      2) Each classification task (probabilities as [taskname]_pred).
      3) The aggregator dimension (as 'brain_health_score').

    Steps for each fold:
      - Split train/test by fold.
      - Fit independent heads (Stage 1) on train set only.
      - Predict on test set => store in df for regression: [taskname]_pred, 
                                          classification: [taskname]_pred.
      - Fit aggregator with internal 92%/8% train/val on the same train fold,
        using Huber for regression tasks and BCE for classification tasks 
        (with sign inversion). 
      - Predict aggregator on test fold => store in df as "brain_health_score".

    Returns
    -------
    df_cv : pd.DataFrame
        A copy of df with out-of-fold predictions stored in columns:
          [task]_pred for regression
          [task]_pred for classification
          brain_health_score for aggregator
    """

    df_cv = df.copy()
    # Initialize columns for out-of-fold predictions
    for col_cog in cols_cog:
        df_cv[col_cog + "_pred"] = np.nan
    for col_dx_task in cols_dx:
        df_cv[col_dx_task + "_pred"] = np.nan
    df_cv["brain_health_score"] = np.nan

    # Build feature matrix for the whole dataset (just for indexing)
    cols_features = cols_covariates + cols_lhl
    X_all = df_cv[cols_features].values

    # Create KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # For each fold, train on train split, predict on test split
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_all), start=1):
        # print(f"\n=== Fold {fold_i}/{n_folds} ===")

        # Split DataFrame
        df_train = df_cv.iloc[train_idx].copy()
        df_test  = df_cv.iloc[test_idx].copy()

        # ------------------ 1) Train Independent Heads ------------------
        # Build X_train, X_test
        X_train = df_train[cols_features].values
        X_test  = df_test[cols_features].values

        # Create + fit the heads
        heads = IndependentHeads()

        if 1:  # can temporarily disable this if you only need brain health score
            # Fit regression tasks
            for col_cog in cols_cog:
                y_train = df_train[col_cog + "_true"].values
                heads.fit_regression(X_train, y_train, col_cog)

            # Fit classification tasks
            for col_dx_task in cols_dx:
                y_train = df_train[col_dx_task + "_true"].values
                heads.fit_classification(X_train, y_train, col_dx_task)

            # Inference => store out-of-fold predictions in df_cv
            #   * For regression: [taskname]_pred_cv
            #   * For classification: [taskname]_prob_cv
            for col_cog in cols_cog:
                y_pred_test = heads.predict_regression(X_test, col_cog)
                df_cv.loc[df_test.index, col_cog + "_pred"] = y_pred_test

            for col_dx_task in cols_dx:
                y_prob_test = heads.predict_classification_proba(X_test, col_dx_task)
                df_cv.loc[df_test.index, col_dx_task + "_pred"] = y_prob_test

            if (global_fold_id == 0) & (fold_i == 1):
                # extract the bias and weights for all of the heads, store 
                # them in a Dataframe and save them to a file
                df_head_weights = pd.DataFrame(columns=["bias"] + list(cols_features))
                for col_cog in cols_cog:
                    w, b = heads.regression_heads[col_cog]
                    df_head_weights.loc[col_cog] = [b] + list(w)
                for col_dx_task in cols_dx:
                    w, b = heads.classif_heads[col_dx_task]
                    df_head_weights.loc[col_dx_task] = [b] + list(w)
                    
                # later with those weights you can do the predictions with:
                # linear regression:
                # y_pred = X_test @ w + b
                # logistic regression:
                # y_pred = 1 / (1 + np.exp(-(X_test @ w + b)))
                
                df_head_weights.to_csv("head_weights.csv")
                


        if 1: # can temporarily disable this if you do not need brain health score
                
            # ------------------ 2) Train Aggregator ------------------
            # We'll reuse the train_brain_health_aggregator, 
            # which does an internal 92%/8% split for early stopping.
            # It's "trained" only on the train fold.
            aggregator, X_torch_train = train_brain_health_aggregator(
                df=df_train,
                cols_cog=cols_cog,
                cols_dx=cols_dx,
                cols_covariates=cols_covariates,
                cols_lhl=cols_lhl,
                n_epochs=n_epochs,
                lr=lr,
                cf_weight_regression=cf_weight_regression,
                cf_weight_classification=cf_weight_classification
            )

            # Inference => aggregator on test fold => store in "brain_health_score_cv"
            aggregator.eval()
            X_test_torch = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                bhs_test = aggregator(X_test_torch).cpu().numpy()
            df_cv.loc[df_test.index, "brain_health_score"] = bhs_test

            if (global_fold_id == 0) & (fold_i == 1):
                # extract the weights and bias of the aggregator and save them
                w, b = aggregator.lin.weight.squeeze().detach().numpy(), aggregator.lin.bias.squeeze().detach().numpy()
                df_head_weights.loc["brain_health_score"] = [b] + list(w)
                df_head_weights.to_csv("head_weights.csv")
                
        else:
            df_cv.loc[df_test.index, "brain_health_score"] = 0.0
            
    return df_cv




def cross_val_brain_health_score_model_foldwise(
    df: pd.DataFrame,
    cols_cog: list,
    cols_dx: list,
    cols_covariates: list,
    cols_lhl: list,
    n_folds: int = 20,
    n_epochs: int = 300,
    lr: float = 1e-3,
    cf_weight_regression: float = 0.7,
    cf_weight_classification: float = 0.3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Perform k-fold cross-validation to produce out-of-fold predictions for:
      1) Each regression task (as [taskname]_pred).
      2) Each classification task (probabilities as [taskname]_pred).
      3) The aggregator dimension (as 'brain_health_score').

    Steps for each fold:
      - Split train/test by fold.
      - Fit independent heads (Stage 1) on train set only.
      - Predict on test set => store in df for regression: [taskname]_pred, 
                                          classification: [taskname]_pred.
      - Fit aggregator with internal 92%/8% train/val on the same train fold,
        using Huber for regression tasks and BCE for classification tasks 
        (with sign inversion). 
      - Predict aggregator on test fold => store in df as "brain_health_score".

    Returns
    -------
    df_cv : pd.DataFrame
        A copy of df with out-of-fold predictions stored in columns:
          [task]_pred for regression
          [task]_pred for classification
          brain_health_score for aggregator
          
    Note:
    This function is similar to cross_val_brain_health_score_model, but it keeps the 
    original folds. This is useful as the LHL latent space representations are fold-specific.
    """

    df_results = pd.DataFrame()

    for fold_id in range(5):
        df_fold = df[df.fold_id == fold_id]
            
        df_results_fold = cross_val_brain_health_score_model(
            df=df_fold,
            cols_cog=cols_cog,
            cols_dx=cols_dx,
            cols_covariates=cols_covariates,
            cols_lhl=cols_lhl,
            n_folds=n_folds,
            n_epochs=n_epochs,
            lr=lr,
            cf_weight_regression=cf_weight_regression,
            cf_weight_classification=cf_weight_classification,
            seed=seed,
            global_fold_id=fold_id
        )
    
        df_results = pd.concat([df_results, df_results_fold])

    df_results = df_results.loc[df.index]
    
    # z-score the brain_health_score
    df_results['brain_health_score'] = (df_results['brain_health_score'] - df_results['brain_health_score'].mean()) / df_results['brain_health_score'].std()
    
    return df_results




###############################################################################
# Stage 3: Full Pipeline + Cross-Validation for Out-of-Fold Predictions
###############################################################################
def train_test_split_brain_health_score_model(
    df: pd.DataFrame,
    fold_id_train: int,
    fold_id_test: int,
    cols_cog: list,
    cols_dx: list,
    cols_covariates: list,
    cols_lhl: list,
    n_epochs: int = 300,
    lr: float = 1e-3,
    cf_weight_regression: float = 0.7,
    cf_weight_classification: float = 0.3,
    delta_huber=1.0,
    seed: int = 42,
    enable_brain_health_score: bool = False
) -> pd.DataFrame:
    """
    For a fixed train/test split, train the brain health score model. Outputs:
      1) Each regression task (as [taskname]_pred).
      2) Each classification task (probabilities as [taskname]_pred).
      3) The aggregator dimension (as 'brain_health_score').

    Steps for each fold:
      - Split train/test by specified fold_id.
      - Fit independent heads (Stage 1) on train set only.
      - Predict on test set => store in df for regression: [taskname]_pred, 
                                          classification: [taskname]_pred.
      - Also: Predict on train set => store in df for regression: [taskname]_pred,
      - Fit aggregator with internal 92%/8% train/val on the same train fold,
        using Huber for regression tasks and BCE for classification tasks 
        (with sign inversion). 
      - Predict aggregator on train and test fold => store in df as "brain_health_score".

    Returns
    -------
    df : pd.DataFrame
        A copy of df with predictions stored in columns:
          [task]_pred for regression
          [task]_pred for classification
          brain_health_score for aggregator
    """

    df = df.copy()
    # Initialize columns for out-of-fold predictions
    for col_cog in cols_cog:
        df[col_cog + "_pred"] = np.nan
    for col_dx_task in cols_dx:
        df[col_dx_task + "_pred"] = np.nan
    df["brain_health_score"] = np.nan

    # Build feature matrix for the whole dataset (just for indexing)
    cols_features = cols_covariates + cols_lhl
    X_all = df[cols_features].values

    train_idx = df[df.fold_id == fold_id_train].index
    test_idx = df[df.fold_id == fold_id_test].index
    
    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    
    # Split DataFrame
    df_train = df.loc[train_idx].copy()
    df_test  = df.loc[test_idx].copy()

    # ------------------ 1) Train Independent Heads ------------------
    # Build X_train, X_test
    X_train = df_train[cols_features].values
    X_test  = df_test[cols_features].values

    # Create + fit the heads
    heads = IndependentHeads()

    if 1:  # can temporarily disable this if you only need brain health score
        # Fit regression tasks
        for col_cog in cols_cog:
            y_train = df_train[col_cog + "_true"].values
            heads.fit_regression(X_train, y_train, col_cog, delta_huber=delta_huber)

        # Fit classification tasks
        for col_dx_task in cols_dx:
            y_train = df_train[col_dx_task + "_true"].values
            heads.fit_classification(X_train, y_train, col_dx_task)

        # Inference
        #   * For regression: [taskname]_pred_cv
        #   * For classification: [taskname]_prob_cv
        for col_cog in cols_cog:
            y_pred_train = heads.predict_regression(X_train, col_cog)
            df.loc[df_train.index, col_cog + "_pred"] = y_pred_train
            y_pred_test = heads.predict_regression(X_test, col_cog)
            df.loc[df_test.index, col_cog + "_pred"] = y_pred_test

        for col_dx_task in cols_dx:
            y_prob_train = heads.predict_classification_proba(X_train, col_dx_task)
            df.loc[df_train.index, col_dx_task + "_pred"] = y_prob_train
            y_prob_test = heads.predict_classification_proba(X_test, col_dx_task)
            df.loc[df_test.index, col_dx_task + "_pred"] = y_prob_test

    if enable_brain_health_score:
        # ------------------ 2) Train Aggregator ------------------
        # We'll reuse the train_brain_health_aggregator, 
        # which does an internal 92%/8% split for early stopping.
        # It's "trained" only on the train fold.
        aggregator, X_torch_train = train_brain_health_aggregator(
            df=df_train,
            cols_cog=cols_cog,
            cols_dx=cols_dx,
            cols_covariates=cols_covariates,
            cols_lhl=cols_lhl,
            n_epochs=n_epochs,
            lr=lr,
            cf_weight_regression=cf_weight_regression,
            cf_weight_classification=cf_weight_classification,
            delta_huber=delta_huber
        )

        # Inference => aggregator on test fold => store in "brain_health_score_cv"
        aggregator.eval()
        
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        with torch.no_grad():
            bhs_train = aggregator(X_train_torch).cpu().numpy()
        df.loc[df_train.index, "brain_health_score"] = bhs_train
        
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            bhs_test = aggregator(X_test_torch).cpu().numpy()
        df.loc[df_test.index, "brain_health_score"] = bhs_test

    else:
        df.loc[df_train.index, "brain_health_score"] = 0.0
        df.loc[df_test.index, "brain_health_score"] = 0.0
            
    return df







from sklearn.metrics import roc_curve, auc

###############################################################################
# Identifying Columns
###############################################################################

def get_prediction_tasks(df: pd.DataFrame):
    """
    Identify which columns correspond to cognition (cog_*) and 
    which columns correspond to dx-tm (dx-tm-*). We also identify 
    LHL features and optional covariates like 'age_z_true'.

    Returns
    -------
    cols_cog : list of str
    cols_dx  : list of str
    cols_lhl : list of str
    cols_covariates : list of str
    """
    # Example defaults (you can adapt to your needs)
    cols_cog = [col.replace('_true', '') for col in df.columns 
                if col.startswith('cog_total') and col.endswith('_true')]
    # A user-defined set of disease columns
    cols_dx = [
        'dx-tm-dementia', 
        'dx-tm-mci', 
        'dx-tm-depression', 
        'dx-tm-bipolar_disorder'
    ]
    
    # Example covariates
    cols_covariates = ['age_z_true']  # or add 'sex' etc.
    
    # LHL features
    cols_lhl = [x for x in df.columns if x.startswith('lhl_')]

    # Make sure _true/_pred exist for these tasks
    for col_task in cols_cog + cols_dx:
        if col_task + '_true' not in df.columns:
            raise ValueError(f"Missing {col_task+'_true'} in the dataframe!")
        if col_task + '_pred' not in df.columns:
            raise ValueError(f"Missing {col_task+'_pred'} in the dataframe!")

    return cols_cog, cols_dx, cols_lhl, cols_covariates


def evaluate_brain_health_score_model(
    df: pd.DataFrame,
    score_column = 'brain_health_score'
):
    """
    Evaluate the brain health score model on the given dataframe.
    """
    
    cols_cog, cols_dx, _, _ = get_prediction_tasks(df)
    
    print('\nPerformance evaluation:')
    for col_cog in cols_cog:
        mask = df[col_cog + '_true'].notna()
        corr = np.corrcoef(df.loc[mask, col_cog + '_true'], df.loc[mask, score_column])[0, 1]
        print(f"R        {col_cog}: {corr:.3f}")
    for col_dx in cols_dx:
        mask = df[col_dx + '_true'].notna()
        roc_auc = auc(*roc_curve(df.loc[mask, col_dx + '_true'], - df.loc[mask, score_column])[0:2])
        print(f"ROC AUC  {col_dx}: {roc_auc:.3f}")
