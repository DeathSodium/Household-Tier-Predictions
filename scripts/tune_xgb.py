import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import os
import joblib

def objective(trial):
    train_path = r"data/processed/split/train_split.csv"
    val_path = r"data/processed/split/val_split.csv"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    from sklearn.preprocessing import LabelEncoder
    cat_features = [col for col in train_df.columns if col.endswith('_mode')]
    for col in cat_features:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col] = val_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        val_df[col] = le.transform(val_df[col].astype(str))
        
    X_train, y_train = train_df.drop(['bag_id', 'label'], axis=1), train_df['label']
    X_val, y_val = val_df.drop(['bag_id', 'label'], axis=1), val_df['label']
    
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'lambda': trial.suggest_float('lambda', 0, 5),
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBClassifier(**param, n_estimators=1000)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    return f1

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    os.makedirs("models/tuning", exist_ok=True)
    joblib.dump(trial.params, "models/tuning/best_xgb_params.joblib")
