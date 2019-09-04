import gc
import time
import warnings

from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


warnings.simplefilter(action='ignore', category=FutureWarning)

DIR_INPUT = '../data/input'
DIR_OUTPUT = '../data/output'
DIR_VIZ = '../visualization'


@contextmanager
def timer(title):
    time_init = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - time_init))


# Helper function to one-hot encode categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_cols = list(df.columns)
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original_cols]
    return df, new_cols


# Feature engineering: application_train.csv and application_test.csv
def fe_app_train_test(num_rows=None, nan_as_category=True):
    # Load data and merge
    df = pd.read_csv(f'{DIR_INPUT}/application_train.csv', nrows=num_rows)
    tst_df = pd.read_csv(f'{DIR_INPUT}/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(tst_df)))
    df = df.append(tst_df).reset_index()

    # Categorical feats
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    df, cat_cols = one_hot_encoder(df, nan_as_category=nan_as_category)
    
    # Impute Nan for dummy value 365.243
    df['DAY_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Add new percentage features
    df['DAY_EMPLOYED_PCT'] = df['DAY_EMPLOYED'] / df['DAY_BIRTH']
    df['INCOME_CRED_PCT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CRED']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PCT'] = df['AMT_ANN'] / df['AMT_INCOME_TOTAL']
    del tst_df
    gc.collect()
    return df


# Feature engineering: bureau.csv and bureau_balance.csv
def fe_bur_and_bal(num_rows=None, nan_as_category=True):
    # Load data and one-hot
    bur = pd.read_csv(f'{DIR_INPUT}/bureau.csv', nrows=num_rows)
    bb = pd.read_csv(f'{DIR_INPUT}/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bur, bur_cat = one_hot_encoder(bur, nan_as_category)

    # Aggregate and merge with bureau.csv
    bb_aggregations = {'MOS_BAL': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bur = bur.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bur.drop(columns='SK_ID_BUREAU', inplace=True)
    del bb, bb_agg
    gc.collect()

    # Numeric feats
    num_aggregations = {
        'DAY_CRED': ['min', 'max', 'mean', 'var'],
        'DAY_CRED_OVERDUE': ['max', 'mean'],
        'DAY_CRED_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CRED_MAX_OVERDUE': ['mean'],
        'CNT_CRED_PROLONG': ['sum'],
        'AMT_CRED_SUM': ['max', 'mean', 'sum'],
        'AMT_CRED_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CRED_SUM_OVERDUE': ['mean'],
        'AMT_CRED_SUM_LIMIT': ['mean', 'sum'],
        'DAY_CRED_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANN': ['max', 'mean'],
        'MOS_BAL_MIN': ['min'],
        'MOS_BAL_MAX': ['max'],
        'MOS_BAL_SIZE': ['mean', 'sum']
    }

    # Categorical feats
    cat_aggs = {}
    for cat in bur_cat:
        cat_aggs[cat] = ['mean']
    for cat in bb_cat:
        cat_aggs[cat + "_MEAN"] = ['mean']

    bur_agg = bur.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggs})
    bur_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bur_agg.columns.tolist()])

    # Active credits - numerical aggregations
    actv = bur[bur['CRED_ACTIVE_Active'] == 1]
    actv_agg = actv.groupby('SK_ID_CURR').agg(num_aggregations)
    actv_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in actv_agg.columns.tolist()])
    bur_agg = bur_agg.join(actv_agg, how='left', on='SK_ID_CURR')
    del actv, actv_agg
    gc.collect()

    # Closed credits - numerical aggregations
    clsd = bur[bur['CRED_ACTIVE_Closed'] == 1]
    clsd_agg = clsd.groupby('SK_ID_CURR').agg(num_aggregations)
    clsd_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in clsd_agg.columns.tolist()])
    bur_agg = bur_agg.join(clsd_agg, how='left', on='SK_ID_CURR')
    del clsd, clsd_agg, bur
    gc.collect()
    return bur_agg


# Feature engineering: previous_application.csv
def fe_prev_apps(num_rows=None, nan_as_category=True):
    prv = pd.read_csv(f'{DIR_INPUT}/previous_application.csv', nrows=num_rows)
    prv, cat_cols = one_hot_encoder(prv, nan_as_category=nan_as_category)

    # Impute Nan for dummy value 365.243
    prv['DAY_FST_DRAWING'].replace(365243, np.nan, inplace=True)
    prv['DAY_FST_DUE'].replace(365243, np.nan, inplace=True)
    prv['DAY_LST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prv['DAY_LST_DUE'].replace(365243, np.nan, inplace=True)
    prv['DAY_TERM'].replace(365243, np.nan, inplace=True)

    # Add new value ask / value received percentage feat
    prv['APP_CRED_PCT'] = prv['AMT_APPL'] / prv['AMT_CRED']

    # Numeric feats
    num_aggregations = {
        'AMT_ANN': ['min', 'max', 'mean'],
        'AMT_APPL': ['min', 'max', 'mean'],
        'AMT_CRED': ['min', 'max', 'mean'],
        'APP_CRED_PCT': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PMNT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PMNT': ['min', 'max', 'mean'],
        'DAY_DECISION': ['min', 'max', 'mean'],
        'CNT_PMNT': ['mean', 'sum'],
    }

    # Categorical feats
    cat_aggs = {}
    for cat in cat_cols:
        cat_aggs[cat] = ['mean']

    prv_agg = prv.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggs})
    prv_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prv_agg.columns.tolist()])

    # Approved Applications - Numerical feats
    apvd = prv[prv['NAME_K_STAT_Approved'] == 1]
    apvd_agg = apvd.groupby('SK_ID_CURR').agg(num_aggregations)
    apvd_agg.columns = pd.Index(['APR_' + e[0] + "_" + e[1].upper() for e in apvd_agg.columns.tolist()])
    prv_agg = prv_agg.join(apvd_agg, how='left', on='SK_ID_CURR')

    # Refused Applications - Numerical feats
    rfsd = prv[prv['NAME_K_STAT_Refused'] == 1]
    rfsd_agg = rfsd.groupby('SK_ID_CURR').agg(num_aggregations)
    rfsd_agg.columns = pd.Index(['REF_' + e[0] + "_" + e[1].upper() for e in rfsd_agg.columns.tolist()])
    prv_agg = prv_agg.join(rfsd_agg, how='left', on='SK_ID_CURR')
    del rfsd, rfsd_agg, apvd, apvd_agg, prv
    gc.collect()
    return prv_agg


# Feature engineering: POS_CASH_bal.csv
def fe_pos_cash(num_rows=None, nan_as_category=True):
    posc = pd.read_csv(f'{DIR_INPUT}/POS_CASH_bal.csv', nrows=num_rows)
    posc, cat_cols = one_hot_encoder(posc, nan_as_category=nan_as_category)

    # Agg feats
    aggregations = {
        'MOS_BAL': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    posc_agg = posc.groupby('SK_ID_CURR').agg(aggregations)
    posc_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in posc_agg.columns.tolist()])

    # Add count of posc cash accounts feat
    posc_agg['POS_COUNT'] = posc.groupby('SK_ID_CURR').size()
    del posc
    gc.collect()
    return posc_agg


# Feature engineering: installments_payments.csv
def fe_inst_pay(num_rows=None, nan_as_category=True):
    inst = pd.read_csv(f'{DIR_INPUT}/installments_payments.csv', nrows=num_rows)
    inst, cat_cols = one_hot_encoder(inst, nan_as_category=nan_as_category)

    # Add percentage and difference for each installment feats
    inst['PMNT_PCT'] = inst['AMT_PMNT'] / inst['AMT_INST']
    inst['PMNT_DIFF'] = inst['AMT_INST'] - inst['AMT_PMNT']

    # Add days past/before due
    inst['DPD'] = inst['DAY_ENTRY_PMNT'] - inst['DAY_INST']
    inst['DBD'] = inst['DAY_INST'] - inst['DAY_ENTRY_PMNT']
    inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0)
    inst['DBD'] = inst['DBD'].apply(lambda x: x if x > 0 else 0)

    # Agg feats
    aggregations = {
        'NUM_INST_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PMNT_PCT': ['max', 'mean', 'sum', 'var'],
        'PMNT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INST': ['max', 'mean', 'sum'],
        'AMT_PMNT': ['min', 'max', 'mean', 'sum'],
        'DAY_ENTRY_PMNT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    inst_agg = inst.groupby('SK_ID_CURR').agg(aggregations)
    inst_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])
    
    # Add count of installments accounts feat
    inst_agg['INS_COUNT'] = inst.groupby('SK_ID_CURR').size()
    del inst
    gc.collect()
    return inst_agg


# Feature engineering: credit_card_bal.csv
def fe_cc_bal(num_rows=None, nan_as_category=True):
    cred = pd.read_csv(f'{DIR_INPUT}/credit_card_bal.csv', nrows=num_rows)
    cred, cat_cols = one_hot_encoder(cred, nan_as_category=nan_as_category)

    # Agg feats
    cred.drop(columns=['SK_ID_PREV'], inplace=True)
    cred_agg = cred.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cred_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cred_agg.columns.tolist()])

    # Add count of credit card lines feat
    cred_agg['CC_COUNT'] = cred.groupby('SK_ID_CURR').size()
    del cred
    gc.collect()
    return cred_agg


# Model: LightGBM with KFold
def lgbm_kfold(df, num_folds, submission_file_name='submission.csv', stratified=False):
    # Separate training and test data
    trn_df = df[df['TARGET'].notnull()]
    tst_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(trn_df.shape, tst_df.shape))
    del df
    gc.collect()

    # Setup cross val model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Setup arrays/df for results
    oof_pred = np.zeros(trn_df.shape[0])
    sub_pred = np.zeros(tst_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in trn_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']]

    for n_fold, (train_idx, val_idx) in enumerate(folds.split(trn_df[feats], trn_df['TARGET'])):
        trn_x, trn_y = trn_df[feats].iloc[train_idx], trn_df['TARGET'].iloc[train_idx]
        val_x, val_y = trn_df[feats].iloc[val_idx], trn_df['TARGET'].iloc[val_idx]

        # LightGBM parameters obtained via Bayesian optimization
        # Parameter tuning cred: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=100)

        oof_pred[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_pred += clf.predict_proba(tst_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fld_impt_df = pd.DataFrame()
        fld_impt_df["feature"] = feats
        fld_impt_df["importance"] = clf.feature_importances_
        fld_impt_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fld_impt_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_pred[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(trn_df['TARGET'], oof_pred))
    
    # Output submission file and display feature importance
    tst_df['TARGET'] = sub_pred
    tst_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    disp_feat_impt(feature_importance_df)
    return feature_importance_df


# Helper function to display feature importance
def disp_feat_impt(feat_impt_df):
    cols = \
        feat_impt_df[["feature", "importance"]]\
        .groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_feats = feat_impt_df.loc[feat_impt_df.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_feats.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(f'{DIR_VIZ}/lgbm_importances-01.png')


def main(debug=False):
    num_rows = 10000 if debug else None
    df = fe_app_train_test(num_rows)
    with timer("Feature engineering bur and bur_bal"):
        bur = fe_bur_and_bal(num_rows)
        print("Bureau df shape:", bur.shape)
        df = df.join(bur, how='left', on='SK_ID_CURR')
        del bur
        gc.collect()
    with timer("Feature engineering fe_prev_apps"):
        prv = fe_prev_apps(num_rows)
        print("Previous applications df shape:", prv.shape)
        df = df.join(prv, how='left', on='SK_ID_CURR')
        del prv
        gc.collect()
    with timer("Feature engineering POS-CASH bal"):
        posc = fe_pos_cash(num_rows)
        print("Pos-cash bal df shape:", posc.shape)
        df = df.join(posc, how='left', on='SK_ID_CURR')
        del posc
        gc.collect()
    with timer("Feature engineering installments payments"):
        inst = fe_inst_pay(num_rows)
        print("Installments payments df shape:", inst.shape)
        df = df.join(inst, how='left', on='SK_ID_CURR')
        del inst
        gc.collect()
    with timer("Feature engineering credit card bal"):
        cred = fe_cc_bal(num_rows)
        print("Credit card bal df shape:", cred.shape)
        df = df.join(cred, how='left', on='SK_ID_CURR')
        del cred
        gc.collect()
    with timer("Run LightGBM with kfold"):
        lgbm_kfold(df, num_folds=5, submission_file_name="submission.csv", stratified=True)


if __name__ == "__main__":
    with timer("Full model run"):
        main(debug=False)
