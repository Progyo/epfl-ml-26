from itertools import combinations

import numpy as np

from implementations import reg_logistic_regression_newton

ALL_COLUMN_NAMES = [
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES",
    "CELLFON3", "LADULT", "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1",
    "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE",
    "HHADULT", "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH", "HLTHPLN1",
    "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4", "BPMEDS", "BLOODCHO",
    "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3", "ASTHNOW", "CHCSCNCR",
    "CHCOCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY", "DIABETE3",
    "DIABAGE2", "SEX", "MARITAL", "EDUCA", "RENTHOM1", "NUMHHOL2",
    "NUMPHON2", "CPDEMO1", "VETERAN3", "EMPLOY1", "CHILDREN", "INCOME2",
    "INTERNET", "WEIGHT2", "HEIGHT3", "PREGNANT", "QLACTLM2", "USEEQUIP",
    "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100",
    "SMOKDAY2", "STOPSMK2", "LASTSMK2", "USENOW3", "ALCDAY5", "AVEDRNK2",
    "DRNK3GE5", "MAXDRNKS", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN",
    "FVORANG", "VEGETAB1", "EXERANY2", "EXRACT11", "EXEROFT1", "EXERHMM1",
    "EXRACT21", "EXEROFT2", "EXERHMM2", "STRENGTH", "LMTJOIN3", "ARTHDIS2",
    "ARTHSOCL", "JOINPAIN", "SEATBELT", "FLUSHOT6", "FLSHTMY2", "IMFVPLAC",
    "PNEUVAC3", "HIVTST6", "HIVTSTD3", "WHRTST10", "PDIABTST", "PREDIAB1",
    "INSULIN", "BLDSUGAR", "FEETCHK2", "DOCTDIAB", "CHKHEMO3", "FEETCHK",
    "EYEEXAM", "DIABEYE", "DIABEDU", "CAREGIV1", "CRGVREL1", "CRGVLNG1",
    "CRGVHRS1", "CRGVPRB1", "CRGVPERS", "CRGVHOUS", "CRGVMST2", "CRGVEXPT",
    "VIDFCLT2", "VIREDIF3", "VIPRFVS2", "VINOCRE2", "VIEYEXM2", "VIINSUR2",
    "VICTRCT4", "VIGLUMA2", "VIMACDG2", "CIMEMLOS", "CDHOUSE", "CDASSIST",
    "CDHELP", "CDSOCIAL", "CDDISCUS", "WTCHSALT", "LONGWTCH", "DRADVISE",
    "ASTHMAGE", "ASATTACK", "ASERVIST", "ASDRVIST", "ASRCHKUP", "ASACTLIM",
    "ASYMPTOM", "ASNOSLEP", "ASTHMED3", "ASINHALR", "HAREHAB1", "STREHAB1",
    "CVDASPRN", "ASPUNSAF", "RLIVPAIN", "RDUCHART", "RDUCSTRK", "ARTTODAY",
    "ARTHWGT", "ARTHEXER", "ARTHEDU", "TETANUS", "HPVADVC2", "HPVADSHT",
    "SHINGLE2", "HADMAM", "HOWLONG", "HADPAP2", "LASTPAP2", "HPVTEST",
    "HPLSTTST", "HADHYST2", "PROFEXAM", "LENGEXAM", "BLDSTOOL", "LSTBLDS3",
    "HADSIGM3", "HADSGCO1", "LASTSIG3", "PCPSAAD2", "PCPSADI1", "PCPSARE1",
    "PSATEST1", "PSATIME", "PCPSARS1", "PCPSADE1", "PCDMDECN", "SCNTMNY1",
    "SCNTMEL1", "SCNTPAID", "SCNTWRK1", "SCNTLPAD", "SCNTLWK1", "SXORIENT",
    "TRNSGNDR", "RCSGENDR", "RCSRLTN2", "CASTHDX2", "CASTHNO2", "EMTSUPRT",
    "LSATISFY", "ADPLEASR", "ADDOWN", "ADSLEEP", "ADENERGY", "ADEAT1",
    "ADFAIL", "ADTHINK", "ADMOVE", "MISTMNT", "ADANXEV", "QSTVER",
    "QSTLANG", "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
    "_CHISPNC", "_CRACE1", "_CPRACE", "_CLLCPWT", "_DUALUSE", "_DUALCOR",
    "_LLCPWT", "_RFHLTH", "_HCVU651", "_RFHYPE5", "_CHOLCHK", "_RFCHOL",
    "_LTASTH1", "_CASTHM1", "_ASTHMS1", "_DRDXAR1", "_PRACE1", "_MRACE1",
    "_HISPANC", "_RACE", "_RACEG21", "_RACEGR3", "_RACE_G1", "_AGEG5YR",
    "_AGE65YR", "_AGE80", "_AGE_G", "HTIN4", "HTM4", "WTKG3", "_BMI5",
    "_BMI5CAT", "_RFBMI5", "_CHLDCNT", "_EDUCAG", "_INCOMG", "_SMOKER3",
    "_RFSMOK3", "DRNKANY5", "DROCDY3_", "_RFBING5", "_DRNKWEK", "_RFDRHV5",
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "_MISFRTN", "_MISVEGN", "_FRTRESP", "_VEGRESP", "_FRUTSUM", "_VEGESUM",
    "_FRTLT1", "_VEGLT1", "_FRT16", "_VEG23", "_FRUITEX", "_VEGETEX",
    "_TOTINDA", "METVL11_", "METVL21_", "MAXVO2_", "FC60_", "ACTIN11_",
    "ACTIN21_", "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "_MINAC11",
    "_MINAC21", "STRFREQ_", "PAMISS1_", "PAMIN11_", "PAMIN21_", "PA1MIN_",
    "PAVIG11_", "PAVIG21_", "PA1VIGM_", "_PACAT1", "_PAINDX1", "_PA150R2",
    "_PA300R2", "_PA30021", "_PASTRNG", "_PAREC1", "_PASTAE1", "_LMTACT1",
    "_LMTWRK1", "_LMTSCL1", "_RFSEAT2", "_RFSEAT3", "_FLSHOT6", "_PNEUMO2",
    "_AIDTST3",
]

DROP = [
    # admin, dates, IDs
    #"_STATE",
      "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "QSTVER", "QSTLANG",
    # phone screening
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT",
    "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT", "NUMHHOL2",
    "NUMPHON2", "CPDEMO1",
    # weights and sampling design
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE", "_DUALCOR",
    "_LLCPWT", "_CLLCPWT", "_CHISPNC", "_CRACE1", "_CPRACE",
    # duplicates
    "WEIGHT2", "HEIGHT3", "HTIN4",
    "_AGEG5YR", "_AGE65YR", "_AGE_G",
    "_PRACE1", "_MRACE1", "_HISPANC", "_RACEG21", "_RACEGR3", "_RACE_G1",
    "EDUCA", "INCOME2",
    "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1", "FRUITJU1",
]

# Columns the width rule of sentinel_codes misreads (or, for computed columns, skips), recoded per CDC's 2015
# codebook instead (codebook_column).
# Computed columns code "don't know / refused / missing" as 9, or as below. The dataset divides out some of the
# codebook's implied decimals: MAXVO2_'s 99900 is 999 here, PAFREQ1_'s 99000 (3 decimals) is 99.
COMPUTED_MISSING = {
    "DROCDY3_": 900, "_DRNKWEK": 99900, "MAXVO2_": 999, "FC60_": 999,
    "PAFREQ1_": 99, "PAFREQ2_": 99, "STRFREQ_": 99,
}
PER_WEEK_OR_MONTH = ["ALCDAY5", "EXEROFT1", "EXEROFT2", "STRENGTH"]  # 1xx = xx per week, 2xx = xx per month, 888 = never
# 1xx/2xx/3xx/4xx = xx per day/week/month/year, 888 = never; FEETCHK2's 555 = no feet
PER_DAY_TO_YEAR = ["BLDSUGAR", "FEETCHK2"]
DURATIONS = ["LONGWTCH"]  # 1xx/2xx/3xx/4xx = xx days/weeks/months/years; 555 = all my life
HOURS_MINUTES = ["EXERHMM1", "EXERHMM2"]  # H:MM, so 130 is 90 minutes
AGES = ["DIABAGE2", "ASTHMAGE"]  # 77 and 88 are ages; ASTHMAGE 97 = 10 or younger; 98/99 = don't know/refused
WEEKLY_HOURS = ["SCNTWRK1", "SCNTLWK1"]  # 77 and 88 are hours; 97 = don't know, 98 = zero, 99 = refused
MONTH_YEAR = ["FLSHTMY2", "HIVTSTD3"]  # MMYYYY; month 77 = unknown, 777777/999999 = don't know/refused
REAL_SEVENS = ["EMPLOY1"]  # 7 = retired, which the width rule ranks with the don't-know codes; only 9 is missing
NOMINAL = ["EXRACT11", "EXRACT21"]  # activity codes, not amounts, but more levels than cat_max_unique

MIN_LEVEL_COUNT = 50  # one-hot levels rarer than this in the training rows get no column of their own

# interaction_pairs: every pair of the N_INTERACTION_COLUMNS columns with the largest |w| in a plain Newton fit
N_INTERACTION_COLUMNS = 40  # 780 pairs
INTERACTION_LAMBDA = 1e-3
INTERACTION_ITERS = 10


def remove_unimportant_columns(x_train, x_test, names, drop, max_nan=0.5):
    # Calculate the percentage of nan entries per column
    nan_fraction = np.isnan(x_train).mean(axis=0)

    # Filter both by either items in drop or by nan percentage
    keep = [i for i, n in enumerate(names) if n not in drop and nan_fraction[i] < max_nan]
    return x_train[:, keep], x_test[:, keep], [names[i] for i in keep]


def is_derived_column(name, uniq):
    # This follows from how derived columns are described in the reference PDF 2015 BRFSS codebook
    if name.startswith("_") or name.endswith("_"):
        return True
    return np.any(uniq != np.round(uniq))


def sentinel_codes(uniq):
    # This generates the sentinel codes that the dataset uses
    width = len(str(int(uniq.max())))
    # A 1-digit 8 is an ordinary answer (CHECKUP1 "never", EMPLOY1 "unable to work"), not "none".
    none_codes = {int("8" * width)} if width > 1 else set()
    return none_codes, {int("7" * width), int("9" * width)}


def codebook_column(name, v, uniq):
    """Raw column v recoded per the codebook constants above as (kind, values), or None if the width rule is right.

    uniq: the column's distinct observed values in train and test.
    """

    # This function handles various exceptions to the missing-values code rule.

    if name in PER_WEEK_OR_MONTH:
        per_month = np.where(v < 200, v % 100 * 30 / 7, v % 100)
        return "continuous", np.where(v == 888, 0, np.where(v < 300, per_month, np.nan))
    if name in PER_DAY_TO_YEAR or name in DURATIONS:
        days = np.select([v < 200, v < 300, v < 400], [1, 7, 30], 365)  # length of the 1xx/2xx/3xx/4xx unit
        # Times per month, or years; 555 has no number, so it is missing like 777 and 999
        amount = v % 100 * 30 / days if name in PER_DAY_TO_YEAR else v % 100 * days / 365
        return "continuous", np.where(v == 888, 0, np.where(v < 500, amount, np.nan))
    if name in HOURS_MINUTES:
        return "continuous", np.where((v == 777) | (v == 999), np.nan, v // 100 * 60 + v % 100)
    if name in AGES:
        age = np.where(v == 97, 10, v) if name == "ASTHMAGE" else v
        return "continuous", np.where(v >= 98, np.nan, age)
    if name in WEEKLY_HOURS:
        return "continuous", np.where((v == 97) | (v == 99), np.nan, np.where(v == 98, 0, v))
    if name in MONTH_YEAR:
        month = np.where(v // 10000 <= 12, v // 10000, 6.5)
        return "continuous", np.where(v >= 777777, np.nan, 2016 - v % 10000 - month / 12)  # years before 2016
    if name in REAL_SEVENS:
        return "categorical", np.where(v == 9, np.nan, v)
    if name in NOMINAL:
        return "categorical", v
    if name in COMPUTED_MISSING:
        return "derived", np.where(v == COMPUTED_MISSING[name], np.nan, v)
    computed = name.startswith("_") or name.endswith("_")
    if computed and uniq.size > 1 and uniq[-1] == 9 and uniq[-2] <= 8 and np.all(uniq == np.round(uniq)):
        return "derived", np.where(v == 9, np.nan, v)
    return None


def recode_sentinels(x_train, x_test, names, cat_max_unique=55, codebook=True):
    """Recode BRFSS answer codes column by column; returns (x_train, x_test, kinds).

    With codebook, the columns codebook_column knows are recoded by it; the others by the width rule of
    sentinel_codes. codebook=False reproduces the width rule alone, as used before the codebook fix.
    """

    # Arguably the most complex function in our code base
    x_train = x_train.copy()
    x_test = x_test.copy()
    kinds = []
    for j, name in enumerate(names):
        train_col, test_col = x_train[:, j], x_test[:, j]

        # Calculate all the unique values in the column
        combined = np.concatenate([train_col, test_col])
        uniq = np.unique(combined[~np.isnan(combined)])

        fixed = codebook_column(name, train_col, uniq) if codebook else None
        # Check if codebook column, if so append the kind and remap the column
        if fixed is not None:
            kinds.append(fixed[0])
            train_col[:], test_col[:] = fixed[1], codebook_column(name, test_col, uniq)[1]
            continue

        # If it is derived keep it that way and don't do anything
        if uniq.size == 0 or is_derived_column(name, uniq):
            kinds.append("derived")
            continue

        # Determine the sentinel codes as specified in spec
        # If the column's largest value has k digits, 7…7 and
        #  9…9 mean don't know/refused, and 8…8 means none
        none_codes, dk_codes = sentinel_codes(uniq)

        # Even if not categorical, map columns with small number of unique values
        # to categorical
        if uniq.size <= cat_max_unique:
            # "None" ranks first so counts keep their order: PHYSHLTH 88 (zero days), 1, 2, ..., 30.
            none = [v for v in uniq if v in none_codes]
            real = sorted(v for v in uniq if v not in dk_codes and v not in none_codes)
            dk = sorted(v for v in uniq if v in dk_codes)
            mapping = {v: i for i, v in enumerate(none + real + dk)}
            for col in (train_col, test_col):
                remapped = np.full_like(col, np.nan)
                for v, i in mapping.items():
                    remapped[col == v] = i
                col[:] = remapped
            kinds.append("categorical")
        else:
            for col in (train_col, test_col):
                for v in uniq:
                    if v in none_codes:
                        col[col == v] = 0
                    elif v in dk_codes:
                        col[col == v] = np.nan
            kinds.append("continuous")
    return x_train, x_test, kinds


def impute_missing(x_train, x_test, kinds):
    x_train = x_train.copy()
    x_test = x_test.copy()
    # Loop through columns
    for j, kind in enumerate(kinds):
        col = x_train[:, j]
        # Select non NaN values to determine fill value
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            fill = 0.0
        elif kind == "categorical":
            vals, counts = np.unique(valid, return_counts=True)
            fill = vals[np.argmax(counts)]
        else:
            fill = np.nanmedian(valid)

        # Set the NaN rows to the fill value
        # This would lose information if we didn't add a missing column
        # per column with NaN values
        x_train[np.isnan(x_train[:, j]), j] = fill
        x_test[np.isnan(x_test[:, j]), j] = fill
    return x_train, x_test


def is_categorical(col, kind, cat_max_unique=55):
    # Some attributes may be integers but classify as categorial variables like
    # the number of states. This is why we set the maximum unique value to 80 to ensure that
    # states and EXTRACT21 values are captured as categorial variables.

    # Actually we set it back to 55 as it had better performance.

    if kind == "categorical":
        return True

    # If non NaN unique value count is less than cat_max_unique and column is derived
    # change to categorical
    uniq = np.unique(col[~np.isnan(col)])
    return kind == "derived" and uniq.size <= cat_max_unique and np.all(uniq == np.round(uniq))


def one_hot(train_col, test_col):
    """0/1 columns, one per level of train_col, with missing as a level of its own.

    The most common level is the reference and gets no column; neither do levels in fewer than MIN_LEVEL_COUNT
    training rows (those rows count as the reference).
    """
    # Set missing to -1 as an indicator for missing value
    train_col, test_col = np.nan_to_num(train_col, nan=-1), np.nan_to_num(test_col, nan=-1)
    
    # Find the unique values in the training column
    levels, counts = np.unique(train_col, return_counts=True)

    # Keep levels seen in at least MIN_LEVEL_COUNT training rows, except the most common one
    levels = levels[(counts >= MIN_LEVEL_COUNT) & (levels != levels[np.argmax(counts)])]
    return train_col[:, None] == levels, test_col[:, None] == levels


def encode(x_train, x_test, kinds, one_hot_encoding=True, missing_indicators=True):
    """Categorical columns one-hot encoded; the others imputed, plus a 0/1 column marking where they were missing.

    Returns float32 matrices: a leading column of ones for the intercept, the numeric columns, then the 0/1
    columns. Exact duplicates among the 0/1 columns are dropped: questions asked together are missing on the same
    rows, and duplicate columns would make tx^T tx singular.

    For the ablation (ablation.py): one_hot_encoding=False keeps the categorical columns as their rank, imputed like
    the numeric ones. missing_indicators=False adds no 0/1 missing columns, and one-hot encoding then counts missing
    as the most common level instead of as a level of its own.
    """
    # Select all categorical and numerical columns
    cat = [j for j, kind in enumerate(kinds) if one_hot_encoding and is_categorical(x_train[:, j], kind)]
    num = [j for j in range(len(kinds)) if j not in cat]
    # Also select any columns with a missing value
    missing = [j for j in num if missing_indicators and np.isnan(x_train[:, j]).any()]

    flags = [(np.isnan(x_train[:, missing]), np.isnan(x_test[:, missing]))]
    if missing_indicators:
        flags += [one_hot(x_train[:, j], x_test[:, j]) for j in cat]
    else:
        # Missing answers count as the most common level, which is the reference and gets no column
        for j in cat:
            train_col, test_col = impute_missing(x_train[:, [j]], x_test[:, [j]], ["categorical"])
            flags.append(one_hot(train_col[:, 0], test_col[:, 0]))
    flags_train, flags_test = np.hstack([f[0] for f in flags]), np.hstack([f[1] for f in flags])
    # np.unique can't take a matrix without columns, which both switches off would give
    first = np.sort(np.unique(flags_train, axis=1, return_index=True)[1]) if flags_train.shape[1] else []
    num_train, num_test = impute_missing(x_train[:, num], x_test[:, num], [kinds[j] for j in num])

    out = []
    for numeric, flag in ((num_train, flags_train), (num_test, flags_test)):
        x = np.empty((len(numeric), 1 + len(num) + len(first)), dtype=np.float32)
        x[:, 0] = 1
        x[:, 1 : 1 + len(num)] = numeric
        x[:, 1 + len(num) :] = flag[:, first]
        out.append(x)
    return out


def standardize(x_train, x_test, chunk=128, start=1):
    """Standardize every column from `start` on (by default all but the leading intercept) with the training mean
    and std.

    In place and a chunk of columns at a time, so the ~1200-column matrices are never copied whole.
    """
    # I think this code is quite self explanatory. The chunking is so that we don't run
    # into memory issues.
    for s in range(start, x_train.shape[1], chunk):
        block_train, block_test = x_train[:, s : s + chunk], x_test[:, s : s + chunk]
        mean, std = block_train.mean(axis=0), block_train.std(axis=0)
        std[std == 0] = 1.0
        block_train -= mean
        block_train /= std
        block_test -= mean
        block_test /= std
    return x_train, x_test


def interaction_pairs(y_train, x_train, n_columns=N_INTERACTION_COLUMNS):
    """Every pair (a, b) of the n_columns columns with the largest |w| in a plain regularized logistic fit.

    Pass only training rows: the choice uses their labels. The intercept (column 0) is never picked.
    """

    # The idea is to look at what weights (dimension) are assigned the most value, so
    # either strongly positive or strongly negative
    w, _ = reg_logistic_regression_newton(
        y_train, x_train, INTERACTION_LAMBDA, np.zeros(x_train.shape[1]), INTERACTION_ITERS
    )
    # Then just select the indices of the top n_columns, and create pairs of them
    top = 1 + np.argsort(-np.abs(w[1:]))[:n_columns]
    return np.array(list(combinations(np.sort(top), 2)))


def add_interactions(x_train, x_test, pairs, chunk=1 << 15):
    """x_train and x_test with the product x[:, a] * x[:, b] of every pair appended, standardized with x_train's
    mean and std (the original columns are left as they are).

    The products are written a chunk of rows at a time straight into the new matrices.
    """
    # The idea is to use the identified 'rich' pairs and combine them to hopefully produce
    # an even stronger signal that the ML model can learn from.
    # Also again, we chunk here to avoid memory limitations
    a, b = pairs[:, 0], pairs[:, 1]
    d = x_train.shape[1]
    out = []
    for x in (x_train, x_test):
        x_new = np.empty((len(x), d + len(pairs)), dtype=x.dtype)
        x_new[:, :d] = x
        for s in range(0, len(x), chunk):
            x_new[s : s + chunk, d:] = x[s : s + chunk, a] * x[s : s + chunk, b]
        out.append(x_new)
    return standardize(out[0], out[1], start=d)


def clean_data(x_train, x_test, y_train, codebook=True, max_nan=1, one_hot_encoding=True, missing_indicators=True):
    """Clean and preprocess the raw dataset.

    Args:
        x_train: shape=(N, D). Raw training features, as returned by load_csv_data.
        x_test: shape=(M, D). Raw test features, as returned by load_csv_data.
        y_train: shape=(N,). Raw training labels, as returned by load_csv_data.
        codebook, max_nan, one_hot_encoding, missing_indicators: switch single steps off for the ablation
            (ablation.py); the defaults are the final pipeline. codebook=False recodes answer codes by the width rule
            alone (recode_sentinels), max_nan=0.5 drops the columns with half or more of their training values
            missing (remove_unimportant_columns), and see encode for the other two.

    Returns:
        x_train: shape=(N, D'). Cleaned/processed float32 training features, with a
            leading column of ones for the intercept.
        x_test: shape=(M, D'). Cleaned/processed float32 test features, using the
            same transformation (e.g. same normalization stats) as x_train.
        y_train: shape=(N,). Labels, recoded to {0, 1} if needed.
    """
    names = list(ALL_COLUMN_NAMES)
    # We iteratively remove and refine the table
    x_train, x_test, names = remove_unimportant_columns(x_train, x_test, names, DROP, max_nan=max_nan)
    x_train, x_test, kinds = recode_sentinels(x_train, x_test, names, codebook=codebook)
    x_train, x_test = encode(x_train, x_test, kinds, one_hot_encoding, missing_indicators)

    # Standardize as a final step to stabilize the inputs
    x_train, x_test = standardize(x_train, x_test)

    y_train = (y_train == 1).astype(int)

    return x_train, x_test, y_train







# def clean_data(x_train, x_test, y_train, n_features=10, seed=0):
#     """Mock of clean_data that ignores the raw values and returns synthetic data.

#     Keeps the row counts N and M of the inputs so test_ids still line up with
#     x_test. Features are standard normal plus a bias column, and labels come
#     from a noisy linear rule so the models have something to learn.

#     Returns:
#         x_train: shape=(N, n_features + 1).
#         x_test: shape=(M, n_features + 1).
#         y_train: shape=(N,). Labels in {0, 1}.
#     """
#     rng = np.random.default_rng(seed)
#     n, m = x_train.shape[0], x_test.shape[0]

#     x_tr = rng.standard_normal((n, n_features))
#     x_te = rng.standard_normal((m, n_features))

#     true_w = rng.standard_normal(n_features)
#     noise = 0.5 * rng.standard_normal(n)
#     y = (x_tr @ true_w + noise > 0).astype(int)

#     x_tr = np.c_[np.ones(n), x_tr]
#     x_te = np.c_[np.ones(m), x_te]
#     return x_tr, x_te, y