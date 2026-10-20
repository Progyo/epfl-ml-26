import numpy as np

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


def remove_unimportant_columns(x_train, x_test, names, drop, max_nan=0.5):
    nan_fraction = np.isnan(x_train).mean(axis=0)
    keep = [i for i, n in enumerate(names) if n not in drop and nan_fraction[i] < max_nan]
    return x_train[:, keep], x_test[:, keep], [names[i] for i in keep]


def is_derived_column(name, uniq):
    if name.startswith("_") or name.endswith("_"):
        return True
    return np.any(uniq != np.round(uniq))


def sentinel_codes(uniq):
    width = len(str(int(uniq.max())))
    return {int("8" * width)}, {int("7" * width), int("9" * width)}


def recode_sentinels(x_train, x_test, names, cat_max_unique=55):
    x_train = x_train.copy()
    x_test = x_test.copy()
    kinds = []
    for j, name in enumerate(names):
        train_col, test_col = x_train[:, j], x_test[:, j]
        combined = np.concatenate([train_col, test_col])
        uniq = np.unique(combined[~np.isnan(combined)])

        if uniq.size == 0 or is_derived_column(name, uniq):
            kinds.append("derived")
            continue

        none_codes, dk_codes = sentinel_codes(uniq)

        if uniq.size <= cat_max_unique:
            real = sorted(v for v in uniq if v not in dk_codes)
            dk = sorted(v for v in uniq if v in dk_codes)
            mapping = {v: i for i, v in enumerate(real + dk)}
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
    for j, kind in enumerate(kinds):
        col = x_train[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            fill = 0.0
        elif kind == "categorical":
            vals, counts = np.unique(valid, return_counts=True)
            fill = vals[np.argmax(counts)]
        else:
            fill = np.nanmedian(valid)
        x_train[np.isnan(x_train[:, j]), j] = fill
        x_test[np.isnan(x_test[:, j]), j] = fill
    return x_train, x_test


def standardize(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def clean_data(x_train, x_test, y_train):
    """Clean and preprocess the raw dataset.

    Args:
        x_train: shape=(N, D). Raw training features, as returned by load_csv_data.
        x_test: shape=(M, D). Raw test features, as returned by load_csv_data.
        y_train: shape=(N,). Raw training labels, as returned by load_csv_data.

    Returns:
        x_train: shape=(N, D'). Cleaned/processed training features, with a
            leading column of ones for the intercept.
        x_test: shape=(M, D'). Cleaned/processed test features, using the
            same transformation (e.g. same normalization stats) as x_train.
        y_train: shape=(N,). Labels, recoded to {0, 1} if needed.
    """
    names = list(ALL_COLUMN_NAMES)
    x_train, x_test, names = remove_unimportant_columns(x_train, x_test, names, DROP)
    x_train, x_test, kinds = recode_sentinels(x_train, x_test, names)
    x_train, x_test = impute_missing(x_train, x_test, kinds)
    x_train, x_test = standardize(x_train, x_test)

    # Intercept column, added after standardize so it stays at 1.
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

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