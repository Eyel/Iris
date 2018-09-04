#

def array1dto2d(array_1d):
    array_2d = []
    for item in array_1d:
        array_2d.append([item])
    return array_2d

def genlistnum(debut, fin):
    generatedlist = []
    for index in range(debut, fin+1):
        generatedlist.append(index)
    return generatedlist


def backwardelimination(X, y, SL):
    import statsmodels.formula.api as stats

    liste_index = genlistnum(0, len(X[0, :]) - 1)
    X_opt = X[:, liste_index]
    regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()

    pvalues = regressor_ols.pvalues
    old_adjr = 0
    stop = False
    nb = 0
    while (max(pvalues) > SL+0.01) : #and (not stop):
        num = pvalues.argmax(axis=0)
        new_adjr = regressor_ols.rsquared_adj
        print "Del", max(pvalues), num, "Adj Rsq =", new_adjr
        #print "bef pop:", liste_index
        liste_index.pop(num)
        #print "Aft pop:", liste_index
        nb += 1
        X_opt = X[:, liste_index]
        regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
        pvalues = regressor_ols.pvalues

        if old_adjr > new_adjr:
            #stop = True
            print "STOP"
        old_adjr = new_adjr

    print regressor_ols.summary()
    print "nb var del:", nb
    return liste_index

if __name__ == '__main__':
    print genlistnum(0,8)