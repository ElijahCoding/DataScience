{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('../data/Position_Salaries.csv')\n",
    "\n",
    "X = dataset.iloc[:, 1:2].values\n",
    "y = dataset.iloc[:, 2].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "sc_X = StandardScaler()\n",
    "sc_y = StandardScaler()\n",
    "\n",
    "X = sc_X.fit_transform(X)\n",
    "y = sc_y.fit_transform([y])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "bad input shape (1, 10)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-47-15913acf3a2a>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mregressor\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mSVR\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkernel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'rbf'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mregressor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m    144\u001b[0m         X, y = check_X_y(X, y, dtype=np.float64,\n\u001b[1;32m    145\u001b[0m                          \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'C'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'csr'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 146\u001b[0;31m                          accept_large_sparse=False)\n\u001b[0m\u001b[1;32m    147\u001b[0m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_validate_targets\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    148\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)\u001b[0m\n\u001b[1;32m    722\u001b[0m                         dtype=None)\n\u001b[1;32m    723\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 724\u001b[0;31m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcolumn_or_1d\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mwarn\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    725\u001b[0m         \u001b[0m_assert_all_finite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    726\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0my_numeric\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mkind\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'O'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcolumn_or_1d\u001b[0;34m(y, warn)\u001b[0m\n\u001b[1;32m    758\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mravel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    759\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 760\u001b[0;31m     \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"bad input shape {0}\"\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    761\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    762\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: bad input shape (1, 10)"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVR\n",
    "\n",
    "regressor = SVR(kernel='rbf')\n",
    "\n",
    "regressor.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "ename": "NotFittedError",
     "evalue": "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNotFittedError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-38-579e6efd6ba5>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mregressor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m6.5\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0my_pred\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py\u001b[0m in \u001b[0;36mpredict\u001b[0;34m(self, X)\u001b[0m\n\u001b[1;32m    320\u001b[0m         \u001b[0my_pred\u001b[0m \u001b[0;34m:\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshape\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mn_samples\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    321\u001b[0m         \"\"\"\n\u001b[0;32m--> 322\u001b[0;31m         \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_validate_for_predict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    323\u001b[0m         \u001b[0mpredict\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sparse_predict\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sparse\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_dense_predict\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    324\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py\u001b[0m in \u001b[0;36m_validate_for_predict\u001b[0;34m(self, X)\u001b[0m\n\u001b[1;32m    449\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    450\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_validate_for_predict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 451\u001b[0;31m         \u001b[0mcheck_is_fitted\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'support_'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    452\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    453\u001b[0m         X = check_array(X, accept_sparse='csr', dtype=np.float64, order=\"C\",\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_is_fitted\u001b[0;34m(estimator, attributes, msg, all_or_any)\u001b[0m\n\u001b[1;32m    912\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    913\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mall_or_any\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mhasattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mestimator\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mattr\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mattr\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mattributes\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 914\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mNotFittedError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmsg\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m'name'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mtype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mestimator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__name__\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    915\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    916\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNotFittedError\u001b[0m: This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
     ]
    }
   ],
   "source": [
    "y_pred = regressor.predict([[6.5]])\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X, y, color='red')\n",
    "plt.plot(X, regressor.predict(X), color='blue')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
