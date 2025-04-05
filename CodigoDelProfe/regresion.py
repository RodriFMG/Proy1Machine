# Librerías de manejo de datos
import pandas as pd
import numpy as np

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import pylab as pl

# Librerías de aprendizaje automático
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

# Técnicas de reducción de dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Herramientas de optimización y matemáticas
import scipy.optimize as opt

# Visualización 3D
from mpl_toolkits.mplot3d import Axes3D



def generate_data_set():
    """ Generates Random Data
    Returns
    -------
    x : array-like, shape = [n_samples, n_features]
            Training samples
    y : array-like, shape = [n_samples, n_target_values]
            Target values
    """
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 3 * x + np.random.rand(100, 1) + 2
    return x, y


def scatter_plot(x, y, size=10, x_label='x', y_label='y', color='b'):
    plt.scatter(x, y, s=size, color=color)
    set_labels(x_label, y_label)


def plot(x, y, x_label='x', y_label='y', color='r'):
    plt.plot(x, y, color=color)
    set_labels(x_label, y_label)


def ploty(y, x_label='x', y_label='y'):
    plt.plot(y)
    set_labels(x_label, y_label)


def set_labels(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


class PerformanceMetrics:
    """Defines methods to evaluate the model
    Parameters
    ----------
    y_actual : array-like, shape = [n_samples]
            Observed values from the training samples
    y_predicted : array-like, shape = [n_samples]
            Predicted values from the model
    """

    def __init__(self, y_actual, y_predicted):
        self.y_actual = y_actual
        self.y_predicted = y_predicted

    def compute_rmse(self):
        """Compute the root mean squared error
        Returns
        ------
        rmse : root mean squared error
        """
        return np.sqrt(self.sum_of_square_of_residuals())

    def compute_r2_score(self):
        """Compute the r-squared score
            Returns
            ------
            r2_score : r-squared score
            """
        # sum of square of residuals
        ssr = self.sum_of_square_of_residuals()

        # total sum of errors
        sst = np.sum((self.y_actual - np.mean(self.y_actual)) ** 2)

        return 1 - (ssr / sst)

    def sum_of_square_of_residuals(self):
        return np.sum((self.y_actual - self.y_predicted) ** 2)


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations


    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals) / m
            self.w_ -= self.eta * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_)


def whole():
    # generate random data-set
    linear_regression_model = LinearRegressionUsingGD()

    # generate the data set
    x, y = generate_data_set()

    # transform the feature vectors to include the bias term
    # adding 1 to all the instances of the training set.
    m = x.shape[0]
    x_train = np.c_[np.ones((m, 1)), x]

    # fit/train the model
    linear_regression_model.fit(x_train, y)

    # predict values
    predicted_values = linear_regression_model.predict(x_train)

    # model parameters
    print(linear_regression_model.w_)
    intercept, coeffs = linear_regression_model.w_

    # cost_function
    cost_function = linear_regression_model.cost_

    # plotting
    scatter_plot(x, y)
    plot(x, predicted_values)
    ploty(cost_function, 'no of iterations', 'cost function')

    # computing metrics
    metrics = PerformanceMetrics(y, predicted_values)
    rmse = metrics.compute_rmse()
    r2_score = metrics.compute_r2_score()

    print('The coefficient is {}'.format(coeffs))
    print('The intercept is {}'.format(intercept))
    print('Root mean squared error of the model is {}.'.format(rmse))
    print('R-squared score is {}.'.format(r2_score))


################################################################################
################################################################################
################################################################################
def whole_():
    # generate random data-set
    linear_regression_model = LinearRegressionUsingGD()

    # generate the data set
    x, y = generate_data_set()

    # transform the feature vectors to include the bias term
    # adding 1 to all the instances of the training set.
    m = x.shape[0]
    x_train = np.c_[np.ones((m, 1)), x]

    # b = (x'x)^-1 x'y

    X = np.matrix(x_train)
    X_t = np.transpose(np.matrix(x_train))
    prod = np.matmul(X_t, X)
    iprod = np.linalg.inv(prod)
    prod2 = np.matmul(iprod, X_t)
    b = np.matmul(prod2, y)

    # fit/train the model
    linear_regression_model.fit(x_train, y)

    # predict values
    predicted_values = linear_regression_model.predict(x_train)

    # model parameters
    print(linear_regression_model.w_)
    intercept, coeffs = linear_regression_model.w_

    # cost_function
    cost_function = linear_regression_model.cost_

    # plotting
    scatter_plot(x, y)
    plot(x, predicted_values)
    ploty(cost_function, 'no of iterations', 'cost function')

    # computing metrics
    metrics = PerformanceMetrics(y, predicted_values)
    rmse = metrics.compute_rmse()
    r2_score = metrics.compute_r2_score()

    print('The coefficient is {}'.format(coeffs))
    print('The intercept is {}'.format(intercept))
    print('Root mean squared error of the model is {}.'.format(rmse))
    print('R-squared score is {}.'.format(r2_score))


################################################################################
################################################################################
################################################################################
def whole2():
    # generate random data-set
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)

    # sckit-learn implementation

    # Model initialization
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(x, y)

    # Predict
    y_predicted = regression_model.predict(x)

    # model evaluation
    rmse = mean_squared_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)

    # printing values
    print('Slope:', regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # plotting values

    # data points
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')

    # predicted values
    plt.plot(x, y_predicted, color='r')
    plt.show()


################################################################################
################################################################################
################################################################################
def ejemplo_linea():
    x = np.array([1, 2, 3, 4], dtype='float')
    y = x
    w = -5
    m = 4
    a = 0.05
    ls = 1e5
    i = 0

    v1 = []
    v2 = []

    while ls > 0.005:
        hx = w * x
        ls = 1 / (2 * m) * ((hx - y) ** 2).sum()
        sl = 1 / m * ((hx - y) * x).sum()
        w = w - a * sl
        i += 1
        v1.append(ls)
        v2.append(sl)

    plt.plot(v2, v1)
    plt.show()
    print(i)
    print(w)


################################################################################
################################################################################
################################################################################

def ejemplo_vino():
    dataset = pd.read_csv('./db/wine.csv')

    # eliminar los nulos
    dataset.isnull().any()
    dataset = dataset.fillna(method='ffill')

    # llenar los datos de las columnas
    cnames = ['fixed acidity', 'volatile acidity', 'citric acid',
              'residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    X = dataset[cnames].values
    y = dataset['quality'].values

    # plt.figure(figsize=(15,10))
    # plt.tight_layout()
    # seabornInstance.distplot(dataset['quality'])
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, cnames, columns=['Coefficient'])

    print(coeff_df)
    # his means that for a unit increase in “density”, there is a decrease of 31.51 units in the quality of the wine.
    # Similarly, a unit decrease in “Chlorides“ results in an increase of 1.87 units in the quality of the wine.
    # We can see that the rest of the features have very little effect on the quality of the wine.
    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    df1 = df.head(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def ejemplo_churn():
    churn_df = pd.read_csv("./db/ChurnData.csv")
    print(churn_df.head())

    churn_df = churn_df[
        ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    print(churn_df.head())
    print(churn_df.describe())

    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    y = np.asarray(churn_df['churn'])

    X_ = TSNE(n_components=2).fit_transform(X)

    for i in X_:
        plt.scatter(*i)
    plt.show()

    # X_ = TSNE(n_components=3).fit_transform(X)

    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')

    # for i in X_:
    #    ax.scatter(*i)
    # plt.show()

    X = preprocessing.StandardScaler().fit(X).transform(X)  # z = (x - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    LR = LogisticRegression(C=0.001, solver='liblinear').fit(X_train, y_train)

    yhat = LR.predict(X_test)

    print(yhat)

    yhat_prob = LR.predict_proba(X_test)
    print(yhat_prob)

    # print(jaccard_similarity_score(y_test, yhat))

    # Calcular la matriz de confusión
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Matriz de confusión')

    print(classification_report(y_test, yhat))
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
  Esta función muestra y dibuja la matriz de confusión.
  La normalización se puede aplicar estableciendo el valor `normalize=True`.
  """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalización')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')


################################################################################
################################################################################
################################################################################

################################################################################   
################################################################################
################################################################################
def ejemplo_vino_pca():
    dataset = pd.read_csv('./db/wine.csv')

    # eliminar los nulos
    dataset.isnull().any()
    dataset = dataset.fillna(method='ffill')

    # llenar los datos de las columnas
    cnames = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    X = dataset[cnames].values
    y = dataset['quality'].values

    pca = PCA(n_components=4)
    pca.fit(X)

    print(X)
    X = pca.transform(X)

    print(X)

    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(dataset['quality'])

    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # coeff_df = pd.DataFrame(regressor.coef_, cnames, columns=['Coefficient'])

    # his means that for a unit increase in “density”, there is a decrease of 31.51 units in the quality of the wine.
    # Similarly, a unit decrease in “Chlorides“ results in an increase of 1.87 units in the quality of the wine.
    # We can see that the rest of the features have very little effect on the quality of the wine.

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    df1 = df.head(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

################################################################################   
################################################################################
################################################################################
