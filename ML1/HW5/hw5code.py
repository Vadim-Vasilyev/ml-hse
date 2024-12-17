import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if (type(feature_vector) != np.ndarray or type(target_vector) != np.ndarray):
        raise ValueError("Feature_vector and target_vactor both should be numpy.ndarray")
    if feature_vector.shape != target_vector.shape:
        raise ValueError(f"Shapes of the input vectors must match. Got {feature_vector.shape} and {target_vector.shape}")
    # if np.all(target_vector == target_vector[0]):
    #     raise ValueError("Can't split since constant target")
    
    sorted_features_ind = np.argsort(feature_vector).astype(int)
    feature_vector_s = feature_vector[sorted_features_ind] # отсортированные значения предиктора
    target_vector_s = target_vector[sorted_features_ind] # отсортированные (соотв. предиктору) значения таргета
    unique_values, value_counts = np.unique(feature_vector_s, return_counts=True) # уникальные значения предиктора и их кол-во
    thresholds = (unique_values[1:] + unique_values[:-1]) / 2 # пороги
    counts_cumsum = np.cumsum(value_counts)

    cumsum = np.cumsum(target_vector_s)
    cumsum_for_unique = cumsum[np.cumsum(value_counts) - 1] # кол-во положительных объектов левее каждого порога
    
    p1_left = cumsum_for_unique[:-1] / counts_cumsum[:-1]
    p1_right = (cumsum_for_unique[-1] - cumsum_for_unique[:-1]) / (counts_cumsum[-1] - counts_cumsum[:-1])

    H_left = 1 - p1_left**2 - (1 - p1_left)**2
    H_right = 1 - p1_right**2 - (1 - p1_right)**2

    left_weights = counts_cumsum[:-1] / counts_cumsum[-1]
    right_weights = 1 - left_weights

    ginis = -left_weights * H_left - right_weights * H_right
    best_ind = np.argmax(ginis)
    threshold_best = thresholds[best_ind]
    gini_best = ginis[best_ind]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = dict(depth=0)
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split 
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if self._max_depth is not None and node["depth"] == self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and sub_X.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_leaf is not None and sub_X.shape[0] // 2 < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        # constant target check
        if np.all(sub_y == sub_y[0]): # sub_y == sub_y[0]
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]): # границы итерирования? почему от 1?
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count # наоборот? current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # в надо брать не ration, а key, т.е. x[0]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2: # надо считать число уникальных значений
                continue

            all_threshols, all_ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            # select best threshold satisfying min_sample_leaf
            if self._min_samples_leaf is not None:
                left_sizes = (feature_vector[:, None] < all_threshols).sum(axis=0) # left subsample size denending on threshold
                right_sizes = len(feature_vector) - left_sizes # right subsample size denending on threshold
                all_ginis[np.where((left_sizes < self._min_samples_leaf) | (right_sizes < self._min_samples_leaf))] = -np.inf
                gini = max(all_ginis)
                threshold = all_threshols[np.argmax(all_ginis)]
                if gini == -np.inf:
                    continue
                
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        # print(feature_best)
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # сохранял какую-то фигню
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            # print(threshold_best)
        else:
            raise ValueError
            
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {"depth": node["depth"] + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"]) # пропущен logical_not для split

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split_value = x[node["feature_split"]]
        if self._feature_types[node["feature_split"]] == "real":
            if feature_split_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        elif self._feature_types[node["feature_split"]] == "categorical":
            if np.any(feature_split_value == node["categories_split"]):
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        if type(self._min_samples_split) is float:
            self._min_samples_split = np.ceil(self._min_samples_split * X.shape[0])

        if type(self._min_samples_leaf) is float:
            self._min_samples_leaf = np.ceil(self._min_samples_leaf * X.shape[0])
            
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

def best_regression_split(X, y, feature, thresholds_number=99):
    """
    X - все фичи
    y - таргет
    feature - фича, по которой будем выбирать порог для разбиения
    thresholds_number - на сколько квантилей разбивать данные по фиче
    
    Все фичи передаются для того, чтобы при рассмотрении разбиения строить регрессию не только по той фиче,
    которая используется для разбиения, но по всем. Это делается потому, что нас интересует не влияние конкретной фичи (feature)
    на таргет, а минимизация MSE для итоговой модели, которая окажется в листе и которая будет использовать все фичи.
    """
    feature_vector = X[:, feature]
    target_vector = y
    
    if (type(feature_vector) != np.ndarray or type(target_vector) != np.ndarray):
        raise ValueError("Feature_vector and target_vactor both should be numpy.ndarray")
    if feature_vector.shape != target_vector.shape:
        raise ValueError(f"Shapes of the input vectors must match. Got {feature_vector.shape} and {target_vector.shape}")
    if np.all(target_vector == target_vector[0]):
        raise ValueError("Can't split since constant target")

    sorted_features_ind = np.argsort(feature_vector).astype(int)
    feature_vector_s = feature_vector[sorted_features_ind] # отсортированные значения предиктора
    target_vector_s = target_vector[sorted_features_ind] # отсортированные (соотв. предиктору) значения таргета

    thresholds = np.quantile(feature_vector_s, np.linspace(0.01, 0.99, thresholds_number))
    mask_by_threshold = feature_vector[:, None] < thresholds

    losses = []
    for i in range(thresholds_number):
        reg_left = LinearRegression()
        reg_right = LinearRegression()
        
        left_mask = mask_by_threshold[:, i]
        right_mask = np.logical_not(mask_by_threshold[:, i])
        left_X, left_target = X[left_mask, :], target_vector[left_mask].reshape(-1, 1)
        right_X, right_target = X[right_mask, :], target_vector[right_mask].reshape(-1, 1)

        if left_mask.sum() < 1 or right_mask.sum() < 1:
            losses.append(np.inf)
            continue
        
        reg_left.fit(left_X, left_target)
        reg_right.fit(right_X, right_target)

        loss_left = mean_squared_error(reg_left.predict(left_X), left_target)
        loss_right = mean_squared_error(reg_right.predict(right_X), right_target)
        loss = (loss_left * left_X.shape[0] + loss_right * right_X.shape[0]) / len(feature_vector)
        losses.append(loss)

    best_ind = np.argmin(losses)
    best_threshold = thresholds[best_ind]
    best_loss = losses[best_ind]
    return thresholds, np.array(losses), best_threshold, best_loss
    

class LinearRegressionTree():
    def __init__(self, feature_types, max_depth, min_samples_split=None, min_samples_leaf=None, number_of_quantiles=10):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = dict(depth=0)
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        self._number_of_quantiles = number_of_quantiles

    def _fit_node(self, sub_X, sub_y, node) -> None:
        if (self._max_depth == node["depth"]) or \
            (self._min_samples_split is not None and sub_X.shape[0] < self._min_samples_split) or \
            (node["depth"] >= self._max_depth or np.all(sub_y == sub_y[0])) or \
            (self._min_samples_leaf is not None and sub_X.shape[0] // 2 < self._min_samples_leaf):
                node["type"] = "terminal"
                reg = LinearRegression()
                reg.fit(sub_X, sub_y)
                node["model"] = reg
                return
            
        feature_best, threshold_best, loss_best, split = None, None, None, None
        for f in range(sub_X.shape[1]):
            thresholds_number = len(np.unique(sub_X[:, f])) if self._feature_types[f] == "categorical" else self._number_of_quantiles
            all_thresholds, all_losses, threshold, loss = best_regression_split(sub_X, sub_y, feature=f, thresholds_number=thresholds_number)

            if loss_best == np.inf:
                continue

            # select best threshold satisfying min_sample_leaf
            if self._min_samples_leaf is not None:
                left_sizes = (sub_X[:, f][:, None] < all_thresholds).sum(axis=0) # left subsample size denending on threshold
                right_sizes = (sub_X[:, f][:, None] >= all_thresholds).sum(axis=0) # right subsample size denending on threshold
                all_losses[np.where((left_sizes < self._min_samples_leaf) | (right_sizes < self._min_samples_leaf))] = np.inf
                loss = min(all_losses)
                threshold = all_thresholds[np.argmin(all_losses)]
                if loss == np.inf:
                    continue
            
            if loss_best is None or loss_best > loss:
                feature_best = f
                threshold_best = threshold
                loss_best = loss
                split = sub_X[:, feature_best] < threshold

        if loss_best == np.inf or loss_best == None:
            node["type"] = "terminal"
            reg = LinearRegression()
            reg.fit(sub_X, sub_y)
            node["model"] = reg
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {"depth": node["depth"] + 1}
        self._fit_node(sub_X[split, :], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split), :], sub_y[np.logical_not(split)], node["right_child"])
            

    def fit(self, X, y):
        if type(self._min_samples_split) is float:
            self._min_samples_split = np.ceil(self._min_samples_split * X.shape[0])

        if type(self._min_samples_leaf) is float:
            self._min_samples_leaf = np.ceil(self._min_samples_leaf * X.shape[0])
            
        self._fit_node(X, y, self._tree)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]

        if x[node["feature_split"]] < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        return self._predict_node(x, node["right_child"])
        
    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)